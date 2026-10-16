"""Complete Physics checkpoint loading and frozen, deployable policy state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from .algorithms.dynamics_history import snapshot_physics
from .holomotion_original_algorithm import (
    flat_observations_to_tensordict,
)
from .k1_policy_backends import build_k1_networks
from .holomotion_physics_actor import (
    HoloMotionPhysicsActor, HoloMotionPhysicsActorConfig, dynamics_actor_features,
    teacher_actor_features,
)
from .privileged_teacher import (
    PrivilegedTeacherConfig, PrivilegedTransformerTeacher, teacher_inputs,
    teacher_shared_critic_inputs, normalized_core,
)
from .models.physics import DynamicsModel
from .robot_contract import RobotContract
from .physics_confidence import PhysicsConfidence, PhysicsConfidenceConfig


def dynamics_configuration(model):
    graph = model.encoder.graph
    return dict(robot=asdict(model.robot), physics_width=model.physics_width,
        memory_width=model.memory_width, latent_width=model.latent_width,
        hidden_dim=model.hidden_dim, check_values=model.check_values,
        interaction_feedback=model.interaction_feedback,
        scene_encoder=model.encoder.scene_encoder,
        graph_layers=2 if graph is None else graph.gnn.n_layers,
        contact_object_indices=None if graph is None else list(graph.mapping),
        system_id_dim=model.system_id_dim, teacher_student=model.teacher_student,
        prediction_horizons=list(model.prediction_horizons), uncertainty=model.uncertainty,
        mask_unobserved_core=model.mask_unobserved_core)


@torch.no_grad()
def physics_policy_inputs(model, frame, starts, env_ids, memory, age, warmup, calibration=None):
    """Training and inference share the exact EMA feature/confidence calculation."""
    encoding, memory = model.step(frame, episode_start=starts, env_ids=env_ids, memory=memory)
    feature = dynamics_actor_features(encoding.context, encoding.latent, encoding.contact_tokens)
    _, logvar = model.identify_privileged_distribution(encoding.latent)
    confidence = (torch.zeros_like(logvar[..., :1]) if calibration is None
                  else calibration(logvar, age, warmup))
    return encoding, memory, feature, confidence


@dataclass
class PhysicsCheckpoint:
    actor: torch.nn.Module
    critic: torch.nn.Module
    dynamics: DynamicsModel
    actor_dynamics: DynamicsModel
    calibration: PhysicsConfidence
    payload: dict
    privileged_teacher: PrivilegedTransformerTeacher | None = None


def load_physics_checkpoint(checkpoint, *, device="cpu", holomotion_root=None):
    """Load all branches into new modules; never silently substitute a base actor.

    Schema 5 contains the corrected model and calibration. Schema 4 reproduces
    legacy inference. Schema 3 lacks residual limits and the resolved network
    config; only its original trainer/configuration can migrate it to schema 4.
    """
    # Keep training-only Teacher/optimizer tensors off the inference device.
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get("physics_schema") not in (4, 5):
        raise ValueError("complete Physics loading requires schema 4 or 5; re-save a schema 3 "
                         "checkpoint using its original pre-schema-5 trainer/configuration")
    legacy = payload["physics_schema"] == 4
    required = {"physics_runtime_config", "base_actor_state_dict", "critic_state_dict",
        "physics_residual_state_dict", "dynamics_state_dict", "actor_dynamics_state_dict",
        "physics_training_config", "physics_loss_weights", "ppo_config", "iter"}
    if required - payload.keys():
        raise ValueError(f"incomplete Physics checkpoint: {sorted(required - payload.keys())}")
    if type(payload["iter"]) is not int or payload["iter"] < 0:
        raise ValueError("invalid checkpoint iteration")
    runtime = payload["physics_runtime_config"]
    teacher_config = runtime.get('privileged_teacher_config')
    teacher = payload.get('privileged_teacher')
    if (teacher_config is None) != (teacher is None):
        raise ValueError('incomplete privileged Transformer Teacher checkpoint')
    if teacher is not None:
        parsed_teacher_config = PrivilegedTeacherConfig(**teacher_config)
        expected_teacher_schema = (3 if parsed_teacher_config.objective_version >= 5 else
                                   2 if parsed_teacher_config.objective_version >= 4 else 1)
        if (teacher.get('schema') != expected_teacher_schema or teacher.get('config') != teacher_config
                or {'model', 'ema', 'optimizer', 'updates'} - teacher.keys()):
            raise ValueError('invalid privileged Transformer Teacher checkpoint')
    if runtime.get("schema") != (1 if legacy else 2):
        raise ValueError("unsupported Physics runtime schema")
    if not legacy and ("confidence_config" not in runtime or "confidence_state_dict" not in payload):
        raise ValueError("incomplete Physics checkpoint: confidence calibration missing")
    for key in ("confidence_warmup_steps", "rollout_steps"):
        if type(runtime.get(key)) is not int or runtime[key] < 1:
            raise ValueError(f"invalid runtime {key}")
    if runtime["confidence_warmup_steps"] != payload["physics_training_config"]["confidence_warmup_steps"]:
        raise ValueError("checkpoint confidence configurations disagree")
    base_actor, critic = build_k1_networks(
        runtime["network_config"], device=device, holomotion_root=holomotion_root)
    config = dict(runtime["dynamics_config"])
    if legacy:
        config.setdefault("mask_unobserved_core", False)
    robot = RobotContract(**config.pop("robot"))
    dynamics = DynamicsModel(robot, **config).to(device)
    ema = DynamicsModel(robot, **config).to(device)
    control_mode = runtime.get("physics_control_mode", "student")
    if control_mode == "teacher_observation":
        if runtime.get("physics_actor_config") is not None:
            raise ValueError("Teacher-observation checkpoint must not contain a residual config")
        if payload["physics_residual_state_dict"]:
            raise ValueError("Teacher-observation checkpoint must not contain residual weights")
        if getattr(getattr(base_actor, "config", None), "teacher_latent_dim", 0) != dynamics.latent_width:
            raise ValueError("Actor and Teacher latent widths disagree")
        teacher_contract = PrivilegedTeacherConfig(**(teacher_config or {}))
        if teacher_contract.objective_version < 5:
            raise ValueError("Teacher-observation checkpoint requires the shared Critic/Teacher v5 contract")
        if getattr(critic, "privileged_dim", 0) != teacher_contract.critic_privileged_dim(
                dynamics.system_id_dim):
            raise ValueError("Critic and Teacher physical privilege widths disagree")
        actor = base_actor
    else:
        actor_config = dict(runtime["physics_actor_config"])
        if legacy:
            actor_config.setdefault("condition_on_base_action", False)
        actor = HoloMotionPhysicsActor(
            base_actor, HoloMotionPhysicsActorConfig(**actor_config)).to(device)
    expected_feature_dim = (3*dynamics.physics_width+dynamics.latent_width
                            if control_mode == "student"
                            else 9+2*robot.dof+(teacher_config or {}).get("contact_slots", 2)
                            if control_mode == "teacher" else None)
    if expected_feature_dim is None and control_mode != "teacher_observation":
        raise ValueError(f"unsupported Physics control mode: {control_mode!r}")
    if expected_feature_dim is not None and actor.config.physics_feature_dim != expected_feature_dim:
        raise ValueError("Physics Actor and control feature widths disagree")
    action_dim = (actor.config.action_dim if hasattr(actor.config, "action_dim") else 22)
    if action_dim != robot.dof or action_dim != 22:
        raise ValueError("Physics Actor and K1 action contracts disagree")
    if not dynamics.uncertainty or not dynamics.system_id_dim:
        raise ValueError("Physics confidence requires uncertainty and system identification")
    base_actor.load_state_dict(payload["base_actor_state_dict"], strict=True)
    critic.load_state_dict(payload["critic_state_dict"], strict=True)
    if control_mode != "teacher_observation":
        actor.physics_residual.load_state_dict(payload["physics_residual_state_dict"], strict=True)
    dynamics.load_state_dict(payload["dynamics_state_dict"], strict=True)
    ema.load_state_dict(payload["actor_dynamics_state_dict"], strict=True)
    privileged_teacher = None
    if teacher is not None:
        privileged_teacher = PrivilegedTransformerTeacher(
            robot, dynamics.latent_width, dynamics.system_id_dim,
            PrivilegedTeacherConfig(**teacher_config)).to(device)
        privileged_teacher.load_state_dict(teacher["ema"], strict=True)
    calibration = PhysicsConfidence(dynamics.system_id_dim,
        PhysicsConfidenceConfig(mode="legacy") if legacy else
        PhysicsConfidenceConfig(**runtime["confidence_config"])).to(device)
    if not legacy:
        calibration.load_state_dict(payload["confidence_state_dict"], strict=True)
        if (calibration.samples < 0 or (calibration.mse < 0).any()
                or (calibration.temperature < 1).any()
                or ((calibration.coverage < 0) | (calibration.coverage > 1)).any()):
            raise ValueError("invalid confidence calibration state")
    for module in (actor, critic, dynamics, ema, calibration, privileged_teacher):
        if module is None:
            continue
        for value in module.state_dict().values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                raise ValueError("Physics checkpoint contains non-finite weights")
        module.eval()
    return PhysicsCheckpoint(actor, critic, dynamics, ema, calibration, payload, privileged_teacher)


def state_fingerprint(*modules):
    """Hash parameters, persistent buffers and model contract metadata."""
    digest = hashlib.sha256()
    for module in modules:
        for name, value in sorted(module.state_dict().items()):
            digest.update(name.encode())
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def validate_task_contract(runtime, task):
    saved = runtime.get("task_contract")
    if saved is not None:
        actual = getattr(getattr(task, "adapter", None), "contract", None)
        if actual is None or asdict(actual) != saved:
            raise ValueError("task joint order / PD / observation contract differs from checkpoint")


class FrozenPhysicsPolicy:
    """No optimizers or Teacher targets; only observed history changes at runtime."""
    def __init__(self, actor, actor_dynamics, *, confidence_warmup_steps, rollout_steps,
                 task_contract=None, calibration=None, ablation="full"):
        if not isinstance(actor, HoloMotionPhysicsActor):
            raise TypeError("a complete HoloMotionPhysicsActor is required")
        if any(type(n) is not int or n < 1 for n in (confidence_warmup_steps, rollout_steps)):
            raise ValueError("warmup and rollout steps must be positive")
        self.actor = actor.eval().requires_grad_(False)
        self.dynamics = actor_dynamics.eval().requires_grad_(False)
        self.calibration = (calibration if calibration is not None else
            PhysicsConfidence(actor_dynamics.system_id_dim).to(next(actor.parameters()).device)).eval()
        self.confidence_warmup_steps = confidence_warmup_steps
        self.rollout_steps = rollout_steps
        self.task_contract = task_contract
        if ablation not in ("full", "no_physics", "no_latent", "shuffle_latent", "no_history"):
            raise ValueError("unknown frozen Physics ablation")
        self.ablation = ablation
        self.reset()

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        ablation = kwargs.pop("ablation", "full")
        loaded = load_physics_checkpoint(checkpoint, **kwargs)
        config = loaded.payload["physics_runtime_config"]
        return cls(loaded.actor, loaded.actor_dynamics,
            confidence_warmup_steps=config["confidence_warmup_steps"],
            rollout_steps=config["rollout_steps"], task_contract=config.get("task_contract"),
            calibration=loaded.calibration, ablation=ablation)

    def reset(self):
        self.memory = None
        self.age = None
        self.pending = None
        self.steps = 0

    @property
    def frozen_modules(self):
        return (self.actor, self.dynamics, self.calibration)

    needs_physics = True
    policy_name = "full_physics_ema_residual"

    @torch.no_grad()
    def act(self, actor_observation, physics_observation, episode_start):
        if self.pending is not None:
            raise RuntimeError("record the executed action before the next observation")
        self.actor.eval(); self.dynamics.eval()
        batch = actor_observation.shape[0]
        if episode_start.shape != (batch,) or episode_start.dtype != torch.bool:
            raise ValueError("episode_start must be a boolean [B] tensor")
        if self.age is None:
            if not episode_start.all().item():
                raise ValueError("a newly loaded policy must start at real episode boundaries")
            self.age = torch.zeros(batch, dtype=torch.long, device=actor_observation.device)
        if self.age.shape != (batch,):
            raise ValueError("batch identity changed; explicitly reset the runtime")
        if self.steps % self.rollout_steps == 0:
            self.actor.reset_kv_cache(batch, actor_observation.device)
        elif episode_start.any().item():
            self.actor.clear_env_cache(episode_start.nonzero(as_tuple=False).flatten())
        self.age = torch.where(episode_start, 0, self.age)
        frame = snapshot_physics(physics_observation)
        encoding, memory, feature, confidence = physics_policy_inputs(
            self.dynamics, frame,
            torch.ones_like(episode_start) if self.ablation == "no_history" else episode_start,
            tuple(range(batch)), None if self.ablation == "no_history" else self.memory,
            self.age, self.confidence_warmup_steps, self.calibration)
        if self.ablation == "no_physics":
            confidence = torch.zeros_like(confidence)
        elif self.ablation in ("no_latent", "shuffle_latent"):
            begin, end = self.dynamics.physics_width, self.dynamics.physics_width+self.dynamics.latent_width
            feature = feature.clone()
            if self.ablation == "shuffle_latent":
                if batch < 2:
                    raise ValueError("shuffle_latent requires at least two parallel environments")
                feature[:, begin:end] = feature[:, begin:end].roll(1, dims=0)
            else:
                feature[:, begin:end] = 0.
        data = flat_observations_to_tensordict(actor_observation,
                                              actor_observation.new_zeros((batch, 843)))
        result = self.actor(data, mode="inference", update_obs_norm=False,
                            physics_features=feature, physics_confidence=confidence)
        action = result.get("actions")
        if action.shape != (batch, self.actor.config.action_dim) or not torch.isfinite(action).all():
            raise RuntimeError("frozen Physics policy produced invalid actions")
        self.pending = (frame, encoding, memory)
        return action

    @torch.no_grad()
    def record_action(self, executed_action):
        if self.pending is None:
            raise RuntimeError("act() must precede record_action()")
        frame, encoding, memory = self.pending
        prediction = self.dynamics.predict_next(frame, executed_action, encoding.latent,
                                                physics_encoding=encoding.physics_encoding())
        self.memory = self.dynamics.remember_prediction(memory, prediction)
        self.age += 1
        self.steps += 1
        self.pending = None


class FrozenTeacherPhysicsPolicy:
    """Privileged frozen evaluator for the Teacher-controlled residual.

    It is intentionally simulator-only: every step reads the Critic observation,
    exact randomized parameters and privileged physical state.  The class exists
    to answer whether a good Teacher can make the residual useful before any
    Student distillation is attempted.
    """
    needs_physics = True
    needs_privileged_teacher = True
    policy_name = "privileged_teacher_residual"

    def __init__(self, actor, teacher, *, rollout_steps, task_contract=None,
                 ablation="full"):
        if teacher is None:
            raise ValueError("Teacher-controlled checkpoint has no privileged Teacher")
        if ablation not in ("full", "no_physics"):
            raise ValueError("Teacher-controlled evaluation supports only full/no_physics")
        self.actor = actor.eval().requires_grad_(False)
        self.teacher = teacher.eval().requires_grad_(False)
        self.rollout_steps = rollout_steps
        self.task_contract = task_contract
        self.ablation = ablation
        self.reset()

    @property
    def frozen_modules(self):
        return (self.actor, self.teacher)

    def reset(self):
        self.steps = 0
        self.history = ([], [], [], [])
        self.pending = False

    @torch.no_grad()
    def act_privileged(self, observation, task, episode_start):
        if self.pending:
            raise RuntimeError("record the executed action before the next observation")
        batch = observation.actor.shape[0]
        if episode_start.shape != (batch,) or episode_start.dtype != torch.bool:
            raise ValueError("episode_start must be a boolean [B] tensor")
        if self.steps % self.rollout_steps == 0:
            self.actor.reset_kv_cache(batch, observation.actor.device)
        elif episode_start.any().item():
            self.actor.clear_env_cache(episode_start.nonzero(as_tuple=False).flatten())
        data = flat_observations_to_tensordict(
            observation.actor, observation.critic)
        base = self.actor.base_actor(data, mode="inference", update_obs_norm=False)
        truth = task.privileged_physics_observation()
        parameters = task.physics_teacher_target()
        packed = teacher_inputs(observation.critic, truth, parameters,
                                self.teacher.robot, self.teacher.config)
        current = normalized_core(truth, self.teacher.robot)
        inputs, starts, actions, cores = self.history
        inputs.append(packed); starts.append(episode_start)
        actions.append(base.get("actions")); cores.append(current)
        for values in self.history:
            del values[:-self.teacher.config.context_steps]
        output = self.teacher(torch.stack(inputs, 1), torch.stack(starts, 1),
                              torch.stack(actions, 1), torch.stack(cores, 1))
        feature = teacher_actor_features(
            {key: value[:, -1] for key, value in output.items()},
            dof=self.teacher.robot.dof, contact_slots=self.teacher.config.contact_slots)
        confidence = feature.new_zeros(batch, 1) if self.ablation == "no_physics" else feature.new_ones(batch, 1)
        # apply_physics mutates the TensorDict in place, so preserve the true
        # pre-residual command before asking it to write corrected actions.
        base_action = base.get("actions").detach().clone()
        result = self.actor.apply_physics(
            base, feature, confidence, mode="inference")
        action = result.get("actions")
        if action.shape != (batch, self.actor.config.action_dim) or not torch.isfinite(action).all():
            raise RuntimeError("frozen Teacher policy produced invalid actions")
        self.last_base_action = base_action
        self.last_residual = (action - base_action).detach().clone()
        self.pending = True
        return action

    def record_action(self, executed_action):
        if not self.pending:
            raise RuntimeError("act_privileged() must precede record_action()")
        actions = self.history[2]
        if not actions:
            raise RuntimeError("Teacher action history was not initialized")
        candidate = actions[-1]
        if (not isinstance(executed_action, torch.Tensor)
                or executed_action.shape != candidate.shape
                or executed_action.device != candidate.device
                or executed_action.dtype != candidate.dtype
                or not torch.isfinite(executed_action).all().item()):
            raise ValueError("executed Teacher action must be finite and match the candidate action")
        # Keep the candidate base action for this step's prediction, then replace
        # it with the accepted command so the next prediction sees causal history.
        actions[-1] = executed_action.detach().clone()
        self.steps += 1
        self.pending = False


class FrozenTeacherObservationPolicy:
    """Frozen Actor whose observation is augmented by privileged Teacher z."""

    needs_physics = False
    needs_privileged_teacher = True
    policy_name = "teacher_latent_observation"

    def __init__(self, actor, teacher, *, rollout_steps, task_contract=None,
                 ablation="full"):
        if ablation not in ("full", "no_physics"):
            raise ValueError("Teacher-observation evaluation supports full/no_physics")
        self.actor = actor.eval().requires_grad_(False)
        self.teacher = teacher.eval().requires_grad_(False)
        self.rollout_steps = rollout_steps
        self.task_contract = task_contract
        self.ablation = ablation
        self.reset()

    @property
    def frozen_modules(self):
        return (self.actor, self.teacher)

    def reset(self):
        self.steps = 0
        self.history = ([], [])

    @torch.no_grad()
    def act_privileged(self, observation, task, episode_start):
        batch = observation.actor.shape[0]
        if episode_start.shape != (batch,) or episode_start.dtype != torch.bool:
            raise ValueError("episode_start must be a boolean [B] tensor")
        if self.steps % self.rollout_steps == 0:
            self.actor.reset_kv_cache(batch, observation.actor.device)
        elif episode_start.any().item():
            self.actor.clear_env_cache(episode_start.nonzero(as_tuple=False).flatten())
        truth, contacts = task.privileged_teacher_observation()
        packed = teacher_shared_critic_inputs(
            observation.critic, truth, contacts, task.physics_teacher_target(),
            self.teacher.config)
        inputs, starts = self.history
        inputs.append(packed); starts.append(episode_start)
        del inputs[:-self.teacher.config.context_steps]
        del starts[:-self.teacher.config.context_steps]
        latent = self.teacher.encode(
            torch.stack(inputs, 1), torch.stack(starts, 1))[:, -1]
        if self.ablation == "no_physics":
            latent = torch.zeros_like(latent)
        data = flat_observations_to_tensordict(observation.actor, observation.critic)
        action = self.actor(data, mode="inference", update_obs_norm=False,
                            teacher_latent=latent).get("actions")
        if action.shape != (batch, 22) or not torch.isfinite(action).all():
            raise RuntimeError("frozen Teacher-observation policy produced invalid actions")
        self.steps += 1
        return action

    def record_action(self, executed_action):
        # z_t is computed before a_t and does not consume action history.
        return None


@torch.no_grad()
def evaluate_physics_policy(policy, task, *, steps: int, stop_on_first_done: bool = False):
    """Bounded deterministic evaluation; never calls PPO, Dynamics update or Teacher."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be positive")
    if type(stop_on_first_done) is not bool:
        raise TypeError("stop_on_first_done must be boolean")
    before = state_fingerprint(*policy.frozen_modules)
    validate_task_contract({"task_contract": policy.task_contract}, task)
    policy.reset()
    observation = task.reset()
    starts = torch.ones(task.num_envs, device=task.device, dtype=torch.bool)
    reward, terminations, timeouts = 0., 0, 0
    completions = 0
    physical_failures, outcome_known = 0, True
    tracking_sums, tracking_counts = {}, {}
    action_digest = hashlib.sha256()
    residual_squares = residual_abs = residual_count = 0.0
    residual_max = base_squares = 0.0
    residual_joint_abs = residual_joint_max = None
    target_clip_count = target_count = target_clip_env_steps = 0
    torque_saturation_count = torque_count = torque_saturation_env_steps = 0
    torque_ratio_max = 0.0
    executed_steps = 0
    for _ in range(steps):
        frame = task.deployable_physics() if policy.needs_physics else None
        action = (policy.act_privileged(observation, task, starts)
                  if getattr(policy, "needs_privileged_teacher", False)
                  else policy.act(observation.actor, frame, starts))
        action_digest.update(action.detach().cpu().contiguous().numpy().tobytes())
        transition = task.step(action)
        executed_steps += 1
        residual = getattr(policy, "last_residual", None)
        base_action = getattr(policy, "last_base_action", None)
        if residual is not None and base_action is not None:
            residual_squares += float(residual.square().sum())
            residual_abs += float(residual.abs().sum())
            residual_count += residual.numel()
            residual_max = max(residual_max, float(residual.abs().max()))
            base_squares += float(base_action.square().sum())
            joint_abs = residual.abs().sum(0).double().cpu()
            joint_max = residual.abs().amax(0).double().cpu()
            residual_joint_abs = (joint_abs if residual_joint_abs is None
                                  else residual_joint_abs + joint_abs)
            residual_joint_max = (joint_max if residual_joint_max is None
                                  else torch.maximum(residual_joint_max, joint_max))
        env = getattr(getattr(task, "bridge", None), "env", None)
        manager = getattr(env, "action_manager", None)
        if manager is not None and "joint_pos" in manager.active_terms:
            term = manager.get_term("joint_pos")
            raw = term.raw_action
            before_clip = raw * term.scale + term.offset
            processed = term._processed_actions
            clipped = (before_clip - processed).abs() > 1.0e-7
            target_clip_count += int(clipped.sum())
            target_count += clipped.numel()
            target_clip_env_steps += int(clipped.any(-1).sum())
            entity = getattr(getattr(task, "bridge", None), "entity", None)
            model = getattr(getattr(env, "sim", None), "model", None)
            if entity is not None and model is not None:
                force = entity.data.actuator_force
                force_range = getattr(model, "actuator_forcerange", None)
                ctrl_ids = getattr(entity.data.indexing, "ctrl_ids", None)
                if force_range is not None and not isinstance(force_range, torch.Tensor):
                    try:
                        import warp as wp
                        force_range = wp.to_torch(force_range)
                    except (AttributeError, RuntimeError, TypeError):
                        force_range = None
                if isinstance(force_range, torch.Tensor) and ctrl_ids is not None:
                    limits = force_range[:, ctrl_ids].abs().amax(-1).clamp_min(1.0e-8)
                else:
                    contract = getattr(getattr(task, "adapter", None), "contract", None)
                    robot = getattr(getattr(task, "bridge", None), "robot", None)
                    joint_names = getattr(getattr(task, "bridge", None), "joint_names", None)
                    if contract is not None and robot is not None and joint_names is not None:
                        # K1 uses action_scale = .25 * effort / Kp.  Recover the
                        # exact configured effort limits and reorder them to the
                        # simulator actuator order used by actuator_force.
                        policy_limits = force.new_tensor(contract.action_scale) * force.new_tensor(
                            contract.joint_stiffness
                        ) * 4.0
                        limits = robot.to_simulator_order(
                            policy_limits, joint_names
                        ).unsqueeze(0).expand_as(force).clamp_min(1.0e-8)
                    else:
                        limits = None
                if limits is not None:
                    ratio = force.abs() / limits
                    saturated = ratio >= 0.99
                    torque_saturation_count += int(saturated.sum())
                    torque_count += saturated.numel()
                    torque_saturation_env_steps += int(saturated.any(-1).sum())
                    torque_ratio_max = max(torque_ratio_max, float(ratio.max()))
        policy.record_action(getattr(task, "last_executed_action", action))
        reward += float(transition.reward.sum())
        terminations += int(transition.terminated.sum())
        timeouts += int(transition.truncated.sum())
        finished = getattr(transition, 'motion_finished', None)
        failure = getattr(transition, 'physical_failure', None)
        if finished is not None and failure is not None:
            completions += int((finished & ~failure).sum())
            physical_failures += int(failure.sum())
        else:
            outcome_known = False
        for name in ('root_position_error_m', 'root_orientation_error_rad',
                     'root_linear_velocity_error_mps', 'keybody_linear_velocity_rmse_mps'):
            value = getattr(transition, 'terms', {}).get(name)
            if value is not None:
                if not torch.isfinite(value).all():
                    raise ValueError('non-finite frozen tracking metric')
                tracking_sums[name] = tracking_sums.get(name, 0.)+float(value.sum())
                tracking_counts[name] = tracking_counts.get(name, 0)+value.numel()
        starts = transition.terminated | transition.truncated
        if starts.any().item():
            if stop_on_first_done:
                break
            observation = task.reset(starts.nonzero(as_tuple=False).flatten(),
                random_start=False, min_remaining_s=float(task.sampler.offsets_s[-1]))
        else:
            observation = task.observe()
    after = state_fingerprint(*policy.frozen_modules)
    if before != after:
        raise RuntimeError("frozen evaluation modified policy weights or persistent buffers")
    report = dict(steps=executed_steps, requested_steps=steps, num_envs=task.num_envs,
        stop_on_first_done=stop_on_first_done,
        reward_mean=reward/(executed_steps*task.num_envs), terminations=terminations, timeouts=timeouts,
        reference_completions=completions if outcome_known else None,
        physical_failures=physical_failures if outcome_known else None,
        tracking={name: total/tracking_counts[name] for name, total in tracking_sums.items()},
        deterministic=True, weights_unchanged=True, weight_sha256=after,
        action_sha256=action_digest.hexdigest(), policy=policy.policy_name, ablation=policy.ablation)
    if residual_count:
        report["control_diagnostics"] = {
            "residual_rms": (residual_squares / residual_count) ** 0.5,
            "residual_mean_abs": residual_abs / residual_count,
            "residual_max_abs": residual_max,
            "residual_to_base_rms_ratio": (
                (residual_squares / max(base_squares, 1.0e-30)) ** 0.5
            ),
            "residual_joint_mean_abs": (
                residual_joint_abs / (executed_steps * task.num_envs)
            ).tolist(),
            "residual_joint_max_abs": residual_joint_max.tolist(),
            "target_clip_fraction": (
                target_clip_count / target_count if target_count else None
            ),
            "target_clip_env_step_fraction": (
                target_clip_env_steps / (executed_steps * task.num_envs)
                if target_count else None
            ),
            "torque_saturation_fraction_99pct": (
                torque_saturation_count / torque_count if torque_count else None
            ),
            "torque_saturation_env_step_fraction_99pct": (
                torque_saturation_env_steps / (executed_steps * task.num_envs)
                if torque_count else None
            ),
            "torque_to_limit_max_ratio": torque_ratio_max if torque_count else None,
        }
    return report
