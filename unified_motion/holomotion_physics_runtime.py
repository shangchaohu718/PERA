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
    actor: HoloMotionPhysicsActor
    critic: torch.nn.Module
    dynamics: DynamicsModel
    actor_dynamics: DynamicsModel
    calibration: PhysicsConfidence
    payload: dict


def load_physics_checkpoint(checkpoint, *, device="cpu", holomotion_root=None):
    """Load all branches into new modules; never silently substitute a base actor.

    Schema 5 contains the corrected model and calibration. Schema 4 reproduces
    legacy inference. Schema 3 lacks residual limits and the resolved network
    config; only its original trainer/configuration can migrate it to schema 4.
    """
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=True)
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
    actor_config = dict(runtime["physics_actor_config"])
    if legacy:
        actor_config.setdefault("condition_on_base_action", False)
    actor = HoloMotionPhysicsActor(base_actor, HoloMotionPhysicsActorConfig(**actor_config)).to(device)
    config = dict(runtime["dynamics_config"])
    if legacy:
        config.setdefault("mask_unobserved_core", False)
    robot = RobotContract(**config.pop("robot"))
    dynamics = DynamicsModel(robot, **config).to(device)
    ema = DynamicsModel(robot, **config).to(device)
    if actor.config.physics_feature_dim != 3*dynamics.physics_width+dynamics.latent_width:
        raise ValueError("Physics Actor and Dynamics feature widths disagree")
    if actor.config.action_dim != robot.dof or actor.config.action_dim != 22:
        raise ValueError("Physics Actor and K1 action contracts disagree")
    if not dynamics.uncertainty or not dynamics.system_id_dim:
        raise ValueError("Physics confidence requires uncertainty and system identification")
    base_actor.load_state_dict(payload["base_actor_state_dict"], strict=True)
    critic.load_state_dict(payload["critic_state_dict"], strict=True)
    actor.physics_residual.load_state_dict(payload["physics_residual_state_dict"], strict=True)
    dynamics.load_state_dict(payload["dynamics_state_dict"], strict=True)
    ema.load_state_dict(payload["actor_dynamics_state_dict"], strict=True)
    calibration = PhysicsConfidence(dynamics.system_id_dim,
        PhysicsConfidenceConfig(mode="legacy") if legacy else
        PhysicsConfidenceConfig(**runtime["confidence_config"])).to(device)
    if not legacy:
        calibration.load_state_dict(payload["confidence_state_dict"], strict=True)
        if (calibration.samples < 0 or (calibration.mse < 0).any()
                or (calibration.temperature < 1).any()
                or ((calibration.coverage < 0) | (calibration.coverage > 1)).any()):
            raise ValueError("invalid confidence calibration state")
    for module in (actor, critic, dynamics, ema, calibration):
        for value in module.state_dict().values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                raise ValueError("Physics checkpoint contains non-finite weights")
        module.eval()
    return PhysicsCheckpoint(actor, critic, dynamics, ema, calibration, payload)


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


@torch.no_grad()
def evaluate_physics_policy(policy, task, *, steps: int):
    """Bounded deterministic evaluation; never calls PPO, Dynamics update or Teacher."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be positive")
    before = state_fingerprint(*policy.frozen_modules)
    validate_task_contract({"task_contract": policy.task_contract}, task)
    policy.reset()
    observation = task.reset()
    starts = torch.ones(task.num_envs, device=task.device, dtype=torch.bool)
    reward, terminations, timeouts = 0., 0, 0
    action_digest = hashlib.sha256()
    for _ in range(steps):
        frame = task.deployable_physics() if policy.needs_physics else None
        action = policy.act(observation.actor, frame, starts)
        action_digest.update(action.detach().cpu().contiguous().numpy().tobytes())
        transition = task.step(action)
        policy.record_action(getattr(task, "last_executed_action", action))
        reward += float(transition.reward.sum())
        terminations += int(transition.terminated.sum())
        timeouts += int(transition.truncated.sum())
        starts = transition.terminated | transition.truncated
        if starts.any().item():
            observation = task.reset(starts.nonzero(as_tuple=False).flatten(),
                random_start=False, min_remaining_s=float(task.sampler.offsets_s[-1]))
        else:
            observation = task.observe()
    after = state_fingerprint(*policy.frozen_modules)
    if before != after:
        raise RuntimeError("frozen evaluation modified policy weights or persistent buffers")
    return dict(steps=steps, num_envs=task.num_envs,
        reward_mean=reward/(steps*task.num_envs), terminations=terminations, timeouts=timeouts,
        deterministic=True, weights_unchanged=True, weight_sha256=after,
        action_sha256=action_digest.hexdigest(), policy=policy.policy_name, ablation=policy.ablation)
