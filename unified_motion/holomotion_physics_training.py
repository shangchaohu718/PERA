"""Physics-student and privileged-Teacher training for HoloMotion on mjlab."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .algorithms.sequence_training import parameter_stamp, replay_dynamics_inputs
from .algorithms.dynamics_history import DynamicsHistory, snapshot_physics
from .algorithms.common import diagonal_gaussian_kl
from .algorithms.dynamics_objective import identification_loss, validate_teacher_weights
from .holomotion_mjlab_ppo import (
    HoloMotionMJLabPPO,
    HoloMotionPPORollout,
    generalized_advantage_estimate,
    normalize_advantages,
    build_episode_causal_mask,
)
from .holomotion_original_algorithm import flat_observations_to_tensordict
from .holomotion_physics_actor import (
    HoloMotionPhysicsActor,
    HoloMotionPhysicsActorConfig,
    dynamics_actor_features,
)
from .models.physics import DynamicsModel, PhysicsObservation
from .holomotion_physics_runtime import physics_policy_inputs
from .physics_data import stack_physics as _stack_physics, concat_physics as _concat_physics, slice_physics as _slice_physics
from .physics_confidence import PhysicsConfidence, PhysicsConfidenceConfig
from .physics_update import PhysicsUpdateConfig
from .privileged_teacher import (
    PrivilegedTeacherConfig, PrivilegedTransformerTeacher, teacher_inputs,
    normalized_core, core_scale, teacher_supervision,
    assess_teacher, predictive_distillation, core_groups,
)


def _slice_encoding(encoding, begin, end):
    """Slice sequence outputs, retaining the final recurrent state unchanged."""
    fields = ('latent', 'context', 'prediction_error', 'contact_tokens',
              'object_tokens', 'prediction_context')
    return replace(encoding, **{name: (None if getattr(encoding, name) is None
                                    else getattr(encoding, name)[:, begin:end]) for name in fields})


@dataclass(frozen=True)
class HoloMotionDynamicsMetrics:
    total_loss: float
    grad_norm: float
    prediction_loss: float
    left_contact_loss: float
    right_contact_loss: float
    system_identification_loss: float
    teacher_reconstruction_loss: float
    teacher_distillation_loss: float
    mean_system_id_std: float
    mean_state_std: float
    optimizer_steps: int
    deployment_update_accepted: bool
    deployment_mean_kl: float
    deployment_max_kl: float
    calibration_samples: int
    calibration_skill: float
    ema_decay: float
    teacher_admitted_fraction: float = 0.
    teacher_validation_samples: int = 0
    multistep_prediction_loss: float = 0.


class HoloMotionPhysicsMJLabPPO(HoloMotionMJLabPPO):
    """Original HoloMotion PPO plus a deployable dynamics residual branch."""

    def __init__(
        self,
        algorithm: Any,
        task: Any,
        dynamics: DynamicsModel,
        *,
        config: Any | None = None,
        physics_actor_config: HoloMotionPhysicsActorConfig | None = None,
        dynamics_learning_rate: float = 1.0e-4,
        system_identification_weight: float = 0.1,
        teacher_reconstruction_weight: float | None = None,
        teacher_distillation_weight: float | None = None,
        dynamics_burn_in_steps: int = 64,
        dynamics_train_steps: int = 128,
        confidence_warmup_steps: int = 128,
        dynamics_ema_decay: float | None = None,
        dynamics_epochs: int = 4,
        dynamics_history_steps: int = 16384,
        dynamics_history_bytes: int = 512*1024**2,
        policy_rollout_steps: int = 32,
        confidence_config: PhysicsConfidenceConfig | None = None,
        update_config: PhysicsUpdateConfig | None = None,
        privileged_teacher_config: dict | None = None,
    ) -> None:
        if not isinstance(dynamics, DynamicsModel):
            raise TypeError("dynamics must be a DynamicsModel")
        if dynamics.robot != task.bridge.robot:
            raise ValueError("HoloMotion task and dynamics robot contracts differ")
        if dynamics.interaction_feedback:
            raise ValueError("first HoloMotion integration does not replay interaction feedback")
        if dynamics.system_id_dim != task.physics_teacher_dim or not dynamics.system_id_dim:
            raise ValueError("dynamics identification width must match slow-physics target")
        if not dynamics.uncertainty:
            raise ValueError("HoloMotion physics dynamics must enable uncertainty heads")
        if not dynamics.mask_unobserved_core:
            raise ValueError("legacy unmasked Dynamics are inference-only")
        super().__init__(algorithm, task, config=config)
        self.update_config = update_config or PhysicsUpdateConfig(
            freeze_base_actor=getattr(algorithm, 'pretrained', False) is True)
        self.confidence = PhysicsConfidence(dynamics.system_id_dim,
            confidence_config or PhysicsConfidenceConfig(mode='bounded')).to(task.device)
        if self.confidence.config.mode == "legacy":
            raise ValueError("legacy confidence is inference-only")
        actor_cfg = physics_actor_config or HoloMotionPhysicsActorConfig(
            physics_feature_dim=3 * dynamics.physics_width + dynamics.latent_width
        )
        self.actor = HoloMotionPhysicsActor(self.actor, actor_cfg).to(task.device)
        if self.update_config.freeze_base_actor:
            self.actor.base_actor.requires_grad_(False)
        # Rebuild the optimizer so the new residual head participates in PPO.
        decay, no_decay = [], []
        for name, parameter in self.actor.named_parameters():
            if not parameter.requires_grad:
                continue
            target = no_decay if (
                parameter.ndim < 2 or "log_std" in name or "bias" in name or "norm" in name
            ) else decay
            target.append(parameter)
        self.actor_optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.01},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.actor_learning_rate,
            betas=(self.config.actor_beta1, self.config.actor_beta2),
        )
        self.dynamics = dynamics.to(task.device)
        self.dynamics.eval()
        self.actor_dynamics = copy.deepcopy(self.dynamics).to(task.device).eval()
        self.actor_dynamics.requires_grad_(False)
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics.parameters(), lr=dynamics_learning_rate
        )
        self.privileged_teacher = self.privileged_teacher_ema = None
        self.privileged_teacher_optimizer = None
        self.privileged_teacher_updates = 0
        if privileged_teacher_config is not None:
            if dynamics.teacher_student:
                raise ValueError('select either the legacy MLP Teacher or the privileged Transformer')
            teacher_cfg = PrivilegedTeacherConfig(**privileged_teacher_config)
            # Teacher initialization must not perturb the Student exploration RNG.
            devices = [task.device] if torch.device(task.device).type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                self.privileged_teacher = PrivilegedTransformerTeacher(
                    dynamics.robot, dynamics.latent_width, dynamics.system_id_dim, teacher_cfg).to(task.device)
            self.privileged_teacher_ema = copy.deepcopy(self.privileged_teacher).eval().requires_grad_(False)
            self.privileged_teacher_optimizer = torch.optim.Adam(
                self.privileged_teacher.parameters(), lr=teacher_cfg.learning_rate)
        has_teacher = dynamics.teacher_student or self.privileged_teacher is not None
        self.system_identification_weight = float(system_identification_weight)
        self.teacher_reconstruction_weight = float((.1 if has_teacher else 0.)
            if teacher_reconstruction_weight is None else teacher_reconstruction_weight)
        self.teacher_distillation_weight = float((.05 if has_teacher else 0.)
            if teacher_distillation_weight is None else teacher_distillation_weight)
        if not math.isfinite(self.system_identification_weight) or self.system_identification_weight <= 0:
            raise ValueError("system identification weight must be positive")
        validate_teacher_weights(dynamics, self.teacher_reconstruction_weight, self.teacher_distillation_weight,
                                 teacher_available=self.privileged_teacher is not None)
        if any(type(value) is not int or value < 1 for value in (
            dynamics_burn_in_steps, dynamics_train_steps, confidence_warmup_steps,
            dynamics_epochs, policy_rollout_steps,
        )):
            raise ValueError("burn-in, train, and confidence warmup steps must be positive")
        if dynamics_ema_decay is not None and not 0.0 < dynamics_ema_decay < 1.0:
            raise ValueError("dynamics_ema_decay must be in (0,1)")
        self.dynamics_burn_in_steps = dynamics_burn_in_steps
        self.dynamics_train_steps = dynamics_train_steps
        self.confidence_warmup_steps = confidence_warmup_steps
        self.dynamics_ema_decay = dynamics_ema_decay
        self.dynamics_epochs = dynamics_epochs
        if dynamics_train_steps <= math.ceil(dynamics_train_steps*self.update_config.calibration_fraction):
            raise ValueError("Dynamics window needs separate training and calibration transitions")
        self._dynamics_rollouts: list[HoloMotionPPORollout] = []
        self._physics_memory = None
        self._physics_episode_start = None
        self._physics_age = None
        self._dynamics_history = DynamicsHistory(dynamics_history_steps, max_bytes=dynamics_history_bytes)
        self._policy_rollout_steps = policy_rollout_steps

    def privileged_teacher_checkpoint(self):
        if self.privileged_teacher is None:
            return None
        return dict(schema=1, config=asdict(self.privileged_teacher.config),
            model=self.privileged_teacher.state_dict(), ema=self.privileged_teacher_ema.state_dict(),
            optimizer=self.privileged_teacher_optimizer.state_dict(), updates=self.privileged_teacher_updates)

    def restore_privileged_teacher(self, payload):
        saved = payload.get('privileged_teacher')
        if (saved is None) != (self.privileged_teacher is None):
            raise ValueError('checkpoint privileged Teacher configuration differs')
        if saved is None:
            return
        if saved.get('schema') != 1 or PrivilegedTeacherConfig.from_saved(saved.get('config', {})) != self.privileged_teacher.config:
            raise ValueError('unsupported privileged Teacher checkpoint')
        if type(saved.get('updates')) is not int or saved['updates'] < 0:
            raise ValueError('invalid privileged Teacher update count')
        for field, module in [('model', self.privileged_teacher), ('ema', self.privileged_teacher_ema)]:
            if any(not torch.isfinite(v).all() for v in saved[field].values() if isinstance(v, torch.Tensor)):
                raise ValueError('non-finite privileged Teacher checkpoint')
            module.load_state_dict(saved[field], strict=True)
        self.privileged_teacher_optimizer.load_state_dict(saved['optimizer'])
        self.privileged_teacher_updates = saved['updates']

    @classmethod
    def from_checkpoint(cls, checkpoint, task, *, holomotion_root=None,
                        dynamics_history_steps=16384, dynamics_history_bytes=512*1024**2):
        """Resume every model branch and optimizer using checkpoint configuration."""
        from .holomotion_physics_runtime import load_physics_checkpoint, validate_task_contract
        from .holomotion_mjlab_ppo import HoloMotionPPOConfig
        from .holomotion_original_algorithm import OriginalHoloMotionK1Algorithm
        from pathlib import Path
        loaded = load_physics_checkpoint(checkpoint, device=task.device,
                                         holomotion_root=holomotion_root)
        payload = loaded.payload
        if payload["physics_schema"] != 5:
            raise ValueError("legacy Physics checkpoints support frozen evaluation; initialize a new schema 5 training run")
        validate_task_contract(payload["physics_runtime_config"], task)
        algorithm = OriginalHoloMotionK1Algorithm(loaded.actor.base_actor, loaded.critic,
            None, Path(checkpoint).parent, payload["physics_runtime_config"]["network_config"])
        result = cls(algorithm, task, loaded.dynamics,
            config=HoloMotionPPOConfig(**payload["ppo_config"]),
            physics_actor_config=loaded.actor.config,
            policy_rollout_steps=payload["physics_runtime_config"]["rollout_steps"],
            dynamics_history_steps=dynamics_history_steps,
            dynamics_history_bytes=dynamics_history_bytes,
            privileged_teacher_config=(None if payload.get('privileged_teacher') is None
                                       else asdict(PrivilegedTeacherConfig.from_saved(payload['privileged_teacher']['config']))),
            confidence_config=loaded.calibration.config,
            update_config=PhysicsUpdateConfig(**payload["physics_update_config"]),
            **payload["physics_training_config"], **payload["physics_loss_weights"])
        result.load_optimizer_checkpoint(checkpoint, _validated_physics=loaded)
        return result

    @torch.no_grad()
    def reset(self, *, random_start: bool = False, min_remaining_s: float = 0.0):
        observation = super().reset(
            random_start=random_start, min_remaining_s=min_remaining_s
        )
        self._physics_memory = None
        self._dynamics_history.clear()
        self._dynamics_rollouts.clear()
        self._physics_episode_start = torch.ones(
            self.task.num_envs, dtype=torch.bool, device=self.task.device
        )
        self._physics_age = torch.zeros(
            self.task.num_envs, dtype=torch.long, device=self.task.device
        )
        return observation

    @torch.no_grad()
    def collect(self, num_steps: int) -> HoloMotionPPORollout:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.actor.eval(); self.critic.eval()
        if self._observation is None:
            self.reset()
        self.actor.reset_kv_cache(self.task.num_envs, self.task.device)
        self.actor_dynamics.eval()
        if self._physics_episode_start is None:
            raise RuntimeError("physics episode state was not initialized")

        if self._physics_age is None:
            raise RuntimeError("physics evidence age was not initialized")
        actor_obs=[];critic_obs=[];actions=[];log_probs=[];means=[];sigmas=[];values=[]
        rewards=[];dones=[];clip_ids=[];features=[];confidences=[];frames=[];next_frames=[];starts=[]
        truncated=[];timeout_values=[]
        reward_terms: dict[str,list[torch.Tensor]]={}
        previous_core=[];previous_valid=[];teacher_targets=[]
        privileged_inputs=[];privileged_cores=[]
        privileged_contacts=[];privileged_contact_masks=[]
        teacher_ids = None
        if self.privileged_teacher is not None:
            count = min(self.task.num_envs, self.privileged_teacher.config.train_envs)
            teacher_ids = ((torch.arange(count, device=self.task.device)
                            + self.privileged_teacher_updates*count) % self.task.num_envs)
        aux_base=[];aux_contacts=[];aux_ref=[];aux_robot=[]
        initial_hidden = None
        stamp = parameter_stamp(self.dynamics)
        env_ids = tuple(range(self.task.num_envs))
        dynamics_prefix = self._dynamics_history.snapshot()
        executed_actions = []
        ages = []
        self._policy_rollout_steps = num_steps

        for _ in range(num_steps):
            observation = self._observation
            assert observation is not None
            frame = snapshot_physics(self.task.deployable_physics())
            start = self._physics_episode_start.clone()
            ages.append(self._physics_age.clone())
            self._dynamics_history.check_capacity(start, frame)
            before = self._physics_memory
            pre_hidden = (
                frame.body_obs.new_zeros(self.task.num_envs, self.actor_dynamics.memory_width)
                if before is None else before.hidden
            )
            if initial_hidden is None:
                initial_hidden = pre_hidden.clone()
            previous_core.append(
                frame.body_obs.new_zeros(self.task.num_envs, 9 + 2*self.actor_dynamics.robot.dof)
                if before is None or before.predicted_core is None
                else before.predicted_core.clone()
            )
            previous_valid.append(
                torch.zeros_like(start)
                if before is None or before.prediction_valid is None
                else before.prediction_valid.clone()
            )
            encoding, memory, feature, confidence = physics_policy_inputs(
                self.actor_dynamics, frame, start, env_ids, before,
                self._physics_age, self.confidence_warmup_steps, self.confidence)
            teacher_target = self.task.physics_teacher_target()
            if teacher_ids is not None:
                truth = self.task.privileged_physics_observation()
                privileged_inputs.append(teacher_inputs(observation.critic, truth, teacher_target,
                    self.dynamics.robot, self.privileged_teacher.config).index_select(0, teacher_ids))
                privileged_cores.append(normalized_core(truth, self.dynamics.robot).index_select(0, teacher_ids).clone())
                privileged_contacts.append(truth.contact_features[..., 10].index_select(0, teacher_ids).clone())
                privileged_contact_masks.append(truth.contact_mask.index_select(0, teacher_ids).clone())
            data = flat_observations_to_tensordict(observation.actor, observation.critic)
            actor_out = self._actor_forward(
                data, mode="sampling", update_obs_norm=False,
                physics_features=feature, physics_confidence=confidence,
            )
            critic_out = self.critic(data, update_obs_norm=False)
            action = actor_out.get("actions")
            transition = self.task.step(action)
            executed = getattr(self.task, "last_executed_action", action).detach().clone()
            executed_actions.append(executed)
            self._dynamics_history.append(frame, start, executed)
            prediction = self.actor_dynamics.predict_next(
                frame, executed, encoding.latent,
                physics_encoding=encoding.physics_encoding(),
            )
            self._physics_memory = self.actor_dynamics.remember_prediction(memory, prediction)
            next_frame = snapshot_physics(self.task.privileged_physics_observation())
            done = transition.terminated | transition.truncated

            actor_obs.append(observation.actor);critic_obs.append(observation.critic)
            actions.append(action);log_probs.append(actor_out.get("actions_log_prob").reshape(-1,1))
            means.append(actor_out.get("mu"));sigmas.append(actor_out.get("sigma"))
            values.append(critic_out.get("values").reshape(-1,1));rewards.append(transition.reward.reshape(-1,1))
            dones.append(done.reshape(-1,1));clip_ids.append(self.task.sampler.clip_ids.clone())
            truncated.append(transition.truncated.reshape(-1,1))
            for name,value in transition.terms.items():
                reward_terms.setdefault(name,[]).append(value.detach().clone())
            features.append(feature);confidences.append(confidence)
            frames.append(frame);next_frames.append(next_frame);starts.append(start)
            teacher_targets.append(teacher_target)
            aux_base.append(observation.auxiliary.base_lin_vel_b)
            aux_contacts.append(observation.auxiliary.keybody_contacts)
            aux_ref.append(observation.auxiliary.ref_keybody_rel_pos)
            aux_robot.append(observation.auxiliary.robot_keybody_rel_pos)

            timeout_values.append(self._finish_transition(transition))
            self._physics_episode_start = done
            self._physics_age = torch.where(done, 0, self._physics_age + 1)

        final=self._observation;assert final is not None
        final_data=flat_observations_to_tensordict(final.actor,final.critic)
        bootstrap=self.critic(final_data,update_obs_norm=False).get("values").reshape(-1,1)
        stack=lambda x:torch.stack(x,dim=1)
        reward_tensor=stack(rewards);value_tensor=stack(values);done_tensor=stack(dones).bool()
        advantage,returns=generalized_advantage_estimate(
            reward_tensor,value_tensor,done_tensor,bootstrap,
            gamma=self.config.gamma,lam=self.config.lam,
            truncated=stack(truncated),timeout_values=stack(timeout_values),
        )
        if self.config.normalize_advantage:
            advantage=normalize_advantages(advantage)
        return HoloMotionPPORollout(
            stack(actor_obs),stack(critic_obs),stack(actions),stack(log_probs),stack(means),
            stack(sigmas),value_tensor,returns,advantage,done_tensor,reward_tensor,
            stack(aux_base),stack(aux_contacts),stack(aux_ref),stack(aux_robot),stack(clip_ids),
            physics_features=stack(features),physics_confidence=stack(confidences),
            physics_observation=_stack_physics(frames),
            next_physics_observation=_stack_physics(next_frames),
            physics_episode_start=stack(starts),physics_initial_hidden=initial_hidden,
            previous_predicted_core=stack(previous_core),previous_prediction_valid=stack(previous_valid),
            physics_teacher_target=stack(teacher_targets),dynamics_stamp=stamp,
            reward_terms={name:stack(values) for name,values in reward_terms.items()},
            truncated=stack(truncated),timeout_values=stack(timeout_values),
            executed_actions=stack(executed_actions),dynamics_prefix=dynamics_prefix,
            physics_age=stack(ages),
            privileged_teacher_input=stack(privileged_inputs) if privileged_inputs else None,
            privileged_teacher_core=stack(privileged_cores) if privileged_cores else None,
            privileged_teacher_env_ids=teacher_ids,
            privileged_teacher_contacts=stack(privileged_contacts) if privileged_contacts else None,
            privileged_teacher_contact_mask=stack(privileged_contact_masks) if privileged_contact_masks else None,
        )

    def update_dynamics(
        self, rollout: HoloMotionPPORollout
    ) -> HoloMotionDynamicsMetrics | None:
        required=(rollout.physics_observation,rollout.next_physics_observation,
                  rollout.physics_episode_start,rollout.physics_initial_hidden,
                  rollout.previous_predicted_core,rollout.previous_prediction_valid,
                  rollout.physics_teacher_target,rollout.dynamics_stamp,
                  rollout.physics_features,rollout.physics_confidence,rollout.physics_age)
        if any(value is None for value in required):
            raise ValueError("rollout requires Physics supervision, features, confidence and evidence age")
        if parameter_stamp(self.dynamics) != rollout.dynamics_stamp:
            raise RuntimeError("dynamics changed after rollout collection")
        self._dynamics_rollouts.append(rollout)
        required_steps = self.dynamics_burn_in_steps + self.dynamics_train_steps
        available_steps = sum(item.actions.shape[1] for item in self._dynamics_rollouts)
        if available_steps < required_steps:
            return None
        batches = self._dynamics_rollouts
        if any(item.dynamics_stamp != rollout.dynamics_stamp for item in batches):
            raise RuntimeError("physics replay mixed different dynamics versions")
        observation = _concat_physics([item.physics_observation for item in batches])
        next_observation = _concat_physics([item.next_physics_observation for item in batches])
        episode_start = torch.cat([item.physics_episode_start for item in batches], dim=1)
        previous_core = torch.cat([item.previous_predicted_core for item in batches], dim=1)
        previous_valid = torch.cat([item.previous_prediction_valid for item in batches], dim=1)
        teacher_target = torch.cat([item.physics_teacher_target for item in batches], dim=1)
        actions = torch.cat([item.actions if item.executed_actions is None
                             else item.executed_actions for item in batches], dim=1)
        burn = self.dynamics_burn_in_steps
        # Keep all accumulated transitions, including a final partial rollout.
        end = available_steps
        calibration_steps = max(1, math.ceil((end-burn)*self.update_config.calibration_fraction))
        calibration_begin = end-calibration_steps
        train_end = calibration_begin
        teacher_v2 = self.privileged_teacher is not None and self.privileged_teacher.config.objective_version == 2
        if teacher_v2 and calibration_begin-burn >= 2:
            teacher_validation_steps = min(calibration_begin-burn-1,
                max(1, math.ceil((end-burn)*self.privileged_teacher.config.validation_fraction)))
            train_end -= teacher_validation_steps
        if train_end <= burn:
            raise ValueError("Dynamics window needs separate training and calibration transitions")
        privileged_targets = self._update_privileged_teacher(
            batches, next_observation, episode_start, actions, teacher_target, burn, train_end,
            validation_end=calibration_begin)
        self.dynamics.train()
        try:
            train_observation=_slice_physics(observation,burn,train_end)
            train_next=_slice_physics(next_observation,burn,train_end)
            target=teacher_target[:,burn:train_end].detach()
            # The randomized parameters exist from the first frame of every
            # episode; prediction validity is unrelated to Teacher labels.
            totals=[]
            teacher_quality = actions.new_zeros(6)
            teacher_validation_samples = 0
            multistep_losses = []
            for _ in range(self.dynamics_epochs):
                prefix = batches[0].dynamics_prefix
                if prefix is None:
                    if not episode_start[:, 0].all().item():
                        raise ValueError("Dynamics update requires a real history prefix")
                    prefix = DynamicsHistory()
                initial_hidden, previous_core, previous_valid, _ = replay_dynamics_inputs(
                    self.dynamics, prefix, observation, episode_start, actions)
                self.dynamics.train()
                # Recompute both burn-in and the differentiable window after
                # every optimizer step; no activation graph is reused with
                # changed Dynamics weights.
                with torch.no_grad():
                    burn_encoding=self.dynamics.forward_sequence(
                        _slice_physics(observation,0,burn),
                        episode_start=episode_start[:,:burn],
                        initial_hidden=initial_hidden.detach(),
                        previous_predicted_core=previous_core[:,:burn].detach(),
                        previous_prediction_valid=previous_valid[:,:burn],
                    )
                encoding=self.dynamics.forward_sequence(
                    _slice_physics(observation, burn, calibration_begin if teacher_v2 else train_end),
                    episode_start=episode_start[:,burn:calibration_begin if teacher_v2 else train_end],
                    initial_hidden=burn_encoding.hidden.detach(),
                    previous_predicted_core=previous_core[:,burn:calibration_begin if teacher_v2 else train_end].detach(),
                    previous_prediction_valid=previous_valid[:,burn:calibration_begin if teacher_v2 else train_end],
                )
                if privileged_targets is not None and teacher_v2 and not totals and train_end < calibration_begin:
                    ids, teacher_output, _, _ = privileged_targets
                    validation_encoding = _slice_encoding(encoding, train_end-burn, calibration_begin-burn)
                    with torch.no_grad():
                        validation_prediction = self.dynamics.predict_next(
                            _slice_physics(observation, train_end, calibration_begin),
                            actions[:, train_end:calibration_begin], validation_encoding.latent,
                            physics_encoding=validation_encoding.physics_encoding())
                        teacher_quality, self.last_teacher_validation = assess_teacher(
                            teacher_output['validation'],
                            validation_prediction.next_core[ids]/core_scale(self.dynamics.robot, actions),
                            validation_prediction.next_contact_logits[ids], teacher_output['shuffled'],
                            normalized_core(next_observation, self.dynamics.robot)[ids, train_end:calibration_begin],
                            teacher_output['current'], next_observation.core_valid[ids, train_end:calibration_begin],
                            next_observation.contact_features[ids, train_end:calibration_begin, :, 10],
                            teacher_output['current_contacts'],
                            next_observation.contact_mask[ids, train_end:calibration_begin]
                                & teacher_output['current_contact_mask'],
                            self.privileged_teacher.config)
                        teacher_validation_samples = len(ids)*(calibration_begin-train_end)
                encoding = _slice_encoding(encoding, 0, train_end-burn)
                prediction=self.dynamics.predict_next(
                    train_observation,actions[:,burn:train_end].detach(),encoding.latent,
                    physics_encoding=encoding.physics_encoding(),
                )
                valid=torch.ones_like(episode_start[:,burn:train_end])
                prediction_loss=self.dynamics.prediction_loss(
                    prediction,train_next,valid_transition=valid
                )
                multistep_loss = self._multistep_loss(observation, next_observation, episode_start,
                    actions, encoding, burn, train_end)
                multistep_losses.append(float(multistep_loss.detach()))
                mask = previous_valid[:,burn:train_end] & ~episode_start[:,burn:train_end]
                auxiliary, _, id_logvar = identification_loss(self.dynamics, encoding.latent, target, mask,
                    reconstruction_weight=self.teacher_reconstruction_weight if privileged_targets is None else 0.,
                    distillation_weight=self.teacher_distillation_weight if privileged_targets is None else 0., teacher_valid=valid)
                system_id=auxiliary["system_identification"]
                reconstruction=auxiliary["teacher_reconstruction"]
                distillation=auxiliary["teacher_distillation"]
                if privileged_targets is not None:
                    ids, teacher_output, teacher_loss, ramp = privileged_targets
                    student_core = prediction.next_core[ids]/core_scale(self.dynamics.robot, prediction.next_core)
                    selected_mask = mask[ids]
                    if teacher_v2:
                        distillation = ramp*predictive_distillation(student_core,
                            prediction.next_contact_logits[ids], teacher_output,
                            train_next.core_valid[ids], train_next.contact_mask[ids], selected_mask, teacher_quality)
                    else:
                        core_valid = train_next.core_valid[ids]
                        core_error = torch.where(core_valid,
                            (student_core-teacher_output['next_core']).square(), 0.)
                        error = ((encoding.latent[ids]-teacher_output['latent']).square().mean(-1)
                                 + core_error.sum(-1)/core_valid.sum(-1).clamp_min(1))
                        distillation = ramp*torch.where(selected_mask, error, 0.).sum()/selected_mask.sum().clamp_min(1)
                    reconstruction = teacher_loss  # Separate Teacher optimizer; detached diagnostic here.
                contact_labels=train_next.contact_features[...,10]
                contact_bce=F.binary_cross_entropy_with_logits(
                    prediction.next_contact_logits,contact_labels,reduction="none"
                )
                left_contact=contact_bce[...,0].mean();right_contact=contact_bce[...,1].mean()
                total=(prediction_loss.total+.1*multistep_loss+self.system_identification_weight*system_id
                       +self.teacher_reconstruction_weight*reconstruction
                       +self.teacher_distillation_weight*distillation)
                self.dynamics_optimizer.zero_grad(set_to_none=True);total.backward()
                grad=nn.utils.clip_grad_norm_(
                    self.dynamics.parameters(),1.0,error_if_nonfinite=True
                )
                self.dynamics_optimizer.step()
                totals.append((total.detach(),grad.detach(),prediction_loss.total.detach(),
                    left_contact.detach(),right_contact.detach(),system_id.detach(),
                    reconstruction.detach(),distillation.detach(),
                    torch.exp(0.5*id_logvar).mean().detach(),
                    torch.exp(0.5*prediction.next_core_logvar).mean().detach()))
            deployment = self._publish_dynamics(batches, observation, episode_start, actions,
                                                teacher_target, calibration_begin)
            averaged=[float(torch.stack([row[i] for row in totals]).mean())
                      for i in range(len(totals[0]))]
            return HoloMotionDynamicsMetrics(*averaged,self.dynamics_epochs, **deployment,
                teacher_admitted_fraction=float(teacher_quality.mean()),
                teacher_validation_samples=teacher_validation_samples,
                multistep_prediction_loss=sum(multistep_losses)/len(multistep_losses))
        finally:
            self._dynamics_rollouts.clear()
            self.dynamics.invalidate_memories();self.dynamics.eval()
            self.actor_dynamics.invalidate_memories();self.actor_dynamics.eval()
            # Rebuild the EMA student's live memory from observed states and
            # executed actions. Observation, episode age, and simulator stay put.
            self._physics_memory, _ = self._dynamics_history.rebuild(
                self.actor_dynamics, tuple(range(self.task.num_envs)))

    def _multistep_loss(self, observation, next_observation, starts, actions, encoding, burn, train_end):
        """Supervise executed action sequences entirely within the training prefix.

        The final transition may terminate, but no intermediate reset may be
        crossed. Direct horizon heads avoid inventing unavailable future inputs.
        """
        terms = []
        for horizon in self.dynamics.prediction_horizons:
            length = train_end-burn-horizon+1
            if horizon <= 1 or length <= 0:
                continue
            valid = torch.ones_like(starts[:, burn:burn+length])
            for offset in range(1, horizon):
                valid &= ~starts[:, burn+offset:burn+offset+length]
            if not valid.any():
                continue
            window = torch.stack([actions[:, burn+i:burn+i+length]
                                  for i in range(horizon)], dim=-2).detach()
            encoded = _slice_encoding(encoding, 0, length)
            prediction = self.dynamics.predict_future_core(
                _slice_physics(observation, burn, burn+length), window, encoded.latent,
                horizon=horizon, physics_encoding=encoded.physics_encoding())
            future = _slice_physics(next_observation, burn+horizon-1, train_end)
            truth = normalized_core(future, self.dynamics.robot).detach()
            prediction = prediction/core_scale(self.dynamics.robot, prediction)
            mask = future.core_valid & valid.unsqueeze(-1)
            groups = []
            for section in core_groups(truth.shape[-1]):
                selected = mask[..., section]
                error = torch.where(selected, prediction[..., section]-truth[..., section], 0.)
                groups.append(error.square().sum()/selected.sum().clamp_min(1))
            terms.append(torch.stack(groups).mean())
        return torch.stack(terms).mean() if terms else actions.new_zeros(())

    def _update_privileged_teacher(self, batches, next_observation, starts, actions, parameters, burn, train_end,
                                   validation_end=None):
        if self.privileged_teacher is None or not self.teacher_reconstruction_weight:
            return None
        ids = batches[0].privileged_teacher_env_ids
        if ids is None or any(batch.privileged_teacher_input is None
                or batch.privileged_teacher_core is None
                or batch.privileged_teacher_contacts is None
                or batch.privileged_teacher_contact_mask is None
                or not torch.equal(ids, batch.privileged_teacher_env_ids) for batch in batches):
            raise ValueError('Transformer Teacher requires matching, current-frame privileged rollout data')
        validation_end = train_end if validation_end is None else validation_end
        inputs = torch.cat([b.privileged_teacher_input for b in batches], 1)[:, :validation_end]
        current = torch.cat([b.privileged_teacher_core for b in batches], 1)[:, :validation_end]
        selected_starts, selected_actions = starts[ids, :validation_end], actions[ids, :validation_end]
        self.privileged_teacher.train()
        output = self.privileged_teacher(inputs[:, :train_end], selected_starts[:, :train_end],
            selected_actions[:, :train_end], current[:, :train_end])
        valid = torch.zeros_like(selected_starts[:, :train_end])
        valid[:, burn:] = True
        truth = normalized_core(next_observation, self.dynamics.robot)[ids, :train_end]
        contacts = next_observation.contact_features[ids, :train_end, :, 10]
        contact_mask = next_observation.contact_mask[ids, :train_end]
        padding = self.privileged_teacher.config.contact_slots-contacts.shape[-1]
        if padding < 0:
            raise ValueError('Teacher contact labels exceed configured slots')
        contacts, contact_mask = F.pad(contacts, (0, padding)), F.pad(contact_mask, (0, padding))
        loss = teacher_supervision(output, truth, next_observation.core_valid[ids, :train_end],
            contacts, contact_mask, parameters[ids, :train_end], valid, self.privileged_teacher.config)
        self.privileged_teacher_optimizer.zero_grad(set_to_none=True)
        (self.teacher_reconstruction_weight*loss).backward()
        nn.utils.clip_grad_norm_(self.privileged_teacher.parameters(), 1., error_if_nonfinite=True)
        self.privileged_teacher_optimizer.step()
        self.privileged_teacher_updates += 1
        with torch.no_grad():
            decay = self.privileged_teacher.config.ema_decay
            for ema, online in zip(self.privileged_teacher_ema.parameters(), self.privileged_teacher.parameters(), strict=True):
                ema.lerp_(online, 1-decay)
            self.privileged_teacher_ema.eval()
            targets = self.privileged_teacher_ema(inputs, selected_starts, selected_actions, current)
            training_targets = {key: value[:, burn:train_end].detach() for key, value in targets.items()}
            if self.privileged_teacher.config.objective_version == 2 and train_end < validation_end:
                validation = {key: value[:, train_end:validation_end].detach() for key, value in targets.items()}
                action_window = selected_actions[:, train_end:validation_end]
                shuffled_actions = action_window.flatten(0, 1).roll(1, 0).reshape_as(action_window)
                training_targets.update(validation=validation,
                    shuffled=self.privileged_teacher_ema.decode(validation['latent'], shuffled_actions, current[:, train_end:validation_end]),
                    current=current[:, train_end:validation_end],
                    current_contacts=torch.cat([b.privileged_teacher_contacts for b in batches], 1)[:, train_end:validation_end],
                    current_contact_mask=torch.cat([b.privileged_teacher_contact_mask for b in batches], 1)[:, train_end:validation_end])
            targets = training_targets
        ramp = min(1., self.privileged_teacher_updates/self.privileged_teacher.config.warmup_updates)
        return ids, targets, loss.detach(), ramp

    @torch.no_grad()
    def _publish_dynamics(self, batches, observation, starts, actions, target, calibration_begin):
        """Stage EMA and calibration, measure the complete policy, then commit or reject.

        The Actor is held at its current (post-PPO) weights on both sides. PPO's
        own change and this deployment change therefore have separate KL signals.
        Rejection leaves the online learner free to improve on the next window.
        """
        decay = (self.update_config.ema_decay(actions.shape[1]) if self.dynamics_ema_decay is None
                 else self.dynamics_ema_decay)
        candidate = copy.deepcopy(self.actor_dynamics).eval()
        for ema, online in zip(candidate.parameters(), self.dynamics.parameters(), strict=True):
            ema.lerp_(online, 1-decay)
        for ema, online in zip(candidate.buffers(), self.dynamics.buffers(), strict=True):
            ema.copy_(online)
        prefix = batches[0].dynamics_prefix or DynamicsHistory()
        hidden, cores, valid, _ = replay_dynamics_inputs(candidate, prefix, observation, starts, actions)
        encoding = candidate.forward_sequence(observation, episode_start=starts,
            initial_hidden=hidden, previous_predicted_core=cores, previous_prediction_valid=valid)
        mean, logvar = candidate.identify_privileged_distribution(encoding.latent)
        calibration = copy.deepcopy(self.confidence)
        heldout = torch.zeros_like(starts)
        heldout[:, calibration_begin:] = True
        # Retain no-response frames in calibration too: prior-only predictions
        # must be penalized, even when these labels cannot support identification.
        calibration.update(mean, logvar, target, heldout)
        features = dynamics_actor_features(encoding.context, encoding.latent, encoding.contact_tokens)
        kl_values, offset = [], 0
        self.actor.eval()
        for batch in batches:
            length = batch.actions.shape[1]
            if batch.physics_age is None or batch.physics_confidence is None:
                raise ValueError("deployment KL needs collected evidence ages and confidences")
            data = flat_observations_to_tensordict(batch.actor_observation, batch.critic_observation)
            mask = build_episode_causal_mask(batch.dones.squeeze(-1))
            old = self._actor_forward(data, mode="sequence_logp", actions=batch.actions,
                attn_mask=mask, update_obs_norm=False, physics_features=batch.physics_features,
                physics_confidence=batch.physics_confidence)
            old_mu, old_sigma = old.get("mu").clone(), old.get("sigma").clone()
            confidence = calibration(logvar[:, offset:offset+length], batch.physics_age,
                                      self.confidence_warmup_steps)
            new = self._actor_forward(data, mode="sequence_logp", actions=batch.actions,
                attn_mask=mask, update_obs_norm=False, physics_features=features[:, offset:offset+length],
                physics_confidence=confidence)
            kl_values.append(diagonal_gaussian_kl(old_mu, old_sigma, new.get("mu"), new.get("sigma")))
            offset += length
        kls = torch.cat(kl_values, dim=1).clamp_min(0)
        mean_kl, max_kl = float(kls.mean()), float(kls.max())
        accepted = (math.isfinite(mean_kl) and math.isfinite(max_kl)
                    and mean_kl <= self.update_config.maximum_mean_kl
                    and max_kl <= self.update_config.maximum_token_kl)
        if accepted:
            self.actor_dynamics.load_state_dict(candidate.state_dict(), strict=True)
            self.confidence.load_state_dict(calibration.state_dict(), strict=True)
        return dict(deployment_update_accepted=accepted, deployment_mean_kl=mean_kl,
            deployment_max_kl=max_kl, calibration_samples=int(calibration.samples),
            calibration_skill=float((1-calibration.mse/calibration.config.prior_variance).amin()),
            ema_decay=decay)
