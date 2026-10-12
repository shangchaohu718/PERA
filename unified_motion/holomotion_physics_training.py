"""Physics-student and privileged-Teacher training for HoloMotion on mjlab."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .algorithms.sequence_training import parameter_stamp
from .holomotion_mjlab_ppo import (
    HoloMotionMJLabPPO,
    HoloMotionPPORollout,
    generalized_advantage_estimate,
)
from .holomotion_original_algorithm import flat_observations_to_tensordict
from .holomotion_physics_actor import (
    HoloMotionPhysicsActor,
    HoloMotionPhysicsActorConfig,
    dynamics_actor_features,
)
from .models.physics import DynamicsModel, PhysicsObservation


def _stack_physics(items: list[PhysicsObservation]) -> PhysicsObservation:
    if not items:
        raise ValueError("cannot stack an empty physics rollout")
    return PhysicsObservation(
        *(torch.stack([getattr(item, name) for item in items], dim=1) for name in (
            "body_obs", "base_lin_vel_b", "contact_features", "contact_mask",
            "object_features", "object_mask",
        ))
    )


def _concat_physics(items: list[PhysicsObservation]) -> PhysicsObservation:
    return PhysicsObservation(
        *(torch.cat([getattr(item, name) for item in items], dim=1) for name in (
            "body_obs", "base_lin_vel_b", "contact_features", "contact_mask",
            "object_features", "object_mask",
        ))
    )


def _slice_physics(value: PhysicsObservation, begin: int, end: int) -> PhysicsObservation:
    return PhysicsObservation(*(getattr(value, name)[:, begin:end] for name in (
        "body_obs", "base_lin_vel_b", "contact_features", "contact_mask",
        "object_features", "object_mask",
    )))


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
        teacher_reconstruction_weight: float = 0.1,
        teacher_distillation_weight: float = 0.05,
        dynamics_burn_in_steps: int = 64,
        dynamics_train_steps: int = 128,
        confidence_warmup_steps: int = 128,
        dynamics_ema_decay: float = 0.99,
        dynamics_epochs: int = 4,
    ) -> None:
        if not isinstance(dynamics, DynamicsModel):
            raise TypeError("dynamics must be a DynamicsModel")
        if dynamics.robot != task.bridge.robot:
            raise ValueError("HoloMotion task and dynamics robot contracts differ")
        if dynamics.interaction_feedback:
            raise ValueError("first HoloMotion integration does not replay interaction feedback")
        if not dynamics.teacher_student or dynamics.system_id_dim != task.physics_teacher_dim:
            raise ValueError("dynamics Teacher width must match slow-physics target")
        if not dynamics.uncertainty:
            raise ValueError("HoloMotion physics dynamics must enable uncertainty heads")
        super().__init__(algorithm, task, config=config)
        actor_cfg = physics_actor_config or HoloMotionPhysicsActorConfig(
            physics_feature_dim=3 * dynamics.physics_width + dynamics.latent_width
        )
        self.actor = HoloMotionPhysicsActor(self.actor, actor_cfg).to(task.device)
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
        self.system_identification_weight = float(system_identification_weight)
        self.teacher_reconstruction_weight = float(teacher_reconstruction_weight)
        self.teacher_distillation_weight = float(teacher_distillation_weight)
        for value in (
            self.system_identification_weight,
            self.teacher_reconstruction_weight,
            self.teacher_distillation_weight,
        ):
            if value <= 0:
                raise ValueError("all physics/Teacher loss weights must be positive")
        if any(type(value) is not int or value < 1 for value in (
            dynamics_burn_in_steps, dynamics_train_steps, confidence_warmup_steps,
            dynamics_epochs,
        )):
            raise ValueError("burn-in, train, and confidence warmup steps must be positive")
        if not 0.0 < dynamics_ema_decay < 1.0:
            raise ValueError("dynamics_ema_decay must be in (0,1)")
        self.dynamics_burn_in_steps = dynamics_burn_in_steps
        self.dynamics_train_steps = dynamics_train_steps
        self.confidence_warmup_steps = confidence_warmup_steps
        self.dynamics_ema_decay = float(dynamics_ema_decay)
        self.dynamics_epochs = dynamics_epochs
        self._dynamics_rollouts: list[HoloMotionPPORollout] = []
        self._physics_memory = None
        self._physics_episode_start = None
        self._physics_age = None

    @torch.no_grad()
    def reset(self, *, random_start: bool = False, min_remaining_s: float = 0.0):
        observation = super().reset(
            random_start=random_start, min_remaining_s=min_remaining_s
        )
        self._physics_memory = None
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
        if self._observation is None:
            self.reset()
        self.actor_dynamics.eval()
        if self._physics_episode_start is None:
            raise RuntimeError("physics episode state was not initialized")

        if self._physics_age is None:
            raise RuntimeError("physics evidence age was not initialized")
        actor_obs=[];critic_obs=[];actions=[];log_probs=[];means=[];sigmas=[];values=[]
        rewards=[];dones=[];clip_ids=[];features=[];confidences=[];frames=[];next_frames=[];starts=[]
        reward_terms: dict[str,list[torch.Tensor]]={}
        previous_core=[];previous_valid=[];teacher_targets=[]
        aux_base=[];aux_contacts=[];aux_ref=[];aux_robot=[]
        initial_hidden = None
        stamp = parameter_stamp(self.dynamics)
        env_ids = tuple(range(self.task.num_envs))

        for _ in range(num_steps):
            observation = self._observation
            assert observation is not None
            frame = self.task.deployable_physics()
            start = self._physics_episode_start.clone()
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
            encoding, memory = self.actor_dynamics.step(
                frame, episode_start=start, env_ids=env_ids, memory=before
            )
            feature = dynamics_actor_features(
                encoding.context, encoding.latent, encoding.contact_tokens
            )
            teacher_target = self.task.physics_teacher_target()
            _, logvar = self.actor_dynamics.identify_privileged_distribution(
                encoding.latent
            )
            uncertainty_confidence = torch.exp(
                -0.5 * torch.exp(logvar).mean(dim=-1, keepdim=True)
            )
            evidence_confidence = (
                (self._physics_age + 1).to(frame.body_obs.dtype)
                / float(self.confidence_warmup_steps)
            ).clamp(max=1.0).unsqueeze(-1)
            confidence = (uncertainty_confidence * evidence_confidence).clamp(0.0, 1.0)
            data = flat_observations_to_tensordict(observation.actor, observation.critic)
            actor_out = self._actor_forward(
                data, mode="sampling", update_obs_norm=False,
                physics_features=feature, physics_confidence=confidence,
            )
            critic_out = self.critic(data, update_obs_norm=False)
            action = actor_out.get("actions")
            prediction = self.actor_dynamics.predict_next(
                frame, action, encoding.latent,
                physics_encoding=encoding.physics_encoding(),
            )
            self._physics_memory = self.actor_dynamics.remember_prediction(memory, prediction)
            transition = self.task.step(action)
            next_frame = self.task.privileged_physics_observation()
            done = transition.terminated | transition.truncated

            actor_obs.append(observation.actor);critic_obs.append(observation.critic)
            actions.append(action);log_probs.append(actor_out.get("actions_log_prob").reshape(-1,1))
            means.append(actor_out.get("mu"));sigmas.append(actor_out.get("sigma"))
            values.append(critic_out.get("values").reshape(-1,1));rewards.append(transition.reward.reshape(-1,1))
            dones.append(done.reshape(-1,1));clip_ids.append(self.task.sampler.clip_ids.clone())
            for name,value in transition.terms.items():
                reward_terms.setdefault(name,[]).append(value.detach().clone())
            features.append(feature);confidences.append(confidence)
            frames.append(frame);next_frames.append(next_frame);starts.append(start)
            teacher_targets.append(teacher_target)
            aux_base.append(observation.auxiliary.base_lin_vel_b)
            aux_contacts.append(observation.auxiliary.keybody_contacts)
            aux_ref.append(observation.auxiliary.ref_keybody_rel_pos)
            aux_robot.append(observation.auxiliary.robot_keybody_rel_pos)

            if done.any().item():
                ids = done.nonzero(as_tuple=False).flatten()
                self.actor.clear_env_cache(ids)
                self._observation = self.task.reset(
                    ids, random_start=True,
                    min_remaining_s=float(self.task.sampler.offsets_s[-1]),
                )
            else:
                self._observation = self.task.observe()
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
        )
        if self.config.normalize_advantage:
            advantage=(advantage-advantage.mean())/advantage.std().clamp_min(1e-8)
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
        )

    def update_dynamics(
        self, rollout: HoloMotionPPORollout
    ) -> HoloMotionDynamicsMetrics | None:
        required=(rollout.physics_observation,rollout.next_physics_observation,
                  rollout.physics_episode_start,rollout.physics_initial_hidden,
                  rollout.previous_predicted_core,rollout.previous_prediction_valid,
                  rollout.physics_teacher_target,rollout.dynamics_stamp)
        if any(value is None for value in required):
            raise ValueError("rollout does not contain physics/Teacher supervision")
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
        actions = torch.cat([item.actions for item in batches], dim=1)
        burn = self.dynamics_burn_in_steps
        end = required_steps
        self.dynamics.train()
        try:
            train_observation=_slice_physics(observation,burn,end)
            train_next=_slice_physics(next_observation,burn,end)
            target=teacher_target[:,burn:end].detach()
            # The randomized parameters exist from the first frame of every
            # episode; prediction validity is unrelated to Teacher labels.
            mask=torch.ones_like(previous_valid[:,burn:end])
            count=mask.sum().clamp_min(1)
            totals=[]
            for _ in range(self.dynamics_epochs):
                # Recompute both burn-in and the differentiable window after
                # every optimizer step; no activation graph is reused with
                # changed Dynamics weights.
                with torch.no_grad():
                    burn_encoding=self.dynamics.forward_sequence(
                        _slice_physics(observation,0,burn),
                        episode_start=episode_start[:,:burn],
                        initial_hidden=batches[0].physics_initial_hidden.detach(),
                        previous_predicted_core=previous_core[:,:burn].detach(),
                        previous_prediction_valid=previous_valid[:,:burn],
                    )
                encoding=self.dynamics.forward_sequence(
                    train_observation,
                    episode_start=episode_start[:,burn:end],
                    initial_hidden=burn_encoding.hidden.detach(),
                    previous_predicted_core=previous_core[:,burn:end].detach(),
                    previous_prediction_valid=previous_valid[:,burn:end],
                )
                prediction=self.dynamics.predict_next(
                    train_observation,actions[:,burn:end].detach(),encoding.latent,
                    physics_encoding=encoding.physics_encoding(),
                )
                valid=torch.ones_like(episode_start[:,burn:end])
                prediction_loss=self.dynamics.prediction_loss(
                    prediction,train_next,valid_transition=valid
                )
                identified,id_logvar=self.dynamics.identify_privileged_distribution(
                    encoding.latent
                )
                id_nll=0.5*((identified-target).square()*torch.exp(-id_logvar)+id_logvar).mean(-1)
                system_id=(id_nll*mask).sum()/count
                teacher=self.dynamics.encode_privileged_teacher(target)
                reconstruction=(self.dynamics.reconstruct_privileged(teacher)-target).square().mean()
                distillation=((encoding.latent-teacher.detach()).square().mean(-1)*mask).sum()/count
                contact_labels=train_next.contact_features[...,10]
                contact_bce=F.binary_cross_entropy_with_logits(
                    prediction.next_contact_logits,contact_labels,reduction="none"
                )
                left_contact=contact_bce[...,0].mean();right_contact=contact_bce[...,1].mean()
                total=(prediction_loss.total+self.system_identification_weight*system_id
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
            with torch.no_grad():
                decay=self.dynamics_ema_decay
                for ema,online in zip(
                    self.actor_dynamics.parameters(),self.dynamics.parameters(),strict=True
                ):
                    ema.mul_(decay).add_(online,alpha=1.0-decay)
                for ema,online in zip(
                    self.actor_dynamics.buffers(),self.dynamics.buffers(),strict=True
                ):
                    ema.copy_(online)
            averaged=[float(torch.stack([row[i] for row in totals]).mean())
                      for i in range(len(totals[0]))]
            return HoloMotionDynamicsMetrics(*averaged,self.dynamics_epochs)
        finally:
            self._dynamics_rollouts.clear()
            self.dynamics.invalidate_memories();self.dynamics.eval()
            self.actor_dynamics.invalidate_memories();self.actor_dynamics.eval()
            # New weights invalidate the online GRU state.  The next collect
            # performs an explicit environment reset rather than inventing history.
            self._observation=None;self._physics_memory=None;self._physics_episode_start=None
            self._physics_age=None
