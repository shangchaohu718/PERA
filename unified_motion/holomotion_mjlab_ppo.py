"""Original HoloMotion sequence-PPO core running on the mjlab task boundary.

This intentionally reuses the original trainable actor and critic modules.  It
only replaces the IsaacLab-specific rollout plumbing.  The auxiliary losses
and curriculum are not part of this first integration slice; the PPO
surrogate, clipped value loss, entropy bonus, GAE, sequence replay, and
optimizer grouping match the original implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .algorithms.common import compute_gae, diagonal_gaussian_kl
from .holomotion_mjlab_task import (
    HoloMotionK1MJLabTask,
    HoloMotionMJLabObservation,
)
from .holomotion_original_algorithm import (
    OriginalHoloMotionK1Algorithm,
    flat_observations_to_tensordict,
)


@dataclass(frozen=True)
class HoloMotionPPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    actor_learning_rate: float = 3.0e-5
    critic_learning_rate: float = 5.0e-5
    actor_beta1: float = 0.9
    actor_beta2: float = 0.999
    critic_beta1: float = 0.9
    critic_beta2: float = 0.999
    use_clipped_value_loss: bool = True
    normalize_advantage: bool = True
    normalize_advantage_per_mini_batch: bool = False
    num_learning_epochs: int = 3
    num_mini_batches: int = 24
    desired_kl: float | None = 0.01
    schedule: str = "adaptive"
    kl_coef: float = 0.0
    adaptive_lr_adapt_critic: bool = False
    adaptive_lr_factor: float = 1.2
    adaptive_lr_kl_high_factor: float = 2.0
    adaptive_lr_kl_low_factor: float = 0.5
    adaptive_lr_min: float = 1.0e-6
    adaptive_lr_max: float = 1.0
    kl_early_stop_enabled: bool = True
    kl_early_stop_window_size: int = 3
    kl_early_stop_factor: float = 1.8
    kl_early_stop_min_updates: int = 1
    aux_state_pred_enabled: bool = True
    aux_w_base_lin_vel: float = 0.01
    aux_w_keybody_contact: float = 0.01
    aux_w_ref_keybody_rel_pos: float = 0.1
    aux_w_robot_keybody_rel_pos: float = 0.1
    aux_min_std: float = 0.01
    aux_max_std: float = 2.0
    inactive_expert_margin_enabled: bool = True
    inactive_expert_margin_weight: float = 10.0

    @classmethod
    def from_original(cls, config: Any) -> "HoloMotionPPOConfig":
        source = config.algo.config
        values: dict[str, Any] = {}
        for field in cls.__dataclass_fields__:
            if field == "normalize_advantage":
                values[field] = bool(source.get("global_advantage_norm", True))
            elif field in source:
                values[field] = source[field]
        adaptive = source.get("adaptive_lr", {})
        values.update(
            adaptive_lr_adapt_critic=bool(adaptive.get("adapt_critic", False)),
            adaptive_lr_factor=float(adaptive.get("lr_scaler", 1.2)),
            adaptive_lr_kl_high_factor=float(adaptive.get("kl_high_factor", 2.0)),
            adaptive_lr_kl_low_factor=float(adaptive.get("kl_low_factor", 0.5)),
            adaptive_lr_min=float(adaptive.get("min_learning_rate", 1.0e-6)),
            adaptive_lr_max=float(adaptive.get("max_learning_rate", 1.0)),
        )
        early = source.get("distributed_update", {}).get("kl_early_stop", {})
        values.update(
            kl_early_stop_enabled=bool(early.get("enabled", False)),
            kl_early_stop_window_size=int(early.get("window_size", 3)),
            kl_early_stop_factor=float(early.get("factor", 2.0)),
            kl_early_stop_min_updates=int(early.get("min_updates", 1)),
        )
        auxiliary = source.get("aux_state_pred", {})
        values.update(
            aux_state_pred_enabled=bool(auxiliary.get("enabled", False)),
            aux_w_base_lin_vel=float(auxiliary.get("w_base_lin_vel", 0.0)),
            aux_w_keybody_contact=float(auxiliary.get("w_keybody_contact", 0.0)),
            aux_w_ref_keybody_rel_pos=float(
                auxiliary.get("w_ref_keybody_rel_pos", 0.0)
            ),
            aux_w_robot_keybody_rel_pos=float(
                auxiliary.get("w_robot_keybody_rel_pos", 0.0)
            ),
            aux_min_std=float(auxiliary.get("min_std", 1.0e-3)),
            aux_max_std=float(auxiliary.get("max_std", 5.0)),
        )
        inactive = source.get("inactive_expert_margin_to_topk", {})
        values.update(
            inactive_expert_margin_enabled=bool(inactive.get("enabled", False)),
            inactive_expert_margin_weight=float(inactive.get("weight", 0.0)),
        )
        return cls(**values)


@dataclass(frozen=True)
class HoloMotionPPORollout:
    actor_observation: torch.Tensor
    critic_observation: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_mu: torch.Tensor
    old_sigma: torch.Tensor
    old_value: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    gt_base_lin_vel_b: torch.Tensor
    gt_keybody_contacts: torch.Tensor
    gt_ref_keybody_rel_pos: torch.Tensor
    gt_robot_keybody_rel_pos: torch.Tensor
    clip_ids: torch.Tensor
    physics_features: torch.Tensor | None = None
    physics_confidence: torch.Tensor | None = None
    physics_observation: Any | None = None
    next_physics_observation: Any | None = None
    physics_episode_start: torch.Tensor | None = None
    physics_initial_hidden: torch.Tensor | None = None
    previous_predicted_core: torch.Tensor | None = None
    previous_prediction_valid: torch.Tensor | None = None
    physics_teacher_target: torch.Tensor | None = None
    dynamics_stamp: tuple[int, ...] | None = None
    reward_terms: dict[str, torch.Tensor] | None = None
    truncated: torch.Tensor | None = None
    timeout_values: torch.Tensor | None = None
    executed_actions: torch.Tensor | None = None
    dynamics_prefix: Any | None = None
    physics_age: torch.Tensor | None = None
    privileged_teacher_input: torch.Tensor | None = None
    privileged_teacher_core: torch.Tensor | None = None
    privileged_teacher_env_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class HoloMotionPPOUpdateMetrics:
    actor_loss: float
    critic_loss: float
    surrogate_loss: float
    value_loss: float
    entropy: float
    ratio_mean: float
    actor_grad_norm: float
    critic_grad_norm: float
    analytic_kl: float
    clip_fraction: float
    num_updates: int
    kl_early_stopped: bool
    actor_learning_rate: float
    critic_learning_rate: float
    aux_base_lin_vel_nll: float
    aux_keybody_contact_bce: float
    aux_ref_keybody_rel_pos_mse: float
    aux_robot_keybody_rel_pos_mse: float
    inactive_expert_margin_loss: float


@dataclass(frozen=True)
class HoloMotionPPOResumeState:
    iteration: int
    checkpoint: Path


def build_episode_causal_mask(dones: torch.Tensor) -> torch.Tensor:
    """Match PPOTF._build_episode_causal_mask for a [B,T] done tensor."""
    if dones.ndim != 2 or dones.dtype != torch.bool:
        raise ValueError("dones must be a boolean [B,T] tensor")
    steps = dones.shape[1]
    done_ids = dones.to(torch.long)
    segment = torch.cumsum(done_ids, dim=1) - done_ids
    same_segment = segment[:, :, None] == segment[:, None, :]
    causal = torch.tril(
        torch.ones(steps, steps, dtype=torch.bool, device=dones.device)
    )
    return same_segment & causal


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    truncated: torch.Tensor | None = None,
    timeout_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt [B,T,1] rollouts to the shared terminal-aware GAE implementation.

    ``dones`` cuts the advantage chain for both kinds of episode boundary.
    Timeouts bootstrap from their pre-reset critic values, never the next
    episode's observation. Legacy callers without timeouts keep their behavior.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have identical shapes")
    if rewards.ndim != 3 or rewards.shape[-1] != 1:
        raise ValueError("rollout tensors must have shape [B,T,1]")
    if bootstrap_value.shape != rewards.shape[:1] + (1,):
        raise ValueError("bootstrap_value must have shape [B,1]")
    if dones.dtype != torch.bool:
        raise ValueError("dones must be boolean")
    if (truncated is None) != (timeout_values is None):
        raise ValueError("truncated and timeout_values must be supplied together")
    next_values = torch.cat((values[:, 1:], bootstrap_value[:, None]), dim=1)
    if truncated is None:
        truncated = torch.zeros_like(dones)
    else:
        if (truncated.shape != dones.shape or truncated.dtype != torch.bool
                or truncated.device != dones.device):
            raise ValueError("truncated must match dones shape, device, and boolean dtype")
        if (truncated & ~dones).any().item():
            raise ValueError("every truncated transition must also be done")
        if (timeout_values.shape != values.shape or timeout_values.dtype != values.dtype
                or timeout_values.device != values.device):
            raise ValueError("timeout_values must match values shape, device, and dtype")
        next_values = torch.where(truncated, timeout_values, next_values)
    advantage, returns = compute_gae(
        rewards.squeeze(-1).T, values.squeeze(-1).T, next_values.squeeze(-1).T,
        (dones & ~truncated).squeeze(-1).T, truncated.squeeze(-1).T,
        gamma=gamma, gae_lambda=lam,
    )
    return advantage.T.unsqueeze(-1), returns.T.unsqueeze(-1)


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Keep sample-std normalization, including finite singleton smoke batches."""
    correction = 1 if advantages.numel() > 1 else 0
    return (advantages - advantages.mean()) / advantages.std(
        correction=correction
    ).clamp_min(1.0e-8)


class HoloMotionMJLabPPO:
    """Small, auditable bridge from mjlab rollouts to the original PPOTF nets."""

    def __init__(
        self,
        algorithm: OriginalHoloMotionK1Algorithm,
        task: HoloMotionK1MJLabTask,
        *,
        config: HoloMotionPPOConfig | None = None,
    ) -> None:
        self.actor = algorithm.actor
        self._network_config = getattr(algorithm, "network_config", None)
        self.critic = algorithm.critic
        self.task = task
        self.config = config or HoloMotionPPOConfig.from_original(algorithm.config)
        if self.config.num_learning_epochs < 1 or self.config.num_mini_batches < 1:
            raise ValueError("PPO epochs and mini-batches must be positive")
        if next(self.actor.parameters()).device != task.device:
            raise ValueError("actor and mjlab task must be on the same device")
        if next(self.critic.parameters()).device != task.device:
            raise ValueError("critic and mjlab task must be on the same device")

        decay, no_decay = [], []
        for name, parameter in self.actor.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                parameter.ndim < 2
                or "log_std" in name
                or "bias" in name
                or "norm" in name
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        self.actor_optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.01},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.actor_learning_rate,
            betas=(self.config.actor_beta1, self.config.actor_beta2),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.config.critic_learning_rate,
            betas=(self.config.critic_beta1, self.config.critic_beta2),
        )
        self._observation: HoloMotionMJLabObservation | None = None
        self._policy_rollout_steps = 32

    def _actor_forward(
        self,
        data: Any,
        *,
        physics_features: torch.Tensor | None = None,
        physics_confidence: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if physics_features is not None:
            kwargs["physics_features"] = physics_features
        if physics_confidence is not None:
            kwargs["physics_confidence"] = physics_confidence
        return self.actor(data, **kwargs)

    @staticmethod
    def _checkpoint_state_to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {
                key: HoloMotionMJLabPPO._checkpoint_state_to_cpu(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                HoloMotionMJLabPPO._checkpoint_state_to_cpu(item)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                HoloMotionMJLabPPO._checkpoint_state_to_cpu(item)
                for item in value
            )
        return value

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        iteration: int,
        infos: dict[str, Any] | None = None,
    ) -> Path:
        """Keep the original directory layout and bundle full Physics state in .pt."""
        from safetensors.torch import save_file

        target = Path(path).expanduser().resolve()
        if hasattr(self, "dynamics") and getattr(self, "_network_config", None) is None:
            raise ValueError("complete Physics checkpoint requires resolved network_config")
        if target.suffix != ".pt":
            raise ValueError("checkpoint path must end in .pt")
        model_dir = target.with_suffix("")
        if target.exists() or model_dir.exists():
            raise FileExistsError(f"checkpoint target already exists: {target}")
        actor_dir = model_dir / "actor"
        critic_dir = model_dir / "critic"
        actor_dir.mkdir(parents=True)
        critic_dir.mkdir(parents=True)

        def state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
            return {
                key.removeprefix("_orig_mod."): value.detach().cpu().contiguous()
                for key, value in module.state_dict().items()
                if isinstance(value, torch.Tensor)
            }

        try:
            checkpoint_actor = getattr(self.actor, "base_actor", self.actor)
            save_file(state(checkpoint_actor), actor_dir / "model.safetensors")
            save_file(state(self.critic), critic_dir / "model.safetensors")
            payload = {
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "iter": int(iteration),
                "infos": infos,
            }
            if getattr(self, "_network_config", None) is not None:
                contract = getattr(getattr(self.task, "adapter", None), "contract", None)
                foot = getattr(self.task, "foot_reward_config", None)
                randomization = getattr(self.task, "physics_randomization", None)
                payload.update(policy_schema=1, network_config=self._network_config,
                    base_actor_state_dict=checkpoint_actor.state_dict(),
                    critic_state_dict=self.critic.state_dict(), ppo_config=asdict(self.config),
                    policy_runtime_config=dict(rollout_steps=self._policy_rollout_steps,
                        task_contract=None if contract is None else asdict(contract),
                        foot_reward=None if foot is None else asdict(foot),
                        physics_randomization=None if randomization is None else asdict(randomization)))
            if hasattr(self, "dynamics"):
                payload.update(
                    physics_schema=3,
                    physics_residual_state_dict=self.actor.physics_residual.state_dict(),
                    dynamics_state_dict=self.dynamics.state_dict(),
                    actor_dynamics_state_dict=self.actor_dynamics.state_dict(),
                    dynamics_optimizer_state_dict=self.dynamics_optimizer.state_dict(),
                    physics_training_config={
                        "dynamics_burn_in_steps": self.dynamics_burn_in_steps,
                        "dynamics_train_steps": self.dynamics_train_steps,
                        "confidence_warmup_steps": self.confidence_warmup_steps,
                        "dynamics_ema_decay": self.dynamics_ema_decay,
                        "dynamics_epochs": self.dynamics_epochs,
                    },
                )
                if getattr(self, "_network_config", None) is not None:
                    from .holomotion_physics_runtime import dynamics_configuration
                    contract = getattr(getattr(self.task, "adapter", None), "contract", None)
                    foot = getattr(self.task, "foot_reward_config", None)
                    randomization = getattr(self.task, "physics_randomization", None)
                    payload.update(
                        physics_schema=5,
                        base_actor_state_dict=checkpoint_actor.state_dict(),
                        critic_state_dict=self.critic.state_dict(),
                        ppo_config=asdict(self.config),
                        physics_update_config=asdict(self.update_config),
                        confidence_state_dict=self.confidence.state_dict(),
                        physics_runtime_config=dict(schema=2, network_config=self._network_config,
                            physics_actor_config=asdict(self.actor.config),
                            dynamics_config=dynamics_configuration(self.dynamics),
                            confidence_config=asdict(self.confidence.config),
                            confidence_warmup_steps=self.confidence_warmup_steps,
                            rollout_steps=getattr(self, "_policy_rollout_steps", 32),
                            task_contract=None if contract is None else asdict(contract),
                            foot_reward=None if foot is None else asdict(foot),
                            physics_randomization=None if randomization is None else asdict(randomization)),
                        physics_loss_weights=dict(
                            system_identification_weight=self.system_identification_weight,
                            teacher_reconstruction_weight=self.teacher_reconstruction_weight,
                            teacher_distillation_weight=self.teacher_distillation_weight),
                    )
            if getattr(self, 'privileged_teacher', None) is not None:
                payload['privileged_teacher'] = self.privileged_teacher_checkpoint()
                payload['physics_runtime_config']['privileged_teacher_config'] = payload['privileged_teacher']['config']
            sampler = getattr(getattr(self, 'task', None), 'sampler', None)
            sampling_weights = getattr(sampler, 'sampling_weights', None)
            if sampling_weights is not None:
                for runtime_name in ('policy_runtime_config', 'physics_runtime_config'):
                    if runtime_name in payload:
                        payload[runtime_name]['motion_sampling_weights'] = sampling_weights.detach().cpu().tolist()
            if hasattr(self, 'training_adjustment'):
                payload['training_adjustment'] = self.training_adjustment
            payload['training_source_checkpoint_sha256'] = getattr(self, 'training_source_checkpoint_sha256', None)
            payload['training_seed'] = getattr(self, 'training_seed', None)
            payload['training_protocol'] = getattr(self, 'training_protocol', None)
            if hasattr(self, 'training_compute_seconds'):
                payload['training_cost'] = dict(compute_seconds=self.training_compute_seconds,
                    transitions=self.training_transitions)
            torch.save(self._checkpoint_state_to_cpu(payload), target)
        except Exception:
            # Leave an obviously incomplete directory rather than publishing a
            # misleading optimizer file. A later run will refuse to overwrite it.
            if target.exists():
                target.unlink()
            raise
        return target

    def load_optimizer_checkpoint(
        self, checkpoint: str | Path, *, _validated_physics=None
    ) -> HoloMotionPPOResumeState:
        """Restore the two original AdamW states from a HoloMotion .pt file."""
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = (_validated_physics.payload if _validated_physics is not None else
                   torch.load(path, map_location=self.task.device, weights_only=True))
        if not isinstance(payload, dict):
            raise ValueError("optimizer checkpoint must contain a dictionary")
        required = {"actor_optimizer_state_dict", "critic_optimizer_state_dict"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"optimizer checkpoint is missing keys: {sorted(missing)}")
        if hasattr(self, "dynamics"):
            if payload.get("physics_schema") != 5:
                raise ValueError("training resume requires physics schema 5; legacy schema 4 is frozen-evaluation only")
            physics_required = {
                "physics_residual_state_dict",
                "dynamics_state_dict",
                "actor_dynamics_state_dict",
                "dynamics_optimizer_state_dict",
                "physics_training_config",
                "physics_update_config", "confidence_state_dict",
            }
            missing_physics = physics_required.difference(payload)
            if missing_physics:
                raise ValueError(
                    "physics trainer cannot resume a checkpoint without physics state: "
                    f"{sorted(missing_physics)}"
                )
        iteration = payload.get("iter", -1)
        if type(iteration) is not int or iteration < 0:
            raise ValueError("optimizer checkpoint has no valid iteration")
        if not hasattr(self, "dynamics") and payload.get("physics_schema") is not None:
            raise ValueError("a Physics checkpoint requires a Physics trainer")
        if not hasattr(self, "dynamics") and payload.get("policy_schema") == 1:
            from .k1_policy_runtime import load_k1_checkpoint
            from .holomotion_physics_runtime import validate_task_contract
            loaded = load_k1_checkpoint(path, device=self.task.device)
            validate_task_contract(payload["policy_runtime_config"], self.task)
            if self._observation is not None:
                raise ValueError("load a checkpoint before starting environment collection")
            if payload["network_config"] != self._network_config or payload["ppo_config"] != asdict(self.config):
                raise ValueError("checkpoint model or PPO configuration differs from this run")
            self.actor.load_state_dict(loaded.actor.state_dict(), strict=True)
            self.critic.load_state_dict(loaded.critic.state_dict(), strict=True)
            self._policy_rollout_steps = payload["policy_runtime_config"]["rollout_steps"]
        if hasattr(self, "dynamics"):
            expected_physics_config = {
                "dynamics_burn_in_steps": self.dynamics_burn_in_steps,
                "dynamics_train_steps": self.dynamics_train_steps,
                "confidence_warmup_steps": self.confidence_warmup_steps,
                "dynamics_ema_decay": self.dynamics_ema_decay,
                "dynamics_epochs": self.dynamics_epochs,
            }
            if payload["physics_training_config"] != expected_physics_config:
                raise ValueError("physics checkpoint training configuration differs from this run")
            if getattr(self, "_observation", None) is not None:
                raise ValueError("load a checkpoint before starting environment collection")
            if payload["physics_schema"] == 5:
                from .holomotion_physics_runtime import (
                    load_physics_checkpoint, dynamics_configuration, validate_task_contract,
                )
                loaded = (_validated_physics if _validated_physics is not None else
                          load_physics_checkpoint(path, device=self.task.device))
                runtime = payload["physics_runtime_config"]
                validate_task_contract(runtime, self.task)
                expected_losses = dict(system_identification_weight=self.system_identification_weight,
                    teacher_reconstruction_weight=self.teacher_reconstruction_weight,
                    teacher_distillation_weight=self.teacher_distillation_weight)
                if payload["ppo_config"] != asdict(self.config) or payload["physics_loss_weights"] != expected_losses:
                    raise ValueError("physics checkpoint optimization configuration differs from this run")
                from .physics_confidence import PhysicsConfidenceConfig
                if (payload["physics_update_config"] != asdict(self.update_config)
                        or PhysicsConfidenceConfig(**runtime["confidence_config"]) != self.confidence.config):
                    raise ValueError("physics update/confidence configuration differs from this run")
                if (runtime["physics_actor_config"] != asdict(self.actor.config)
                        or runtime["dynamics_config"] != dynamics_configuration(self.dynamics)
                        or runtime["network_config"] != self._network_config):
                    raise ValueError("physics checkpoint model configuration differs from this run")
                self.actor.load_state_dict(loaded.actor.state_dict(), strict=True)
                self.critic.load_state_dict(loaded.critic.state_dict(), strict=True)
                self.confidence.load_state_dict(loaded.calibration.state_dict(), strict=True)
        saved_runtime = payload.get('physics_runtime_config', payload.get('policy_runtime_config', {}))
        if saved_runtime.get('motion_sampling_weights') is not None:
            self.task.sampler.set_sampling_weights(saved_runtime['motion_sampling_weights'])
        if 'training_adjustment' in payload:
            self.training_adjustment = payload['training_adjustment']
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        if hasattr(self, "dynamics"):
            self.actor.physics_residual.load_state_dict(
                payload["physics_residual_state_dict"], strict=True
            )
            self.dynamics.load_state_dict(payload["dynamics_state_dict"], strict=True)
            self.actor_dynamics.load_state_dict(
                payload["actor_dynamics_state_dict"], strict=True
            )
            self.dynamics_optimizer.load_state_dict(
                payload["dynamics_optimizer_state_dict"]
            )
            self.restore_privileged_teacher(payload)
            self.dynamics.invalidate_memories()
            self.actor_dynamics.invalidate_memories()
            self._dynamics_rollouts.clear()
        return HoloMotionPPOResumeState(iteration=iteration, checkpoint=path)

    @torch.no_grad()
    def reset(
        self,
        *,
        random_start: bool = False,
        min_remaining_s: float = 0.0,
    ) -> HoloMotionMJLabObservation:
        self.actor.reset_kv_cache(self.task.num_envs, self.task.device)
        self._observation = self.task.reset(
            random_start=random_start, min_remaining_s=min_remaining_s
        )
        return self._observation

    @torch.no_grad()
    def collect(self, num_steps: int) -> HoloMotionPPORollout:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self._policy_rollout_steps = num_steps
        if self._observation is None:
            self.reset()
        # PPO sequence_logp replays only this rollout, without a history prefix.
        self.actor.reset_kv_cache(self.task.num_envs, self.task.device)

        actor_obs, critic_obs = [], []
        actions, log_probs, means, sigmas, values = [], [], [], [], []
        rewards, dones, truncated, timeout_values = [], [], [], []
        reward_terms: dict[str, list[torch.Tensor]] = {}
        clip_ids = []
        aux_base_vel, aux_contacts, aux_ref_pos, aux_robot_pos = [], [], [], []
        for _ in range(num_steps):
            observation = self._observation
            assert observation is not None
            clip_ids.append(self.task.sampler.clip_ids.clone())
            data = flat_observations_to_tensordict(
                observation.actor, observation.critic
            )
            actor_out = self._actor_forward(
                data, mode="sampling", update_obs_norm=False
            )
            critic_out = self.critic(data, update_obs_norm=False)
            action = actor_out.get("actions")
            transition = self.task.step(action)
            done = transition.terminated | transition.truncated

            actor_obs.append(observation.actor)
            critic_obs.append(observation.critic)
            actions.append(action)
            log_probs.append(actor_out.get("actions_log_prob").reshape(-1, 1))
            means.append(actor_out.get("mu"))
            sigmas.append(actor_out.get("sigma"))
            values.append(critic_out.get("values").reshape(-1, 1))
            rewards.append(transition.reward.reshape(-1, 1))
            dones.append(done.reshape(-1, 1))
            truncated.append(transition.truncated.reshape(-1, 1))
            for name, value in transition.terms.items():
                reward_terms.setdefault(name, []).append(value.detach().clone())
            aux_base_vel.append(observation.auxiliary.base_lin_vel_b)
            aux_contacts.append(observation.auxiliary.keybody_contacts)
            aux_ref_pos.append(observation.auxiliary.ref_keybody_rel_pos)
            aux_robot_pos.append(observation.auxiliary.robot_keybody_rel_pos)

            timeout_values.append(self._finish_transition(transition))

        final = self._observation
        assert final is not None
        final_data = flat_observations_to_tensordict(final.actor, final.critic)
        bootstrap = self.critic(final_data, update_obs_norm=False).get("values")
        bootstrap = bootstrap.reshape(-1, 1)

        def stack(items: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(items, dim=1)

        reward_tensor = stack(rewards)
        value_tensor = stack(values)
        done_tensor = stack(dones).bool()
        advantage, returns = generalized_advantage_estimate(
            reward_tensor,
            value_tensor,
            done_tensor,
            bootstrap,
            gamma=self.config.gamma,
            lam=self.config.lam,
            truncated=stack(truncated),
            timeout_values=stack(timeout_values),
        )
        if self.config.normalize_advantage:
            advantage = normalize_advantages(advantage)
        return HoloMotionPPORollout(
            actor_observation=stack(actor_obs),
            critic_observation=stack(critic_obs),
            actions=stack(actions),
            old_log_prob=stack(log_probs),
            old_mu=stack(means),
            old_sigma=stack(sigmas),
            old_value=value_tensor,
            returns=returns,
            advantages=advantage,
            dones=done_tensor,
            rewards=reward_tensor,
            gt_base_lin_vel_b=stack(aux_base_vel),
            gt_keybody_contacts=stack(aux_contacts),
            gt_ref_keybody_rel_pos=stack(aux_ref_pos),
            gt_robot_keybody_rel_pos=stack(aux_robot_pos),
            clip_ids=stack(clip_ids),
            reward_terms={name: stack(values) for name, values in reward_terms.items()},
            truncated=stack(truncated),
            timeout_values=stack(timeout_values),
        )

    @torch.no_grad()
    def _finish_transition(self, transition) -> torch.Tensor:
        """Read timeout values before reset, then prepare exactly one observation.

        Only timeout rows are read: other rows may already have exhausted their
        reference clip. The critic-only task read must not advance actor history
        or the task's observe/step handshake.
        """
        timeout_values = transition.reward.new_zeros((self.task.num_envs, 1))
        if (transition.terminated & transition.truncated).any().item():
            raise ValueError("terminated and truncated must be disjoint")
        if transition.truncated.any().item():
            ids = transition.truncated.nonzero(as_tuple=False).flatten()
            critic_obs = self.task.critic_observation(ids)
            data = flat_observations_to_tensordict(
                critic_obs.new_zeros((len(ids), 506)), critic_obs
            )
            timeout_values[ids] = self.critic(
                data, update_obs_norm=False
            ).get("values").reshape(-1, 1)
        done = transition.terminated | transition.truncated
        if done.any().item():
            ids = done.nonzero(as_tuple=False).flatten()
            self.actor.clear_env_cache(ids)
            self._observation = self.task.reset(
                ids, random_start=True,
                min_remaining_s=float(self.task.sampler.offsets_s[-1]),
            )
        else:
            self._observation = self.task.observe()
        return timeout_values

    @staticmethod
    def _analytic_kl(
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        new_mu: torch.Tensor,
        new_sigma: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            value = torch.sum(
                torch.log((new_sigma + 1.0e-8) / (old_sigma + 1.0e-8))
                + (old_sigma.square() + (old_mu - new_mu).square())
                / (2.0 * new_sigma.square() + 1.0e-8)
                - 0.5,
                dim=-1,
            )
        return float(value.mean())

    def _apply_adaptive_lr(self, kl_signal: float | None) -> None:
        cfg = self.config
        if cfg.desired_kl is None or cfg.schedule != "adaptive" or kl_signal is None:
            return
        actor_lr = float(self.actor_optimizer.param_groups[0]["lr"])
        critic_lr = float(self.critic_optimizer.param_groups[0]["lr"])
        if kl_signal > cfg.desired_kl * cfg.adaptive_lr_kl_high_factor:
            actor_lr = max(cfg.adaptive_lr_min, actor_lr / cfg.adaptive_lr_factor)
            if cfg.adaptive_lr_adapt_critic:
                critic_lr = max(
                    cfg.adaptive_lr_min, critic_lr / cfg.adaptive_lr_factor
                )
        elif 0.0 < kl_signal < cfg.desired_kl * cfg.adaptive_lr_kl_low_factor:
            actor_lr = min(cfg.adaptive_lr_max, actor_lr * cfg.adaptive_lr_factor)
            if cfg.adaptive_lr_adapt_critic:
                critic_lr = min(
                    cfg.adaptive_lr_max, critic_lr * cfg.adaptive_lr_factor
                )
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_lr

    def update(self, rollout: HoloMotionPPORollout) -> HoloMotionPPOUpdateMetrics:
        cfg = self.config
        num_envs = rollout.actions.shape[0]
        mini_batches = min(cfg.num_mini_batches, num_envs)
        envs_per_batch = (num_envs + mini_batches - 1) // mini_batches
        permutation = torch.randperm(num_envs, device=rollout.actions.device)
        totals = {
            "actor": 0.0,
            "critic": 0.0,
            "surrogate": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "ratio": 0.0,
            "actor_norm": 0.0,
            "critic_norm": 0.0,
            "kl": 0.0,
            "clip": 0.0,
            "aux_base": 0.0,
            "aux_contact": 0.0,
            "aux_ref": 0.0,
            "aux_robot": 0.0,
            "inactive_margin": 0.0,
        }
        updates = 0
        kl_measurements = 0
        recent_kls: list[float] = []
        kl_windowed = None
        early_stopped = False

        for _ in range(cfg.num_learning_epochs):
            for start in range(0, num_envs, envs_per_batch):
                ids = permutation[start : start + envs_per_batch]
                data = flat_observations_to_tensordict(
                    rollout.actor_observation[ids], rollout.critic_observation[ids]
                )
                actions = rollout.actions[ids]
                old_log_prob = rollout.old_log_prob[ids]
                advantages = rollout.advantages[ids]
                if cfg.normalize_advantage_per_mini_batch:
                    advantages = normalize_advantages(advantages)
                mask = build_episode_causal_mask(
                    rollout.dones[ids].squeeze(-1)
                )
                features = (
                    None
                    if rollout.physics_features is None
                    else rollout.physics_features[ids]
                )
                confidence = (
                    None
                    if rollout.physics_confidence is None
                    else rollout.physics_confidence[ids]
                )
                actor_out = self._actor_forward(
                    data,
                    actions=actions,
                    mode="sequence_logp",
                    attn_mask=mask,
                    update_obs_norm=False,
                    physics_features=features,
                    physics_confidence=confidence,
                )
                batch, steps = actions.shape[:2]
                critic_out = self.critic(data.flatten(0, 1), update_obs_norm=False)
                value = critic_out.get("values").reshape(batch, steps, 1)
                new_log_prob = actor_out.get("actions_log_prob")
                entropy = actor_out.get("entropy")
                ratio = torch.exp(new_log_prob.float() - old_log_prob.float())

                analytic_kl = self._analytic_kl(
                    rollout.old_mu[ids].float(),
                    rollout.old_sigma[ids].float(),
                    actor_out.get("mu").float(),
                    actor_out.get("sigma").float(),
                )
                recent_kls.append(analytic_kl)
                recent_kls = recent_kls[-cfg.kl_early_stop_window_size :]
                kl_measurements += 1
                if len(recent_kls) == cfg.kl_early_stop_window_size:
                    kl_windowed = sum(recent_kls) / len(recent_kls)
                required = max(
                    cfg.kl_early_stop_min_updates,
                    cfg.kl_early_stop_window_size,
                )
                if (
                    cfg.kl_early_stop_enabled
                    and cfg.desired_kl is not None
                    and kl_windowed is not None
                    and kl_measurements >= required
                    and kl_windowed > cfg.desired_kl * cfg.kl_early_stop_factor
                ):
                    early_stopped = True
                    break

                surrogate_1 = ratio * advantages
                surrogate_2 = ratio.clamp(
                    1.0 - cfg.clip_param, 1.0 + cfg.clip_param
                ) * advantages
                surrogate_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
                old_value = rollout.old_value[ids]
                returns = rollout.returns[ids]
                if cfg.use_clipped_value_loss:
                    value_clipped = old_value + (value - old_value).clamp(
                        -cfg.clip_param, cfg.clip_param
                    )
                    value_loss = torch.maximum(
                        (value - returns).square(),
                        (value_clipped - returns).square(),
                    ).mean()
                else:
                    value_loss = (value - returns).square().mean()
                actor_loss = surrogate_loss - cfg.entropy_coef * entropy.mean()
                custom_aux = actor_out.get("actor_aux_loss", None)
                if custom_aux is not None:
                    actor_loss = actor_loss + custom_aux.mean()
                aux_base_loss = actor_loss.new_zeros(())
                aux_contact_loss = actor_loss.new_zeros(())
                aux_ref_loss = actor_loss.new_zeros(())
                aux_robot_loss = actor_loss.new_zeros(())
                if cfg.aux_state_pred_enabled:
                    base_std = actor_out.get("aux_base_lin_vel_log_std").exp().clamp(
                        cfg.aux_min_std, cfg.aux_max_std
                    )
                    base_error = (
                        rollout.gt_base_lin_vel_b[ids]
                        - actor_out.get("aux_base_lin_vel_loc")
                    ) / base_std
                    aux_base_loss = 0.5 * (
                        base_error.square() + 2.0 * torch.log(base_std + 1.0e-8)
                    ).sum(-1).mean()
                    aux_contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        actor_out.get("aux_keybody_contact_logits"),
                        rollout.gt_keybody_contacts[ids],
                    )
                    aux_ref_loss = (
                        actor_out.get("aux_ref_keybody_rel_pos")
                        - rollout.gt_ref_keybody_rel_pos[ids]
                    ).square().mean()
                    aux_robot_loss = (
                        actor_out.get("aux_robot_keybody_rel_pos")
                        - rollout.gt_robot_keybody_rel_pos[ids]
                    ).square().mean()
                    actor_loss = (
                        actor_loss
                        + cfg.aux_w_base_lin_vel * aux_base_loss
                        + cfg.aux_w_keybody_contact * aux_contact_loss
                        + cfg.aux_w_ref_keybody_rel_pos * aux_ref_loss
                        + cfg.aux_w_robot_keybody_rel_pos * aux_robot_loss
                    )
                    actor_loss = actor_loss + 0.0 * (
                        actor_out.get("aux_root_height_loc").sum()
                        + actor_out.get("aux_root_height_log_std").sum()
                    )
                inactive_margin_loss = actor_loss.new_zeros(())
                if cfg.inactive_expert_margin_enabled:
                    layers = getattr(self.actor.actor_module, "layers", ())
                    losses = [
                        layer.last_inactive_expert_margin_to_topk_loss
                        for layer in layers
                        if getattr(
                            layer, "last_inactive_expert_margin_to_topk_loss", None
                        )
                        is not None
                    ]
                    if losses:
                        inactive_margin_loss = torch.stack(losses).mean()
                        actor_loss = actor_loss + (
                            cfg.inactive_expert_margin_weight * inactive_margin_loss
                        )
                if cfg.kl_coef > 0.0:
                    actor_loss = actor_loss + cfg.kl_coef * diagonal_gaussian_kl(
                        rollout.old_mu[ids].float(), rollout.old_sigma[ids].float(),
                        actor_out.get("mu").float(), actor_out.get("sigma").float()).mean()
                critic_loss = cfg.value_loss_coef * value_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), cfg.max_grad_norm
                )
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                clip_fraction = ((ratio - 1.0).abs() > cfg.clip_param).float().mean()
                values = (
                    actor_loss,
                    critic_loss,
                    surrogate_loss,
                    value_loss,
                    entropy.mean(),
                    ratio.mean(),
                    actor_norm,
                    critic_norm,
                    analytic_kl,
                    clip_fraction,
                    aux_base_loss,
                    aux_contact_loss,
                    aux_ref_loss,
                    aux_robot_loss,
                    inactive_margin_loss,
                )
                for key, value_item in zip(totals, values, strict=True):
                    if isinstance(value_item, torch.Tensor):
                        value_item = value_item.detach()
                    totals[key] += float(value_item)
                updates += 1
            if early_stopped:
                break

        self._apply_adaptive_lr(kl_windowed)
        denominator = max(1, updates)
        return HoloMotionPPOUpdateMetrics(
            actor_loss=totals["actor"] / denominator,
            critic_loss=totals["critic"] / denominator,
            surrogate_loss=totals["surrogate"] / denominator,
            value_loss=totals["value"] / denominator,
            entropy=totals["entropy"] / denominator,
            ratio_mean=totals["ratio"] / denominator,
            actor_grad_norm=totals["actor_norm"] / denominator,
            critic_grad_norm=totals["critic_norm"] / denominator,
            analytic_kl=totals["kl"] / denominator,
            clip_fraction=totals["clip"] / denominator,
            num_updates=updates,
            kl_early_stopped=early_stopped,
            actor_learning_rate=float(self.actor_optimizer.param_groups[0]["lr"]),
            critic_learning_rate=float(self.critic_optimizer.param_groups[0]["lr"]),
            aux_base_lin_vel_nll=totals["aux_base"] / denominator,
            aux_keybody_contact_bce=totals["aux_contact"] / denominator,
            aux_ref_keybody_rel_pos_mse=totals["aux_ref"] / denominator,
            aux_robot_keybody_rel_pos_mse=totals["aux_robot"] / denominator,
            inactive_expert_margin_loss=totals["inactive_margin"] / denominator,
        )
