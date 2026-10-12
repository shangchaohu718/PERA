"""Full checkpoint and frozen evaluation for every K1 Actor, with/without Physics."""
from dataclasses import dataclass
from pathlib import Path

import torch

from .k1_policy_backends import build_k1_networks
from .holomotion_original_algorithm import flat_observations_to_tensordict, OriginalHoloMotionK1Algorithm
from .holomotion_physics_runtime import (
    FrozenPhysicsPolicy, load_physics_checkpoint, validate_task_contract, evaluate_physics_policy,
)


@dataclass
class K1Checkpoint:
    actor: torch.nn.Module
    critic: torch.nn.Module
    payload: dict


def runtime_configuration(payload):
    return payload["physics_runtime_config" if "physics_schema" in payload else "policy_runtime_config"]


def load_k1_checkpoint(checkpoint, *, device="cpu", holomotion_root=None):
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("K1 checkpoint must be a dictionary")
    if "physics_schema" in payload:
        return load_physics_checkpoint(checkpoint, device=device, holomotion_root=holomotion_root)
    required = {"network_config", "base_actor_state_dict", "critic_state_dict", "ppo_config",
                "policy_runtime_config", "iter"}
    if payload.get("policy_schema") != 1 or required-payload.keys():
        raise ValueError("standalone K1 policy bundle missing; legacy sidecars require the original loader")
    runtime = payload["policy_runtime_config"]
    if (type(payload["iter"]) is not int or payload["iter"] < 0 or
            type(runtime.get("rollout_steps")) is not int or runtime["rollout_steps"] < 1):
        raise ValueError("invalid K1 checkpoint iteration/rollout length")
    actor, critic = build_k1_networks(payload["network_config"], device=device, holomotion_root=holomotion_root)
    for module, key in ((actor, "base_actor_state_dict"), (critic, "critic_state_dict")):
        module.load_state_dict(payload[key], strict=True)
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()
               if isinstance(value, torch.Tensor)):
            raise ValueError("K1 checkpoint contains non-finite weights")
        module.eval()
    return K1Checkpoint(actor, critic, payload)


class FrozenK1BasePolicy:
    needs_physics = False
    policy_name = "base_actor"
    ablation = "full"

    def __init__(self, actor, *, rollout_steps, task_contract=None):
        self.actor = actor.eval().requires_grad_(False)
        self.rollout_steps = rollout_steps
        self.task_contract = task_contract
        self.reset()

    @property
    def frozen_modules(self):
        return (self.actor,)

    def reset(self):
        self.steps, self.batch, self.pending = 0, None, False

    @torch.no_grad()
    def act(self, actor_observation, physics_observation, episode_start):
        if self.pending:
            raise RuntimeError("record the executed action before the next observation")
        batch = actor_observation.shape[0]
        if episode_start.shape != (batch,) or episode_start.dtype != torch.bool:
            raise ValueError("episode_start must be a boolean [B] tensor")
        if self.batch is None:
            if not episode_start.all():
                raise ValueError("start frozen evaluation at real episode boundaries")
            self.batch = batch
        if self.batch != batch:
            raise ValueError("batch changed; reset the runtime")
        self.actor.eval()
        if self.steps % self.rollout_steps == 0:
            self.actor.reset_kv_cache(batch, actor_observation.device)
        elif episode_start.any():
            self.actor.clear_env_cache(episode_start.nonzero(as_tuple=False).flatten())
        data = flat_observations_to_tensordict(actor_observation, actor_observation.new_zeros(batch,843))
        action = self.actor(data, mode="inference", update_obs_norm=False).get("actions")
        if action.shape != (batch,22) or not torch.isfinite(action).all():
            raise RuntimeError("invalid frozen K1 actions")
        self.pending = True
        return action

    def record_action(self, executed_action):
        if not self.pending:
            raise RuntimeError("act must precede record_action")
        self.pending = False
        self.steps += 1


def frozen_k1_policy(loaded, *, ablation="full"):
    runtime = runtime_configuration(loaded.payload)
    if "physics_schema" in loaded.payload:
        return FrozenPhysicsPolicy(loaded.actor, loaded.actor_dynamics,
            confidence_warmup_steps=runtime["confidence_warmup_steps"],
            rollout_steps=runtime["rollout_steps"], task_contract=runtime.get("task_contract"),
            calibration=loaded.calibration, ablation=ablation)
    if ablation != "full":
        raise ValueError("Physics ablations require a Physics checkpoint")
    return FrozenK1BasePolicy(loaded.actor, rollout_steps=runtime["rollout_steps"],
                              task_contract=runtime.get("task_contract"))


def resume_k1_trainer(checkpoint, task, *, holomotion_root=None, **physics_history_limits):
    from .holomotion_mjlab_ppo import HoloMotionMJLabPPO, HoloMotionPPOConfig
    from .holomotion_physics_training import HoloMotionPhysicsMJLabPPO
    loaded = load_k1_checkpoint(checkpoint, device=task.device, holomotion_root=holomotion_root)
    if "physics_schema" in loaded.payload:
        return HoloMotionPhysicsMJLabPPO.from_checkpoint(checkpoint, task,
            holomotion_root=holomotion_root, **physics_history_limits)
    validate_task_contract(runtime_configuration(loaded.payload), task)
    algorithm = OriginalHoloMotionK1Algorithm(loaded.actor, loaded.critic, None,
        Path(checkpoint).parent, loaded.payload["network_config"])
    trainer = HoloMotionMJLabPPO(algorithm, task, config=HoloMotionPPOConfig(**loaded.payload["ppo_config"]))
    trainer.load_optimizer_checkpoint(checkpoint)
    return trainer


def evaluate_k1_policy(policy, task, *, steps):
    return evaluate_physics_policy(policy, task, steps=steps)
