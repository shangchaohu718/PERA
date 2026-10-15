"""Frozen Isaac Lab/RSL-RL K1 policy interoperability.

This deliberately does not use :func:`load_k1_checkpoint`: RSL policies consume
the already-concatenated 125-D Isaac Lab policy observation, not this project's
506-D HoloMotion observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


RSL_K1_BUNDLE_SCHEMA = "rsl_k1_frozen_policy_v1"
RSL_K1_OBSERVATION_SCHEMA = "isaaclab_k1_mj_dance_004_policy_125_v1"
RSL_K1_OBSERVATION_TERMS = (
    ("motion_command_joint_pos_vel", 44),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b_matrix_first_two_rows", 6),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos_rel", 22),
    ("joint_vel_rel", 22),
    ("last_action", 22),
)
RSL_K1_OBSERVATION_DIM = sum(width for _, width in RSL_K1_OBSERVATION_TERMS)
RSL_K1_ACTION_DIM = 22
RSL_EMPIRICAL_NORMALIZATION_EPS = 1e-2

_SOURCE_ACTOR_KEYS = tuple(
    f"actor.{layer}.{suffix}"
    for layer in (0, 2, 4, 6)
    for suffix in ("weight", "bias")
)
_SOURCE_TO_FROZEN = {key: key.removeprefix("actor.") for key in _SOURCE_ACTOR_KEYS}


@dataclass(frozen=True)
class RSLK1PolicyMetadata:
    checkpoint_id: str
    source_sha256: str
    source_iteration: int
    observation_schema: str = RSL_K1_OBSERVATION_SCHEMA
    deterministic_action: str = "actor_mean_before_action_manager"


class FrozenRSLK1Policy(nn.Module):
    """Exact deterministic Actor inference for the audited RSL-RL checkpoints."""

    def __init__(self, actor_state: Mapping[str, torch.Tensor], mean: torch.Tensor,
                 std: torch.Tensor, metadata: RSLK1PolicyMetadata):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(125, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 22),
        )
        self.actor.load_state_dict(dict(actor_state), strict=True)
        self.register_buffer("observation_mean", _normalizer_vector(mean, "mean"))
        self.register_buffer("observation_std", _normalizer_vector(std, "std"))
        self.metadata = metadata
        _validate_finite_tensors(self.state_dict(), "RSL K1 policy")
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def act(self, policy_observation: torch.Tensor) -> torch.Tensor:
        """Return RSL ``act_inference`` output before action-manager processing."""
        if not isinstance(policy_observation, torch.Tensor):
            raise TypeError("policy_observation must be a torch.Tensor")
        if policy_observation.ndim < 1 or policy_observation.shape[-1] != RSL_K1_OBSERVATION_DIM:
            raise ValueError("RSL K1 policy requires [..., 125] Isaac Lab policy observations")
        if not policy_observation.is_floating_point() or not torch.isfinite(policy_observation).all():
            raise ValueError("policy_observation must be finite floating point values")
        if policy_observation.device != self.observation_mean.device:
            raise ValueError("policy_observation and frozen policy must be on the same device")
        normalized = (policy_observation - self.observation_mean) / (
            self.observation_std + RSL_EMPIRICAL_NORMALIZATION_EPS
        )
        action = self.actor(normalized)
        if action.shape != policy_observation.shape[:-1] + (RSL_K1_ACTION_DIM,):
            raise RuntimeError("invalid RSL K1 action shape")
        if not torch.isfinite(action).all():
            raise RuntimeError("non-finite RSL K1 action")
        return action


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalizer_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape not in ((125,), (1, 125)):
        raise ValueError(f"RSL observation normalizer {name} must have shape [1,125]")
    value = value.detach().to(device="cpu").reshape(125).clone()
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"RSL observation normalizer {name} must be finite floating point")
    if name == "std" and (value < 0).any():
        raise ValueError("RSL observation normalizer std must be non-negative")
    return value


def _validate_finite_tensors(values: Mapping[str, object], label: str) -> None:
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite tensor {key!r}")


def _policy_from_source_payload(payload: object, *, checkpoint_id: str,
                                source_sha256: str) -> FrozenRSLK1Policy:
    if not isinstance(payload, dict):
        raise ValueError("RSL-RL checkpoint must be a dictionary")
    required = {"model_state_dict", "obs_norm_state_dict", "iter"}
    if missing := required - payload.keys():
        raise ValueError(f"RSL-RL checkpoint is missing {sorted(missing)}")
    iteration = payload["iter"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("RSL-RL checkpoint iteration must be a non-negative integer")
    model_state = payload["model_state_dict"]
    normalizer = payload["obs_norm_state_dict"]
    if not isinstance(model_state, dict) or not isinstance(normalizer, dict):
        raise ValueError("invalid RSL-RL model/normalizer state")
    actor_keys = {key for key in model_state if isinstance(key, str) and key.startswith("actor.")}
    if actor_keys != set(_SOURCE_ACTOR_KEYS):
        raise ValueError("checkpoint does not match the audited RSL K1 125-D Actor architecture")
    actor_state = {_SOURCE_TO_FROZEN[key]: model_state[key].detach().cpu().clone()
                   for key in _SOURCE_ACTOR_KEYS}
    expected_shapes = {
        "0.weight": (512, 125), "0.bias": (512,),
        "2.weight": (256, 512), "2.bias": (256,),
        "4.weight": (128, 256), "4.bias": (128,),
        "6.weight": (22, 128), "6.bias": (22,),
    }
    if {key: tuple(value.shape) for key, value in actor_state.items()} != expected_shapes:
        raise ValueError("RSL K1 Actor tensor shapes do not match the audited architecture")
    if not {"_mean", "_std"} <= normalizer.keys():
        raise ValueError("RSL-RL observation normalizer is missing mean/std")
    return FrozenRSLK1Policy(actor_state, normalizer["_mean"], normalizer["_std"],
        RSLK1PolicyMetadata(checkpoint_id, source_sha256, iteration))


def load_rsl_k1_checkpoint(checkpoint: str | Path, *, device: str | torch.device = "cpu",
                           trusted_source: bool = False,
                           checkpoint_id: str | None = None) -> FrozenRSLK1Policy:
    """Load native RSL; unsafe pickle fallback requires explicit trust."""
    path = Path(checkpoint)
    checkpoint_id = path.stem if checkpoint_id is None else checkpoint_id.strip()
    if not checkpoint_id:
        raise ValueError("checkpoint_id must be non-empty")
    source_hash = sha256_file(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if not trusted_source:
            raise ValueError(
                "safe loading rejected this RSL checkpoint (usually because infos contains NumPy RNG state); "
                "use the converter with --trusted-source only for a checkpoint you trust"
            ) from safe_error
        payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = _policy_from_source_payload(payload, checkpoint_id=checkpoint_id,
                                         source_sha256=source_hash)
    return policy.to(device).eval().requires_grad_(False)


def frozen_rsl_k1_bundle(policy: FrozenRSLK1Policy) -> dict:
    metadata = policy.metadata
    return {
        "policy_schema": RSL_K1_BUNDLE_SCHEMA,
        "observation_schema": metadata.observation_schema,
        "observation_terms": list(RSL_K1_OBSERVATION_TERMS),
        "normalization_eps": RSL_EMPIRICAL_NORMALIZATION_EPS,
        "actor_state_dict": {key: value.detach().cpu().clone()
                             for key, value in policy.actor.state_dict().items()},
        "observation_mean": policy.observation_mean.detach().cpu().clone(),
        "observation_std": policy.observation_std.detach().cpu().clone(),
        "checkpoint_id": metadata.checkpoint_id,
        "source_sha256": metadata.source_sha256,
        "source_iteration": metadata.source_iteration,
        "deterministic_action": metadata.deterministic_action,
        "frozen": True,
    }


def save_frozen_rsl_k1_bundle(policy: FrozenRSLK1Policy, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frozen_rsl_k1_bundle(policy), output)


def load_frozen_rsl_k1_bundle(bundle: str | Path, *,
                              device: str | torch.device = "cpu") -> FrozenRSLK1Policy:
    payload = torch.load(Path(bundle), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("policy_schema") != RSL_K1_BUNDLE_SCHEMA:
        raise ValueError("not a frozen RSL K1 policy bundle")
    if (payload.get("observation_schema") != RSL_K1_OBSERVATION_SCHEMA or
            payload.get("observation_terms") != list(RSL_K1_OBSERVATION_TERMS) or
            payload.get("normalization_eps") != RSL_EMPIRICAL_NORMALIZATION_EPS or
            payload.get("frozen") is not True):
        raise ValueError("unsupported RSL K1 observation/normalization contract")
    metadata = RSLK1PolicyMetadata(
        checkpoint_id=payload["checkpoint_id"], source_sha256=payload["source_sha256"],
        source_iteration=payload["source_iteration"],
        observation_schema=payload["observation_schema"],
        deterministic_action=payload["deterministic_action"],
    )
    policy = FrozenRSLK1Policy(payload["actor_state_dict"], payload["observation_mean"],
                               payload["observation_std"], metadata)
    return policy.to(device).eval().requires_grad_(False)


def verify_frozen_rsl_k1_provenance(policy: FrozenRSLK1Policy, *,
                                    checkpoint: str | Path,
                                    checkpoint_id: str) -> str:
    """Bind a safe bundle to the exact native checkpoint named by a campaign."""
    if policy.metadata.checkpoint_id != checkpoint_id:
        raise ValueError(
            "policy bundle checkpoint_id does not match the requested checkpoint: "
            f"{policy.metadata.checkpoint_id!r} != {checkpoint_id!r}"
        )
    source_hash = sha256_file(checkpoint)
    if policy.metadata.source_sha256 != source_hash:
        raise ValueError(
            "policy bundle source_sha256 does not match the provenance checkpoint"
        )
    return source_hash
