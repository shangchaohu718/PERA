"""Versioned tensor and metadata contract for Teacher V1 datasets."""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA_VERSION = 1
CONTROL_DT_S = 0.02
CONTACT_MODES = ("separation", "stable", "sliding", "impact")
FOOT_NAMES = ("left", "right")


@dataclass(frozen=True)
class RegressionStatistics:
    """Train-split-only statistics.  Validation/test must reuse this object."""

    mean: Mapping[str, tuple[float, ...]]
    std: Mapping[str, tuple[float, ...]]
    count: Mapping[str, int]
    std_floor: float = 1.0e-5

    def __post_init__(self) -> None:
        if not math.isfinite(self.std_floor) or self.std_floor <= 0:
            raise ValueError("std_floor must be positive and finite")
        if set(self.mean) != set(self.std) or set(self.mean) != set(self.count):
            raise ValueError("statistics keys differ")
        for name in self.mean:
            if len(self.mean[name]) != len(self.std[name]) or not self.mean[name]:
                raise ValueError(f"invalid statistics width for {name}")
            if self.count[name] < 1 or any(
                not math.isfinite(v) for v in (*self.mean[name], *self.std[name])
            ) or any(v < self.std_floor for v in self.std[name]):
                raise ValueError(f"invalid statistics for {name}")

    def tensors(self, name: str, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (like.new_tensor(self.mean[name]), like.new_tensor(self.std[name]))

    def standardize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors(name, value)
        return (value - mean) / std

    def decode(self, name: str, value: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors(name, value)
        return value * std + mean

    def to_json(self, path: str | Path) -> None:
        payload = dict(schema=SCHEMA_VERSION, mean=self.mean, std=self.std,
                       count=self.count, std_floor=self.std_floor)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class FootGroundBatch:
    """A sequence batch whose targets are aligned to ``t + 1``.

    Leading dimensions are ``[batch, time]``.  ``valid_transition`` is false
    for terminal observations and padding, so reset states can never become a
    dynamics target.  Foot quantities have an additional fixed ``[left,right]``
    axis.  All forces are N, impulses N*s, distances m and velocities m/s.
    """

    student_observation: torch.Tensor
    privileged_current: torch.Tensor
    executed_action: torch.Tensor
    foot_current: torch.Tensor
    episode_start: torch.Tensor
    valid_transition: torch.Tensor
    contact_valid: torch.Tensor
    contact: torch.Tensor
    contact_mode: torch.Tensor
    contact_point_local: torch.Tensor
    contact_normal: torch.Tensor
    relative_velocity_normal: torch.Tensor
    relative_velocity_tangent: torch.Tensor
    slip_speed: torch.Tensor
    normal_force: torch.Tensor
    tangential_force: torch.Tensor
    contact_impulse: torch.Tensor
    foot_load_ratio: torch.Tensor
    delta_joint_velocity: torch.Tensor
    delta_root_angular_velocity: torch.Tensor
    joint_velocity_residual: torch.Tensor
    active_sliding: torch.Tensor
    unexpected_slip: torch.Tensor
    friction: torch.Tensor
    joint_velocity_residual_valid: torch.Tensor | None = None
    reference_intent: torch.Tensor | None = None

    def validate(self, *, dof: int = 22) -> "FootGroundBatch":
        lead = self.student_observation.shape[:2]
        if self.student_observation.ndim != 3 or min(lead) < 1:
            raise ValueError("student_observation must be [B,T,D]")
        exact = {
            "privileged_current": lead + (self.privileged_current.shape[-1],),
            "executed_action": lead + (dof,),
            "foot_current": lead + (2, self.foot_current.shape[-1]),
            "episode_start": lead,
            "valid_transition": lead,
            "contact_valid": lead + (2,),
            "contact": lead + (2,),
            "contact_mode": lead + (2,),
            "contact_point_local": lead + (2, 3),
            "contact_normal": lead + (2, 3),
            "relative_velocity_normal": lead + (2,),
            "relative_velocity_tangent": lead + (2, 2),
            "slip_speed": lead + (2,),
            "normal_force": lead + (2,),
            "tangential_force": lead + (2, 2),
            "contact_impulse": lead + (2,),
            "foot_load_ratio": lead + (2,),
            "delta_joint_velocity": lead + (dof,),
            "delta_root_angular_velocity": lead + (3,),
            "joint_velocity_residual": lead + (dof,),
            "active_sliding": lead + (2,),
            "unexpected_slip": lead + (2,),
            "friction": lead,
        }
        if self.reference_intent is not None:
            if self.reference_intent.ndim != 3 or self.reference_intent.shape[:2] != lead:
                raise ValueError("reference_intent must be [B,T,D]")
            if self.reference_intent.shape[-1] < 1:
                raise ValueError("reference_intent must have at least one feature")
        for name, shape in exact.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has {tuple(value.shape)}, expected {shape}")
        for name in ("episode_start", "valid_transition", "contact_valid", "contact",
                     "active_sliding", "unexpected_slip"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must be boolean")
        if self.joint_velocity_residual_valid is None:
            self.joint_velocity_residual_valid = self.valid_transition.clone()
        if (self.joint_velocity_residual_valid.shape != lead or
                self.joint_velocity_residual_valid.dtype != torch.bool):
            raise TypeError("joint_velocity_residual_valid must be boolean [B,T]")
        if (self.joint_velocity_residual_valid & ~self.valid_transition).any():
            raise ValueError("residual validity cannot exceed transition validity")
        if self.contact_mode.dtype not in (torch.int32, torch.int64):
            raise TypeError("contact_mode must be integer")
        if ((self.contact_mode < 0) | (self.contact_mode >= len(CONTACT_MODES))).any():
            raise ValueError("contact_mode is out of range")
        if (self.contact_valid & ~self.contact).any():
            raise ValueError("contact_valid means a physical contact exists")
        if (self.active_sliding & self.unexpected_slip).any():
            raise ValueError("active and unexpected sliding must be disjoint")
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and value.dtype.is_floating_point and not torch.isfinite(value).all():
                raise ValueError(f"{item.name} contains NaN/Inf")
        normal_length = torch.linalg.vector_norm(self.contact_normal, dim=-1)
        mask = self.contact_valid & self.valid_transition.unsqueeze(-1)
        if mask.any() and not torch.allclose(
            normal_length[mask], torch.ones_like(normal_length[mask]), atol=1e-4, rtol=1e-4
        ):
            raise ValueError("valid contact normals must be unit vectors")
        ratio_sum = self.foot_load_ratio.sum(-1)
        loaded = self.normal_force.sum(-1) > 1.0e-6
        if (loaded & self.valid_transition).any() and not torch.allclose(
            ratio_sum[loaded & self.valid_transition],
            torch.ones_like(ratio_sum[loaded & self.valid_transition]), atol=1e-4,
        ):
            raise ValueError("foot load ratios must sum to one under load")
        return self


def split_key(metadata: Mapping[str, Any]) -> str:
    """Group complete trajectories; adjacent windows can never cross splits."""
    required = ("checkpoint_id", "trajectory_id", "reference_id")
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise ValueError(f"missing split identity: {missing}")
    return "|".join(str(metadata[name]) for name in required)


def deterministic_split(metadata: Mapping[str, Any], *, heldout_checkpoints=(),
                        validation_fraction: float = 0.15,
                        test_fraction: float = 0.15) -> str:
    """Return train/validation/test without window-level leakage."""
    if metadata["checkpoint_id"] in set(heldout_checkpoints):
        return "test"
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 \
            or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid split fractions")
    value = int.from_bytes(hashlib.sha256(split_key(metadata).encode()).digest()[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"
