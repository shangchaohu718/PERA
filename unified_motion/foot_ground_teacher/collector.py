"""Simulator-agnostic trajectory recorder used by MJLab or Isaac adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import write_trajectory
from .mjlab_collection import FootGroundFrame
from .schema import CONTROL_DT_S


def _cpu(value: torch.Tensor) -> np.ndarray:
    if value.shape[0] != 1:
        raise ValueError("one recorder represents one environment/trajectory")
    return value[0].detach().cpu().contiguous().numpy()


class FootGroundTrajectoryRecorder:
    """Align current inputs/actions with pre-reset next-state simulator labels."""

    def __init__(self, *, checkpoint_id: str, checkpoint_sha256: str, seed: int,
                 reference_id: str, trajectory_id: str, start_phase: float,
                 friction: float, mass_scale: float, disturbance: dict[str, Any]):
        self.metadata = dict(
            checkpoint_id=checkpoint_id, checkpoint_sha256=checkpoint_sha256,
            seed=int(seed), reference_id=reference_id, trajectory_id=trajectory_id,
            start_phase=float(start_phase), friction=float(friction),
            mass_scale=float(mass_scale), disturbance=disturbance,
            control_dt_s=CONTROL_DT_S,
        )
        self.rows: dict[str, list[np.ndarray]] = {}
        self._extra_names: set[str] | None = None
        self.closed = False

    def _add(self, name, value):
        self.rows.setdefault(name, []).append(_cpu(value))

    @torch.no_grad()
    def append(self, *, student_observation_t: torch.Tensor,
               privileged_current_t: torch.Tensor, executed_action_t: torch.Tensor,
               foot_current_t: torch.Tensor, episode_start_t: torch.Tensor,
               next_contact: FootGroundFrame, joint_velocity_t: torch.Tensor,
               joint_velocity_next: torch.Tensor, root_angular_velocity_t: torch.Tensor,
               root_angular_velocity_next: torch.Tensor,
               simple_physics_delta_joint_velocity: torch.Tensor,
               active_sliding_next: torch.Tensor, unexpected_slip_next: torch.Tensor,
               friction_t: torch.Tensor, valid_transition: torch.Tensor,
               joint_velocity_residual_valid: torch.Tensor | None = None,
               extra_tensors: dict[str, torch.Tensor] | None = None) -> None:
        if self.closed:
            raise RuntimeError("trajectory recorder is closed")
        if valid_transition.shape != (1,) or episode_start_t.shape != (1,):
            raise ValueError("recorder expects scalar batch masks")
        delta = joint_velocity_next - joint_velocity_t
        self._add("student_observation", student_observation_t)
        self._add("privileged_current", privileged_current_t)
        self._add("executed_action", executed_action_t)
        self._add("foot_current", foot_current_t)
        self._add("episode_start", episode_start_t.bool())
        self._add("valid_transition", valid_transition.bool())
        self._add("contact", next_contact.contact.bool())
        # Point/normal/relative velocity are valid only for an end-of-step contact.
        self._add("contact_valid", next_contact.contact.bool())
        self._add("contact_mode", next_contact.contact_mode.long())
        self._add("contact_point_local", next_contact.contact_point_local)
        self._add("contact_normal", next_contact.contact_normal)
        self._add("relative_velocity_normal", next_contact.relative_velocity_normal)
        self._add("relative_velocity_tangent", next_contact.relative_velocity_tangent)
        self._add("slip_speed", next_contact.slip_speed)
        self._add("normal_force", next_contact.normal_force)
        self._add("tangential_force", next_contact.tangential_force)
        self._add("contact_impulse", next_contact.contact_impulse)
        self._add("foot_load_ratio", next_contact.foot_load_ratio)
        self._add("delta_joint_velocity", delta)
        self._add("delta_root_angular_velocity",
                  root_angular_velocity_next - root_angular_velocity_t)
        self._add("joint_velocity_residual",
                  delta - simple_physics_delta_joint_velocity)
        if joint_velocity_residual_valid is None:
            joint_velocity_residual_valid = valid_transition
        self._add("joint_velocity_residual_valid", joint_velocity_residual_valid.bool())
        self._add("active_sliding", active_sliding_next.bool())
        self._add("unexpected_slip", unexpected_slip_next.bool())
        self._add("friction", friction_t)
        # Audit-only raw solver records are retained beside aggregate labels.
        self._add("raw_contact_found", next_contact.raw_found.bool())
        self._add("raw_contact_position_w", next_contact.raw_position_w)
        self._add("raw_contact_normal_w", next_contact.raw_normal_w)
        self._add("raw_contact_force_w", next_contact.raw_force_w)
        self._add("friction_utilization", next_contact.friction_utilization)
        for output_name, attribute in (
            ("raw_substep_contact_found", "raw_substep_found"),
            ("raw_substep_contact_position_w", "raw_substep_position_w"),
            ("raw_substep_contact_normal_w", "raw_substep_normal_w"),
            ("raw_substep_normal_force_n", "raw_substep_normal_force_n"),
            ("raw_substep_friction_found", "raw_substep_friction_found"),
            ("raw_substep_friction_position_w", "raw_substep_friction_position_w"),
            ("raw_substep_friction_force_w", "raw_substep_friction_force_w"),
        ):
            value = getattr(next_contact, attribute)
            if value is not None:
                self._add(output_name, value)
        extras = extra_tensors or {}
        if self._extra_names is None:
            reserved = set(self.rows)
            if any(name in reserved or not name for name in extras):
                raise ValueError("extra trajectory tensor collides with the core schema")
            self._extra_names = set(extras)
        if set(extras) != self._extra_names:
            raise ValueError("extra trajectory tensor keys changed within a trajectory")
        for name, value in extras.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"extra trajectory tensor {name!r} is not a tensor")
            self._add(name, value)

    def finish(self, path: str | Path, *, termination_reason: str) -> Path:
        if self.closed or not self.rows:
            raise RuntimeError("cannot finish an empty/closed trajectory")
        if not termination_reason:
            raise ValueError("termination_reason is required")
        self.closed = True
        arrays = {name: np.stack(values) for name, values in self.rows.items()}
        metadata = {**self.metadata, "termination_reason": termination_reason}
        return write_trajectory(path, arrays, metadata)
