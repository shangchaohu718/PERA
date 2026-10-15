"""Isaac Lab raw-contact collection for Foot-Ground Teacher V1.

The collector samples PhysX contact buffers on every physics substep.  Labels
at ``t+1`` use the final substep's real contact points, while normal impulse is
the integral over the complete control interval.  No foot-link-center proxy is
used for contact-point kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .mjlab_collection import FootGroundFrame
from .physics import aggregate_contacts


ISAAC_RAW_SENSOR_NAMES = (
    "left_foot_ground_teacher_v1_raw",
    "right_foot_ground_teacher_v1_raw",
)


def configure_isaac_teacher_contact_sensors(scene_cfg, *,
        ground_prim_path: str = "/World/ground/terrain/GroundPlane/CollisionPlane",
        max_contact_points: int = 8) -> None:
    """Add two ground-filtered sensors with point and friction tracking."""
    if max_contact_points < 1:
        raise ValueError("max_contact_points must be positive")
    from isaaclab.sensors import ContactSensorCfg

    for side, name in zip(("left", "right"), ISAAC_RAW_SENSOR_NAMES):
        if hasattr(scene_cfg, name):
            raise ValueError(f"Isaac Teacher contact sensor {name!r} already exists")
        sensor_kwargs = dict(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link",
            filter_prim_paths_expr=[ground_prim_path],
            history_length=0,
            track_contact_points=True,
            max_contact_data_count_per_prim=max_contact_points,
            track_air_time=False,
            force_threshold=0.0,
            debug_vis=False,
        )
        try:
            sensor = ContactSensorCfg(track_friction_forces=True, **sensor_kwargs)
        except TypeError as error:
            # Isaac Lab 2.2 / the pinned 18c7c58 runtime does not expose this
            # config field, although its underlying RigidContactView still has
            # get_friction_data().  The accumulator calls that view directly.
            if "track_friction_forces" not in str(error):
                raise
            sensor = ContactSensorCfg(**sensor_kwargs)
        setattr(scene_cfg, name, sensor)


@dataclass(frozen=True)
class _RawSubstep:
    found: torch.Tensor                 # [B,2,S]
    point_w: torch.Tensor               # [B,2,S,3]
    normal_w: torch.Tensor              # [B,2,S,3]
    normal_force_n: torch.Tensor        # [B,2,S]
    friction_found: torch.Tensor        # [B,2,S]
    friction_point_w: torch.Tensor      # [B,2,S,3]
    friction_force_w: torch.Tensor      # [B,2,S,3]
    dt_s: float


def _padded_pair_buffer(view, values: torch.Tensor, counts: torch.Tensor,
                        starts: torch.Tensor, *, batch: int, slots: int,
                        width: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack one-body/one-filter PhysX buffers without discarding contacts."""
    counts = counts.reshape(batch, -1)
    starts = starts.reshape(batch, -1)
    if counts.shape[1] != 1 or starts.shape[1] != 1:
        raise RuntimeError("Teacher contact sensor must have exactly one terrain filter")
    tail = () if width is None else (width,)
    output = values.new_zeros((batch, slots) + tail)
    found = torch.zeros((batch, slots), dtype=torch.bool, device=values.device)
    for env in range(batch):
        count = int(counts[env, 0].item())
        start = int(starts[env, 0].item())
        if count > slots:
            raise RuntimeError(
                f"PhysX returned {count} contacts but raw schema has only {slots} slots"
            )
        if count:
            chunk = values[start:start + count]
            if width is None:
                chunk = chunk.reshape(count)
            else:
                chunk = chunk.reshape(count, width)
            output[env, :count] = chunk
            found[env, :count] = True
    return found, output


def _orient_normals_up(normal: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    unit = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-8)
    up = up_w[:, None, :]
    return torch.where(((unit * up).sum(-1, keepdim=True) < 0), -unit, unit)


def _quat_inverse_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    w = quaternion[..., :1]
    xyz = -quaternion[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (w * uv + torch.cross(xyz, uv, dim=-1))


class IsaacFootContactAccumulator:
    """Accumulate exact raw PhysX buffers over one 20 ms control step.

    Call :meth:`begin_control_step`, execute exactly one environment step, then
    call :meth:`finish_control_step`.  ``subscribe`` can attach the sampler to
    Isaac's physics-step events; tests and custom loops may call
    :meth:`sample_substep` directly.
    """

    def __init__(self, sensors: Sequence[object], *, batch: int, slots: int = 8,
                 slip_threshold_m_s: float = .03,
                 impact_impulse_threshold_n_s: float = 2.0,
                 contact_force_threshold_n: float = 1.0):
        if len(sensors) != 2 or batch < 1 or slots < 1:
            raise ValueError("expected two foot sensors and positive batch/slots")
        self.sensors = tuple(sensors)
        self.batch = int(batch)
        self.slots = int(slots)
        self.slip_threshold = float(slip_threshold_m_s)
        self.impact_threshold = float(impact_impulse_threshold_n_s)
        if contact_force_threshold_n < 0:
            raise ValueError("contact force threshold must be non-negative")
        self.contact_force_threshold = float(contact_force_threshold_n)
        self._armed = False
        self._steps: list[_RawSubstep] = []
        self._subscription = None
        self._previous_active: torch.Tensor | None = None

    def subscribe(self) -> None:
        if self._subscription is not None:
            raise RuntimeError("Isaac contact accumulator is already subscribed")
        from omni.physx import get_physx_interface
        self._subscription = get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )

    def close(self) -> None:
        self._subscription = None
        self._armed = False

    def begin_control_step(self) -> None:
        if self._armed:
            raise RuntimeError("previous control step has not been finished")
        self._steps = []
        self._armed = True

    def _on_physics_step(self, dt_s: float) -> None:
        if self._armed:
            self.sample_substep(float(dt_s))

    @torch.no_grad()
    def sample_substep(self, dt_s: float) -> None:
        if not self._armed:
            raise RuntimeError("begin_control_step must precede substep sampling")
        if dt_s <= 0:
            raise ValueError("physics dt must be positive")
        normal_items = []
        friction_items = []
        for sensor in self.sensors:
            view = sensor.contact_physx_view
            forces, points, normals, _, counts, starts = view.get_contact_data(dt=dt_s)
            found, force = _padded_pair_buffer(
                view, forces, counts, starts, batch=self.batch, slots=self.slots)
            _, point = _padded_pair_buffer(
                view, points, counts, starts, batch=self.batch, slots=self.slots, width=3)
            _, normal = _padded_pair_buffer(
                view, normals, counts, starts, batch=self.batch, slots=self.slots, width=3)
            friction, friction_points, friction_counts, friction_starts = (
                view.get_friction_data(dt=dt_s)
            )
            friction_found, friction_force = _padded_pair_buffer(
                view, friction, friction_counts, friction_starts,
                batch=self.batch, slots=self.slots, width=3)
            _, friction_point = _padded_pair_buffer(
                view, friction_points, friction_counts, friction_starts,
                batch=self.batch, slots=self.slots, width=3)
            normal_items.append((found, point, normal, force.abs()))
            friction_items.append((friction_found, friction_point, friction_force))
        self._steps.append(_RawSubstep(
            found=torch.stack([x[0] for x in normal_items], 1).clone(),
            point_w=torch.stack([x[1] for x in normal_items], 1).clone(),
            normal_w=torch.stack([x[2] for x in normal_items], 1).clone(),
            normal_force_n=torch.stack([x[3] for x in normal_items], 1).clone(),
            friction_found=torch.stack([x[0] for x in friction_items], 1).clone(),
            friction_point_w=torch.stack([x[1] for x in friction_items], 1).clone(),
            friction_force_w=torch.stack([x[2] for x in friction_items], 1).clone(),
            dt_s=float(dt_s),
        ))

    @torch.no_grad()
    def finish_control_step(self, *, body_position_w: torch.Tensor,
            body_quaternion_wxyz: torch.Tensor, body_linear_velocity_w: torch.Tensor,
            body_angular_velocity_w: torch.Tensor, heading_w: torch.Tensor,
            friction: torch.Tensor, active_sliding: torch.Tensor,
            ground_velocity_w: torch.Tensor | None = None,
            up_w: torch.Tensor | None = None) -> FootGroundFrame:
        if not self._armed or not self._steps:
            raise RuntimeError("no sampled Isaac physics substeps")
        self._armed = False
        for name, value in (
            ("body_position_w", body_position_w),
            ("body_quaternion_wxyz", body_quaternion_wxyz),
            ("body_linear_velocity_w", body_linear_velocity_w),
            ("body_angular_velocity_w", body_angular_velocity_w),
            ("heading_w", heading_w),
        ):
            width = 4 if name == "body_quaternion_wxyz" else 3
            if value.shape != (self.batch, 2, width):
                raise ValueError(f"{name} must have shape [B,2,{width}]")
        if friction.shape != (self.batch,) or active_sliding.shape != (self.batch, 2):
            raise ValueError("friction/active_sliding shape mismatch")
        last = self._steps[-1]
        # PhysX may retain a geometric contact patch with exactly zero solver
        # impulse. Preserve it in raw buffers, but do not label it as loaded
        # foot-ground contact or stable support.
        active_found = last.found & (
            last.normal_force_n > self.contact_force_threshold)
        if up_w is None:
            up_w = body_position_w.new_tensor((0.0, 0.0, 1.0)).expand(self.batch, 3)
        if up_w.shape != (self.batch, 3):
            raise ValueError("up_w must have shape [B,3]")
        normal = _orient_normals_up(last.normal_w.flatten(0, 1),
                                    up_w[:, None].expand(-1, 2, -1).flatten(0, 1))
        normal = normal.reshape(self.batch, 2, self.slots, 3)
        normal_vector_force = normal * last.normal_force_n.unsqueeze(-1)
        friction_net = (last.friction_force_w * last.friction_found.unsqueeze(-1)).sum(-2)
        compression = last.normal_force_n * last.found
        weights = compression / compression.sum(-1, keepdim=True).clamp_min(1e-8)
        reconstructed_force = normal_vector_force + weights.unsqueeze(-1) * friction_net.unsqueeze(-2)
        if ground_velocity_w is None:
            ground_velocity_w = torch.zeros_like(last.point_w)
        if ground_velocity_w.shape != last.point_w.shape:
            raise ValueError("ground_velocity_w must have shape [B,2,S,3]")

        per_foot = []
        for foot in range(2):
            per_foot.append(aggregate_contacts(
                found=active_found[:, foot], points=last.point_w[:, foot],
                normals=normal[:, foot], forces=reconstructed_force[:, foot],
                body_origin=body_position_w[:, foot],
                body_linear_velocity=body_linear_velocity_w[:, foot],
                body_angular_velocity=body_angular_velocity_w[:, foot],
                ground_velocity=ground_velocity_w[:, foot], heading=heading_w[:, foot],
                friction=friction, physics_dt_s=last.dt_s,
            ))
        contact = torch.stack([x.found for x in per_foot], 1)
        point = torch.stack([x.point for x in per_foot], 1)
        aggregate_normal = torch.stack([x.normal for x in per_foot], 1)
        velocity_n = torch.stack([x.relative_velocity_normal for x in per_foot], 1)
        velocity_t = torch.stack([x.relative_velocity_tangent for x in per_foot], 1)
        normal_force = torch.stack([x.normal_force for x in per_foot], 1)
        tangential_force = torch.stack([x.tangential_force for x in per_foot], 1)
        slip = torch.stack([x.slip_speed for x in per_foot], 1)
        utilization = torch.stack([x.friction_utilization for x in per_foot], 1)
        impulse = normal_force.new_zeros((self.batch, 2))
        for step in self._steps:
            step_active = step.found & (
                step.normal_force_n > self.contact_force_threshold)
            impulse += (step.normal_force_n * step_active).sum(-1) * step.dt_s
        active_history = torch.stack([
            (step.found & (step.normal_force_n > self.contact_force_threshold)).any(-1)
            for step in self._steps
        ])
        previous = (active_history[0] if self._previous_active is None
                    else self._previous_active)
        onset = torch.zeros_like(previous)
        for current in active_history:
            onset |= current & ~previous
            previous = current
        self._previous_active = active_history[-1].clone()
        mode = torch.where(contact, torch.ones_like(contact, dtype=torch.long),
                           torch.zeros_like(contact, dtype=torch.long))
        mode = torch.where(contact & (slip >= self.slip_threshold),
                           torch.full_like(mode, 2), mode)
        mode = torch.where(onset & (impulse >= self.impact_threshold),
                           torch.full_like(mode, 3), mode)
        load = normal_force.sum(-1, keepdim=True)
        ratio = torch.where(load > 1e-8, normal_force / load.clamp_min(1e-8),
                            torch.zeros_like(normal_force))
        point_local = _quat_inverse_apply_wxyz(
            body_quaternion_wxyz, point - body_position_w)
        normal_local = _quat_inverse_apply_wxyz(body_quaternion_wxyz, aggregate_normal)
        substep_found = torch.stack([step.found for step in self._steps], 2)
        substep_point = torch.stack([step.point_w for step in self._steps], 2)
        substep_normal = torch.stack([step.normal_w for step in self._steps], 2)
        substep_normal_force = torch.stack([step.normal_force_n for step in self._steps], 2)
        substep_friction_found = torch.stack([step.friction_found for step in self._steps], 2)
        substep_friction_point = torch.stack([step.friction_point_w for step in self._steps], 2)
        substep_friction_force = torch.stack([step.friction_force_w for step in self._steps], 2)
        return FootGroundFrame(
            contact=contact, contact_mode=mode, contact_point_local=point_local,
            contact_normal=normal_local, relative_velocity_normal=velocity_n,
            relative_velocity_tangent=velocity_t, slip_speed=slip,
            normal_force=normal_force, tangential_force=tangential_force,
            contact_impulse=impulse, foot_load_ratio=ratio,
            friction_utilization=utilization, raw_found=last.found,
            raw_position_w=last.point_w, raw_normal_w=normal,
            raw_force_w=reconstructed_force,
            raw_substep_found=substep_found,
            raw_substep_position_w=substep_point,
            raw_substep_normal_w=substep_normal,
            raw_substep_normal_force_n=substep_normal_force,
            raw_substep_friction_found=substep_friction_found,
            raw_substep_friction_position_w=substep_friction_point,
            raw_substep_friction_force_w=substep_friction_force,
        )
