"""MJLab-only raw contact sensor and V1 label extraction.

Imports of MJLab are local so offline training does not require a simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .physics import aggregate_contacts


RAW_SENSOR_NAME = "k1_feet_terrain_contact_teacher_v1_raw"


def add_raw_contact_sensor(cfg, *, num_slots: int = 8, physics_substeps: int = 10):
    """Add a non-reduced audit sensor without changing the Actor's old sensor."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    if num_slots < 2 or physics_substeps < 1:
        raise ValueError("raw collection requires >=2 slots and positive substeps")
    if any(sensor.name == RAW_SENSOR_NAME for sensor in cfg.scene.sensors):
        raise ValueError("Teacher raw sensor already exists")
    sensor = ContactSensorCfg(
        name=RAW_SENSOR_NAME,
        primary=ContactMatch(mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        # Keep the strongest slots individually. The complete fixed-size raw
        # record is saved; aggregation happens later and is fully auditable.
        reduce="maxforce", num_slots=num_slots, global_frame=True,
        history_length=physics_substeps,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (sensor,)
    return cfg


@dataclass(frozen=True)
class FootGroundFrame:
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
    friction_utilization: torch.Tensor
    raw_found: torch.Tensor
    raw_position_w: torch.Tensor
    raw_normal_w: torch.Tensor
    raw_force_w: torch.Tensor
    raw_substep_found: torch.Tensor | None = None
    raw_substep_position_w: torch.Tensor | None = None
    raw_substep_normal_w: torch.Tensor | None = None
    raw_substep_normal_force_n: torch.Tensor | None = None
    raw_substep_friction_found: torch.Tensor | None = None
    raw_substep_friction_position_w: torch.Tensor | None = None
    raw_substep_friction_force_w: torch.Tensor | None = None


def _quat_inverse_apply(quat_wxyz, vector):
    from ..reference_observations import _quat_to_matrix
    matrix = _quat_to_matrix(quat_wxyz).transpose(-1, -2)
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


class FootGroundMJLabLabeler:
    """Extract two-foot labels from raw slots after each 20ms control step."""

    def __init__(self, task, *, slip_threshold_m_s=.03,
                 impact_impulse_threshold_n_s=2.0):
        self.task = task
        self.bridge = task.bridge
        self.entity = task.bridge.entity
        self.sensor = task.bridge.env.scene[RAW_SENSOR_NAME]
        cfg = self.sensor.cfg
        if cfg.reduce != "maxforce" or cfg.num_slots < 2 or not cfg.global_frame:
            raise ValueError("unexpected raw Teacher sensor contract")
        self.slots = cfg.num_slots
        self.physics_dt = float(task.bridge.env.cfg.sim.timestep)
        self.slip_threshold = float(slip_threshold_m_s)
        self.impact_threshold = float(impact_impulse_threshold_n_s)
        names = tuple(self.entity.body_names)
        self.foot_ids = [names.index(name) for name in (
            "left_ankle_roll_link", "right_ankle_roll_link")]

    def _reshape(self, value, width=None):
        batch = self.bridge.num_envs
        tail = () if width is None else (width,)
        expected = (batch, 2 * self.slots) + tail
        if value.shape != expected:
            raise RuntimeError(f"raw contact tensor has {value.shape}, expected {expected}")
        return value.reshape(batch, 2, self.slots, *tail)

    @torch.no_grad()
    def read(self, *, friction: torch.Tensor, active_sliding: torch.Tensor,
             ground_velocity_w: torch.Tensor | None = None) -> FootGroundFrame:
        """Read the current frame; intent comes from reference, not measured slip."""
        batch = self.bridge.num_envs
        if friction.shape != (batch,) or active_sliding.shape != (batch, 2):
            raise ValueError("friction/active_sliding shape mismatch")
        c = self.sensor.data
        required = (c.found, c.force, c.pos, c.normal, c.force_history)
        if any(value is None for value in required):
            raise RuntimeError("raw contact sensor fields/history are incomplete")
        found = self._reshape(c.found) > 0
        force = self._reshape(c.force, 3)
        point = self._reshape(c.pos, 3)
        normal = self._reshape(c.normal, 3)
        data = self.entity.data
        pose = data.body_link_pose_w[:, self.foot_ids]
        velocity = data.body_link_vel_w[:, self.foot_ids]
        root_quat = data.root_link_quat_w
        from ..reference_observations import _quat_to_matrix
        heading = _quat_to_matrix(root_quat)[..., 0]
        heading = heading[:, None, :].expand(batch, 2, 3)
        if ground_velocity_w is None:
            ground_velocity_w = torch.zeros_like(point)
        if ground_velocity_w.shape != point.shape:
            raise ValueError("ground_velocity_w must match raw points [B,2,S,3]")
        per_foot = []
        for foot in range(2):
            per_foot.append(aggregate_contacts(
                found=found[:, foot], points=point[:, foot], normals=normal[:, foot],
                forces=force[:, foot], body_origin=pose[:, foot, :3],
                body_linear_velocity=velocity[:, foot, :3],
                body_angular_velocity=velocity[:, foot, 3:],
                ground_velocity=ground_velocity_w[:, foot], heading=heading[:, foot],
                friction=friction, physics_dt_s=self.physics_dt,
            ))
        contact = torch.stack([item.found for item in per_foot], 1)
        aggregate_point = torch.stack([item.point for item in per_foot], 1)
        aggregate_normal = torch.stack([item.normal for item in per_foot], 1)
        velocity_n = torch.stack([item.relative_velocity_normal for item in per_foot], 1)
        velocity_t = torch.stack([item.relative_velocity_tangent for item in per_foot], 1)
        normal_force = torch.stack([item.normal_force for item in per_foot], 1)
        tangent_force = torch.stack([item.tangential_force for item in per_foot], 1)
        slip = torch.stack([item.slip_speed for item in per_foot], 1)
        utilization = torch.stack([item.friction_utilization for item in per_foot], 1)
        # Integrate all raw slot forces over solver substeps. This is N*s.
        history = c.force_history.reshape(batch, 2, self.slots,
                                          c.force_history.shape[-2], 3)
        net_history = history.sum(2)
        normal_history = (net_history * aggregate_normal.unsqueeze(-2)).sum(-1).abs()
        impulse = normal_history.sum(-1) * self.physics_dt
        historical_contact = torch.linalg.vector_norm(history, dim=-1).amax((-1, -2)) > 1e-6
        impulse = torch.where(historical_contact, impulse, torch.zeros_like(impulse))
        point_local = _quat_inverse_apply(
            pose[..., 3:7], aggregate_point - pose[..., :3])
        normal_local = _quat_inverse_apply(pose[..., 3:7], aggregate_normal)
        mode = torch.zeros_like(contact, dtype=torch.long)
        mode = torch.where(contact, torch.ones_like(mode), mode)
        mode = torch.where(contact & (slip >= self.slip_threshold),
                           torch.full_like(mode, 2), mode)
        # A short impact may have separated again by the final solver substep.
        # Keep contact_probability as the end-of-step state while retaining the
        # within-step event as impact mode and a non-zero integrated impulse.
        mode = torch.where(historical_contact & (impulse >= self.impact_threshold),
                           torch.full_like(mode, 3), mode)
        total_load = normal_force.sum(-1, keepdim=True)
        load_ratio = torch.where(total_load > 1e-6,
                                 normal_force / total_load.clamp_min(1e-6),
                                 torch.zeros_like(normal_force))
        return FootGroundFrame(
            contact, mode, point_local, normal_local, velocity_n, velocity_t,
            slip, normal_force, tangent_force, impulse, load_ratio, utilization,
            found, point, normal, force,
        )
