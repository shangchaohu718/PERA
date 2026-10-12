"""HoloMotion v1.4 K1 tracking rewards expressed on backend-neutral tensors.

The formulas and weights mirror ``rew_motion_tracking.yaml``, including the
undesired-contact penalty exposed by the HoloMotion mjlab configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data.motion_sampler import BodyReferenceWindow
from .reference_observations import ReferenceWindow
from .robot_contract import get_robot_contract

K1_KEY_BODY_NAMES = (
    "trunk",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
    "aahead_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
)

K1_BODY_NAMES = (
    "trunk",
    "aahead_yaw_link",
    "aahead_pitch_link",
    "aaleft_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "aaright_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
)

K1_REWARD_POINT_BODY_NAMES = (
    "trunk",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

HOLOMOTION_UNDESIRED_CONTACT_SENSOR = "holomotion_undesired_contacts"
HOLOMOTION_AUX_CONTACT_SENSOR = "holomotion_aux_contacts"

K1_AUX_CONTACT_BODY_NAMES = (
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_knee_pitch_link",
    "right_knee_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_pitch_link",
    "right_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
)

K1_AUX_POSITION_BODY_NAMES = (
    "left_knee_pitch_link",
    "right_knee_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
)


@dataclass(frozen=True)
class HoloMotionMJLabState:
    """One post-physics-step robot state used by rewards and diagnostics."""

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_wxyz: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_wxyz: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    soft_joint_pos_limits: torch.Tensor
    undesired_contact_force_history: torch.Tensor | None = None

    @classmethod
    def from_adapter(
        cls,
        adapter,
        *,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> HoloMotionMJLabState:
        """Capture mjlab tensors after ``adapter.step`` without importing mjlab."""
        bridge = adapter.bridge
        data = bridge.entity.data
        body_names = tuple(bridge.entity.body_names)
        sensor = bridge.env.scene.sensors.get(HOLOMOTION_UNDESIRED_CONTACT_SENSOR)
        contact_history = None if sensor is None else sensor.data.force_history
        return cls(
            joint_names=tuple(bridge.joint_names),
            body_names=body_names,
            joint_pos=data.joint_pos,
            joint_vel=data.joint_vel,
            root_pos_w=data.root_link_pos_w,
            root_quat_wxyz=data.root_link_quat_w,
            root_lin_vel_w=data.root_link_lin_vel_w,
            root_ang_vel_w=data.root_link_ang_vel_w,
            body_pos_w=data.body_link_pos_w,
            body_quat_wxyz=data.body_link_quat_w,
            body_lin_vel_w=data.body_link_lin_vel_w,
            body_ang_vel_w=data.body_link_ang_vel_w,
            action=action,
            previous_action=previous_action,
            soft_joint_pos_limits=data.soft_joint_pos_limits,
            undesired_contact_force_history=contact_history,
        )


def _unit(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_inverse(q: torch.Tensor) -> torch.Tensor:
    q = _unit(q)
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_multiply(_quat_multiply(_unit(q), pure), _quat_inverse(q))[..., 1:]


def _quat_apply_inverse(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return _quat_apply(_quat_inverse(q), vector)


def _quat_error_magnitude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    relative = _unit(_quat_multiply(a, _quat_inverse(b)))
    scalar = relative[..., :1].abs().clamp(max=1.0)
    vector_norm = torch.linalg.vector_norm(relative[..., 1:], dim=-1, keepdim=True)
    return (2.0 * torch.atan2(vector_norm, scalar)).squeeze(-1)


def _name_indices(
    source: tuple[str, ...], target: tuple[str, ...], label: str
) -> list[int]:
    if len(source) != len(set(source)):
        raise ValueError(f"{label} names contain duplicates")
    lookup = {name: index for index, name in enumerate(source)}
    missing = [name for name in target if name not in lookup]
    if missing:
        raise ValueError(f"{label} is missing {missing}")
    return [lookup[name] for name in target]


def _check_state(state: HoloMotionMJLabState) -> int:
    robot = get_robot_contract("k1_22dof")
    robot.check_joint_set(state.joint_names, source="HoloMotion reward state")
    batch = state.joint_pos.shape[0]
    shapes = {
        "joint_pos": (batch, 22),
        "joint_vel": (batch, 22),
        "root_pos_w": (batch, 3),
        "root_quat_wxyz": (batch, 4),
        "root_lin_vel_w": (batch, 3),
        "root_ang_vel_w": (batch, 3),
        "body_pos_w": (batch, len(state.body_names), 3),
        "body_quat_wxyz": (batch, len(state.body_names), 4),
        "body_lin_vel_w": (batch, len(state.body_names), 3),
        "body_ang_vel_w": (batch, len(state.body_names), 3),
        "action": (batch, 22),
        "previous_action": (batch, 22),
        "soft_joint_pos_limits": (batch, 22, 2),
    }
    for name, shape in shapes.items():
        value = getattr(state, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"state.{name} must have shape {shape}")
        if (
            value.device != state.joint_pos.device
            or value.dtype != state.joint_pos.dtype
        ):
            raise ValueError(f"state.{name} device/dtype differs from joint_pos")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"state.{name} contains NaN/Inf")
    contact = state.undesired_contact_force_history
    if contact is not None:
        if (
            contact.ndim != 4
            or contact.shape[0] != batch
            or contact.shape[1] != 19
            or contact.shape[2] < 1
            or contact.shape[3] != 3
        ):
            raise ValueError(
                "undesired contact history must have shape [B,19,H,3], "
                f"got {tuple(contact.shape)}"
            )
        if (
            contact.device != state.joint_pos.device
            or contact.dtype != state.joint_pos.dtype
        ):
            raise ValueError(
                "undesired contact history device/dtype differs from joint_pos"
            )
        if not torch.isfinite(contact).all().item():
            raise ValueError("undesired contact history contains NaN/Inf")
    return batch


def holomotion_k1_reward_terms(
    state: HoloMotionMJLabState,
    reference: ReferenceWindow,
    body_reference: BodyReferenceWindow,
    *,
    reference_index: int = 1,
) -> dict[str, torch.Tensor]:
    """Compute the official K1 tracking reward and its diagnostic terms."""
    batch = _check_state(state)
    if reference_index < 0 or reference_index >= reference.joint_pos.shape[1]:
        raise ValueError("reference_index is outside the reference window")
    if not reference.valid_mask[:, reference_index].all().item():
        raise ValueError("selected reward reference frame must be valid")
    if not body_reference.valid_mask[:, reference_index].all().item():
        raise ValueError("selected body reward reference frame must be valid")
    if reference.joint_pos.shape[0] != batch or body_reference.pos_w.shape[0] != batch:
        raise ValueError("reference batch differs from state batch")

    body_ids = _name_indices(state.body_names, K1_KEY_BODY_NAMES, "mjlab bodies")
    ref_body_ids = _name_indices(
        body_reference.body_names, K1_KEY_BODY_NAMES, "reference bodies"
    )
    point_body_ids = _name_indices(
        state.body_names, K1_REWARD_POINT_BODY_NAMES, "mjlab reward points"
    )
    ref_point_body_ids = _name_indices(
        body_reference.body_names,
        K1_REWARD_POINT_BODY_NAMES,
        "reference reward points",
    )

    ref_root_pos = reference.root_pos_w[:, reference_index]
    ref_root_quat = reference.root_quat_wxyz[:, reference_index]
    ref_root_lin = reference.root_lin_vel_w[:, reference_index]
    ref_root_ang = reference.root_ang_vel_w[:, reference_index]

    robot_root_inv = _quat_inverse(state.root_quat_wxyz)
    ref_root_inv = _quat_inverse(ref_root_quat)
    robot_lin_local = _quat_apply(robot_root_inv, state.root_lin_vel_w)
    ref_lin_local = _quat_apply(ref_root_inv, ref_root_lin)
    robot_ang_local = _quat_apply(robot_root_inv, state.root_ang_vel_w)
    ref_ang_local = _quat_apply(ref_root_inv, ref_root_ang)
    lin_ratio = torch.linalg.vector_norm(robot_lin_local - ref_lin_local, dim=-1) / (
        torch.linalg.vector_norm(ref_lin_local, dim=-1) + 0.1
    )
    ang_ratio = torch.linalg.vector_norm(robot_ang_local - ref_ang_local, dim=-1) / (
        torch.linalg.vector_norm(ref_ang_local, dim=-1) + 0.1
    )

    robot_body_pos = state.body_pos_w[:, body_ids]
    ref_body_pos = body_reference.pos_w[:, reference_index, ref_body_ids]
    robot_body_quat = state.body_quat_wxyz[:, body_ids]
    ref_body_quat = body_reference.quat_wxyz[:, reference_index, ref_body_ids]
    key_count = len(K1_KEY_BODY_NAMES)
    robot_root_inv_key = robot_root_inv[:, None].expand(-1, key_count, -1)
    ref_root_inv_key = ref_root_inv[:, None].expand(-1, key_count, -1)
    robot_rel_pos = _quat_apply(
        robot_root_inv_key, robot_body_pos - state.root_pos_w[:, None]
    )
    ref_rel_pos = _quat_apply(ref_root_inv_key, ref_body_pos - ref_root_pos[:, None])
    pos_error = torch.square(robot_rel_pos - ref_rel_pos).sum(-1).mean(-1)
    robot_rel_quat = _quat_multiply(robot_root_inv_key, robot_body_quat)
    ref_rel_quat = _quat_multiply(ref_root_inv_key, ref_body_quat)
    rot_error = torch.square(_quat_error_magnitude(ref_rel_quat, robot_rel_quat)).mean(
        -1
    )

    offsets = state.joint_pos.new_zeros((len(K1_REWARD_POINT_BODY_NAMES), 3))
    offsets[0, 2] = 0.5
    robot_point_pos = state.body_pos_w[:, point_body_ids] + _quat_apply(
        state.body_quat_wxyz[:, point_body_ids], offsets[None]
    )
    ref_point_pos = body_reference.pos_w[
        :, reference_index, ref_point_body_ids
    ] + _quat_apply(
        body_reference.quat_wxyz[:, reference_index, ref_point_body_ids], offsets[None]
    )
    point_count = len(K1_REWARD_POINT_BODY_NAMES)
    robot_points_local = _quat_apply(
        robot_root_inv[:, None].expand(-1, point_count, -1),
        robot_point_pos - state.root_pos_w[:, None],
    )
    ref_points_local = _quat_apply(
        ref_root_inv[:, None].expand(-1, point_count, -1),
        ref_point_pos - ref_root_pos[:, None],
    )
    point_error = torch.square(robot_points_local - ref_points_local).sum(-1).mean(-1)

    ref_body_lin = body_reference.lin_vel_w[:, reference_index, ref_body_ids]
    ref_body_ang = body_reference.ang_vel_w[:, reference_index, ref_body_ids]
    lin_body_error = (
        torch.square(state.body_lin_vel_w[:, body_ids] - ref_body_lin).sum(-1).mean(-1)
    )
    ang_body_error = (
        torch.square(state.body_ang_vel_w[:, body_ids] - ref_body_ang).sum(-1).mean(-1)
    )

    lower = state.soft_joint_pos_limits[..., 0]
    upper = state.soft_joint_pos_limits[..., 1]
    joint_limit = (lower - state.joint_pos).clamp_min(0.0) + (
        state.joint_pos - upper
    ).clamp_min(0.0)
    joint_limit = joint_limit.sum(-1)
    if state.undesired_contact_force_history is None:
        undesired_contacts = state.joint_pos.new_zeros(batch)
    else:
        peak_force = torch.linalg.vector_norm(
            state.undesired_contact_force_history, dim=-1
        ).amax(dim=-1)
        undesired_contacts = (peak_force > 1.0).sum(-1).to(state.joint_pos.dtype)

    terms = {
        "is_alive": state.joint_pos.new_ones(batch),
        "root_rel_keybodylink_pos_tracking_l2_exp": torch.exp(-pos_error / 0.3**2),
        "root_rel_keybodylink_rot_tracking_l2_exp": torch.exp(-rot_error / 0.4**2),
        "root_lin_vel_tracking_rel_ratio_exp": torch.exp(-torch.square(lin_ratio)),
        "root_ang_vel_tracking_rel_ratio_exp": torch.exp(-torch.square(ang_ratio)),
        "local_reward_point_body_pos_tracking_exp": torch.exp(-point_error / 0.1**2),
        "global_keybodylink_lin_vel_tracking_l2_exp": torch.exp(
            -lin_body_error / 1.0**2
        ),
        "global_keybodylink_ang_vel_tracking_l2_exp": torch.exp(
            -ang_body_error / 3.14**2
        ),
        "action_rate_l2": torch.square(state.action - state.previous_action).sum(-1),
        "joint_pos_limits": joint_limit,
        "undesired_contacts": undesired_contacts,
        "root_position_error_m": torch.linalg.vector_norm(
            state.root_pos_w - ref_root_pos, dim=-1
        ),
        "root_orientation_error_rad": _quat_error_magnitude(
            state.root_quat_wxyz, ref_root_quat
        ),
        "root_linear_velocity_error_mps": torch.linalg.vector_norm(
            robot_lin_local - ref_lin_local, dim=-1
        ),
        "root_angular_velocity_error_radps": torch.linalg.vector_norm(
            robot_ang_local - ref_ang_local, dim=-1
        ),
        "keybody_linear_velocity_rmse_mps": torch.sqrt(lin_body_error),
        "keybody_angular_velocity_rmse_radps": torch.sqrt(ang_body_error),
    }
    terms["weighted_total"] = (
        0.5 * terms["is_alive"]
        + terms["root_rel_keybodylink_pos_tracking_l2_exp"]
        + terms["root_rel_keybodylink_rot_tracking_l2_exp"]
        + 2.0 * terms["root_lin_vel_tracking_rel_ratio_exp"]
        + 2.0 * terms["root_ang_vel_tracking_rel_ratio_exp"]
        + 5.0 * terms["local_reward_point_body_pos_tracking_exp"]
        + terms["global_keybodylink_lin_vel_tracking_l2_exp"]
        + terms["global_keybodylink_ang_vel_tracking_l2_exp"]
        - 0.1 * terms["action_rate_l2"]
        - 10.0 * terms["joint_pos_limits"]
        - 0.1 * terms["undesired_contacts"]
    )
    # Compatibility alias for audit reports written before contact support.
    terms["known_weighted_total"] = terms["weighted_total"]
    return terms


def gravity_projection_distance(
    robot_root_quat_wxyz: torch.Tensor,
    reference_root_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Official motion termination signal before applying threshold 0.8."""
    gravity = robot_root_quat_wxyz.new_tensor((0.0, 0.0, -1.0))
    robot = _quat_apply_inverse(
        robot_root_quat_wxyz, gravity.expand_as(robot_root_quat_wxyz[..., :3])
    )
    reference = _quat_apply_inverse(
        reference_root_quat_wxyz,
        gravity.expand_as(reference_root_quat_wxyz[..., :3]),
    )
    return torch.linalg.vector_norm(robot - reference, dim=-1)


def _yaw_quaternion(q: torch.Tensor) -> torch.Tensor:
    q = _unit(q)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return torch.stack(
        (
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        ),
        dim=-1,
    )


def _rotation_6d(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = _unit(q).unbind(-1)
    matrix = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    return matrix[..., :2].reshape(*q.shape[:-1], 6)


class HoloMotionK1CriticObservationBuilder:
    """Build the exact 843-value asymmetric critic input used by the K1 run."""

    TERM_WIDTHS = (
        ("critic_ref_dof_pos_cur", 22),
        ("critic_global_anchor_diff", 9),
        ("critic_ref_motion_cur_heading_aligned_root_pos", 3),
        ("critic_ref_motion_cur_heading_aligned_root_rot6d", 6),
        ("critic_ref_motion_cur_heading_aligned_root_lin_vel", 3),
        ("critic_ref_motion_cur_heading_aligned_root_ang_vel", 3),
        ("critic_rel_robot_root_lin_vel", 3),
        ("critic_rel_robot_root_ang_vel", 3),
        ("critic_global_robot_bodylink_lin_vel_flat", 69),
        ("critic_global_robot_bodylink_ang_vel_flat", 69),
        ("critic_root_rel_robot_bodylink_pos_flat", 69),
        ("critic_root_rel_robot_bodylink_rot_mat_flat", 138),
        ("critic_dof_pos", 22),
        ("critic_dof_vel", 22),
        ("critic_last_action", 22),
        ("critic_ref_dof_pos_fut", 220),
        ("critic_ref_root_height_fut", 10),
        ("critic_ref_motion_fut_heading_aligned_root_pos", 30),
        ("critic_ref_motion_fut_heading_aligned_root_rot6d", 60),
        ("critic_ref_motion_fut_heading_aligned_root_lin_vel", 30),
        ("critic_ref_motion_fut_heading_aligned_root_ang_vel", 30),
    )
    OBS_DIM = 843

    def __init__(self, critic_joint_names: tuple[str, ...]) -> None:
        get_robot_contract("k1_22dof").check_joint_set(
            critic_joint_names, source="HoloMotion critic joint order"
        )
        self.critic_joint_names = tuple(critic_joint_names)
        offset = 0
        self.term_slices = {}
        for name, width in self.TERM_WIDTHS:
            self.term_slices[name] = slice(offset, offset + width)
            offset += width
        if offset != self.OBS_DIM:
            raise RuntimeError("HoloMotion critic schema width is inconsistent")

    def __call__(
        self,
        state: HoloMotionMJLabState,
        reference: ReferenceWindow,
        body_reference: BodyReferenceWindow,
    ) -> torch.Tensor:
        batch = _check_state(state)
        if reference.joint_pos.shape[:2] != (batch, 12):
            raise ValueError("critic reference must contain current + 11 future frames")
        if body_reference.pos_w.shape[:2] != (batch, 12):
            raise ValueError(
                "critic body reference must contain current + 11 future frames"
            )
        if not reference.valid_mask[:, :11].all().item():
            raise ValueError("critic current and ten future frames must be valid")

        ref_joint_ids = _name_indices(
            reference.joint_names, self.critic_joint_names, "critic reference joints"
        )
        robot_joint_ids = _name_indices(
            state.joint_names, self.critic_joint_names, "critic robot joints"
        )
        body_ids = _name_indices(
            state.body_names, body_reference.body_names, "critic mjlab bodies"
        )
        if len(body_ids) != 23:
            raise ValueError("the K1 critic requires all 23 robot bodies")

        ref_root_pos = reference.root_pos_w[:, :11]
        ref_root_quat = reference.root_quat_wxyz[:, :11]
        ref_root_lin = reference.root_lin_vel_w[:, :11]
        ref_root_ang = reference.root_ang_vel_w[:, :11]
        heading = _yaw_quaternion(state.root_quat_wxyz)
        heading_inv = _quat_inverse(heading)
        heading_all = heading_inv[:, None].expand(-1, 11, -1)
        ref_heading_pos = _quat_apply(
            heading_all, ref_root_pos - state.root_pos_w[:, None]
        )
        ref_heading_quat = _quat_multiply(heading_all, ref_root_quat)
        ref_heading_lin = _quat_apply(heading_all, ref_root_lin)
        ref_heading_ang = _quat_apply(heading_all, ref_root_ang)

        anchor_pos_diff = _quat_apply(
            _quat_inverse(state.root_quat_wxyz),
            ref_root_pos[:, 0] - state.root_pos_w,
        )
        anchor_rot_diff = _quat_multiply(
            _quat_inverse(state.root_quat_wxyz), ref_root_quat[:, 0]
        )
        global_anchor_diff = torch.cat(
            (anchor_pos_diff, _rotation_6d(anchor_rot_diff)), dim=-1
        )

        root_inv_bodies = _quat_inverse(state.root_quat_wxyz)[:, None].expand(
            -1, 23, -1
        )
        body_pos = state.body_pos_w[:, body_ids]
        body_quat = state.body_quat_wxyz[:, body_ids]
        body_rel_pos = _quat_apply(
            root_inv_bodies, body_pos - state.root_pos_w[:, None]
        )
        body_rel_quat = _quat_multiply(root_inv_bodies, body_quat)

        current_terms = (
            reference.joint_pos[:, 0, ref_joint_ids],
            global_anchor_diff,
            ref_heading_pos[:, 0],
            _rotation_6d(ref_heading_quat[:, 0]),
            ref_heading_lin[:, 0],
            ref_heading_ang[:, 0],
            _quat_apply_inverse(state.root_quat_wxyz, state.root_lin_vel_w),
            _quat_apply_inverse(state.root_quat_wxyz, state.root_ang_vel_w),
            state.body_lin_vel_w[:, body_ids].reshape(batch, -1),
            state.body_ang_vel_w[:, body_ids].reshape(batch, -1),
            body_rel_pos.reshape(batch, -1),
            _rotation_6d(body_rel_quat).reshape(batch, -1),
            state.joint_pos[:, robot_joint_ids],
            state.joint_vel[:, robot_joint_ids],
            state.action,
        )
        future_terms = (
            reference.joint_pos[:, 1:11, ref_joint_ids],
            ref_root_pos[:, 1:11, 2:3],
            ref_heading_pos[:, 1:11],
            _rotation_6d(ref_heading_quat[:, 1:11]),
            ref_heading_lin[:, 1:11],
            ref_heading_ang[:, 1:11],
        )
        observation = torch.cat(
            (*current_terms, *(value.reshape(batch, -1) for value in future_terms)),
            dim=-1,
        )
        if observation.shape != (batch, self.OBS_DIM):
            raise RuntimeError(
                f"assembled critic observation has shape {tuple(observation.shape)}"
            )
        if not torch.isfinite(observation).all().item():
            raise RuntimeError("critic observation contains NaN/Inf")
        return observation
