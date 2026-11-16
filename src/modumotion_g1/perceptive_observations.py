"""Differentiable G1 observation/action bridge for HoloMotion imagination.

The frozen public actor still receives its released 604-dimensional contract.
Scene features enter domain plugins separately. Reference inputs here must be
the *raw* command, never the terrain-optimized reward reference. Instinct uses
torso_link as its floating base; HoloMotion uses pelvis and a different joint
order. All state-dependent operations below retain autograd.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

CURRENT_TERMS = (
    "actor_ref_gravity_projection_cur",
    "actor_ref_base_linvel_cur",
    "actor_ref_base_angvel_cur",
    "actor_ref_dof_pos_cur",
    "actor_ref_root_height_cur",
    "actor_ref_robot_yaw_error_sin_cos",
    "actor_projected_gravity",
    "actor_rel_robot_root_ang_vel",
    "actor_dof_pos",
    "actor_dof_vel",
    "actor_last_action",
)
FUTURE_TERMS = (
    "actor_ref_dof_pos_fut",
    "actor_ref_root_height_fut",
    "actor_ref_gravity_projection_fut",
    "actor_ref_base_linvel_fut",
    "actor_ref_base_angvel_fut",
    "actor_ref_future_yaw_delta_sin_cos",
    "actor_ref_future_root_ori_robot_frame_6d",
)
ROUTER_TERMS = CURRENT_TERMS[:5] + FUTURE_TERMS
ACTOR_TERMS = CURRENT_TERMS + FUTURE_TERMS


@dataclass(frozen=True)
class PerceptiveActorObservationInput:
    """Same tensor semantics as HoloMotion MotionActorObservationInput."""

    reference_qpos: torch.Tensor
    robot_root_quat_wxyz: torch.Tensor
    robot_root_angvel_local: torch.Tensor
    robot_dof_pos: torch.Tensor
    robot_dof_vel: torch.Tensor
    last_action: torch.Tensor
    default_dof_pos: torch.Tensor
    reference_dof_indices: torch.Tensor | None = None
    reference_sample_time: torch.Tensor | None = None
    reference_yaw_alignment_wxyz: torch.Tensor | None = None


def _qnormalize(q):
    return F.normalize(q, dim=-1, eps=1e-9)


def _qmul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat(
        (
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.cross(av, bv, dim=-1),
        ),
        -1,
    )


def _qinv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def _qrotate_inverse(q, v):
    qv = -q[..., 1:]
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def _matquat(m):
    # Largest-component branch is smooth in a neighborhood of each rotation;
    # clamping protects unselected square-root branches from NaN gradients.
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    a = (
        torch.sqrt(
            torch.stack(
                (
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                -1,
            ).clamp_min(1e-10)
        )
        / 2
    )
    w, x, y, z = a.unbind(-1)
    m01, m02, m10 = m[..., 0, 1], m[..., 0, 2], m[..., 1, 0]
    m12, m20, m21 = m[..., 1, 2], m[..., 2, 0], m[..., 2, 1]
    candidates = torch.stack(
        (
            torch.stack(
                (
                    w,
                    (m21 - m12) / (4 * w),
                    (m02 - m20) / (4 * w),
                    (m10 - m01) / (4 * w),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m21 - m12) / (4 * x),
                    x,
                    (m01 + m10) / (4 * x),
                    (m02 + m20) / (4 * x),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m02 - m20) / (4 * y),
                    (m01 + m10) / (4 * y),
                    y,
                    (m12 + m21) / (4 * y),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m10 - m01) / (4 * z),
                    (m02 + m20) / (4 * z),
                    (m12 + m21) / (4 * z),
                    z,
                ),
                -1,
            ),
        ),
        -2,
    )
    index = a.argmax(-1)
    q = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    return _qnormalize(q) * torch.where(q[..., :1] < 0, -1.0, 1.0)


def _yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _gravity(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)),
        -1,
    )


def flatten_actor_terms(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Canonical term-major layout used by the official TensorDictAssembler."""
    return torch.cat([terms[n].flatten(1) for n in ACTOR_TERMS], -1)


def flatten_router_terms(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Released 509-D reference router order (604 indices 0:39,134:604)."""
    return torch.cat([terms[n].flatten(1) for n in ROUTER_TERMS], -1)


def build_perceptive_actor_terms(
    inputs: PerceptiveActorObservationInput,
    *,
    fps: float = 50.0,
    current_index: int = 1,
    num_future_frames: int = 10,
) -> dict[str, torch.Tensor]:
    """Pure-Torch equivalent of HoloMotion's canonical non-autograd Warp path.

    Include one padding frame before current and one after the final future
    frame to retain the release's centered finite-difference velocities.
    """
    qp = inputs.reference_qpos
    if qp.ndim != 3 or qp.shape[-1] != 36:
        raise ValueError("reference_qpos must be [B,T,36]")
    batch, frames, _ = qp.shape
    end = current_index + 1 + num_future_frames
    if current_index < 0 or end > frames or frames < 2:
        raise ValueError("reference sequence does not cover requested frames")
    if fps <= 0:
        raise ValueError("fps must be positive")
    times = inputs.reference_sample_time
    if times is None:
        times = (torch.arange(frames, device=qp.device, dtype=qp.dtype) / fps)[None].expand(
            batch, -1
        )
    else:
        times = times.to(qp)
        if times.ndim == 1:
            times = times[None].expand(batch, -1)
        if times.shape != (batch, frames):
            raise ValueError("reference_sample_time must be [T] or [B,T]")
    lo = (torch.arange(frames, device=qp.device) - 1).clamp_min(0)
    hi = (lo + 2).clamp_max(frames - 1)
    # The first frame uses a forward difference.
    hi = torch.where(torch.arange(frames, device=qp.device) == 0, 1, hi)
    dt = (times[:, hi] - times[:, lo]).clamp_min(1e-6).unsqueeze(-1)
    lin_world = (qp[:, hi, :3] - qp[:, lo, :3]) / dt
    quat = _qnormalize(qp[..., 3:7])
    delta = _qnormalize(_qmul(quat[:, hi], _qinv(quat[:, lo])))
    delta = delta * torch.where(delta[..., :1] < 0, -1.0, 1.0)
    magnitude = delta[..., 1:].norm(dim=-1, keepdim=True)
    scale = torch.where(
        magnitude > 1e-7,
        2 * torch.atan2(magnitude, delta[..., :1]) / magnitude.clamp_min(1e-9),
        torch.full_like(magnitude, 2),
    )
    ang_world = delta[..., 1:] * scale / dt
    lin_local = _qrotate_inverse(quat, lin_world)
    ang_local = _qrotate_inverse(quat, ang_world)
    gravity = _gravity(quat)
    dof_idx = inputs.reference_dof_indices
    if dof_idx is None:
        dof_idx = torch.arange(29, device=qp.device)
    dof_idx = dof_idx.to(device=qp.device, dtype=torch.long)
    if dof_idx.shape != (29,):
        raise ValueError("reference_dof_indices must contain 29 indices")
    q = qp[..., 7:].index_select(-1, dof_idx)
    robot_quat = _qnormalize(inputs.robot_root_quat_wxyz)
    aligned = quat
    if inputs.reference_yaw_alignment_wxyz is not None:
        alignment = _qnormalize(inputs.reference_yaw_alignment_wxyz)
        aligned = _qnormalize(_qmul(alignment[:, None].expand_as(quat), quat))
    cur = current_index
    fut = slice(cur + 1, end)
    error = _yaw(aligned[:, cur]) - _yaw(robot_quat)
    future_yaw = _yaw(quat[:, fut]) - _yaw(quat[:, cur, None])
    relative = _qnormalize(
        _qmul(
            _qinv(robot_quat)[:, None].expand_as(aligned[:, fut]),
            aligned[:, fut],
        )
    )
    values = (
        gravity[:, cur],
        lin_local[:, cur],
        ang_local[:, cur],
        q[:, cur],
        qp[:, cur, 2:3],
        torch.stack((error.sin(), error.cos()), -1),
        _gravity(robot_quat),
        inputs.robot_root_angvel_local,
        inputs.robot_dof_pos - inputs.default_dof_pos,
        inputs.robot_dof_vel,
        inputs.last_action,
        q[:, fut],
        qp[:, fut, 2:3],
        gravity[:, fut],
        lin_local[:, fut],
        ang_local[:, fut],
        torch.stack((future_yaw.sin(), future_yaw.cos()), -1),
        _qmat(relative)[..., :, :2].flatten(-2),
    )
    return dict(zip(ACTOR_TERMS, values, strict=True))


def sample_holomotion_reference(
    raw_qpos: torch.Tensor,
    frame: torch.Tensor,
    phase: torch.Tensor,
    clip_start: torch.Tensor,
    clip_end: torch.Tensor,
    *,
    reference_fps: float,
    actor_dt: float = 0.02,
    num_future_frames: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 50Hz raw commands independently of Instinct's 0.1s goal stride.

    ``frame`` is the global current index; clip_start/end are per-environment
    inclusive bounds. Root/joints use linear and quaternions shortest-path
    spherical interpolation. Clamping never crosses into the next motion.
    """
    if raw_qpos.ndim != 2 or raw_qpos.shape[-1] != 36:
        raise ValueError("raw_qpos must be a concatenated [frames,36] library")
    if reference_fps <= 0 or actor_dt <= 0:
        raise ValueError("reference_fps and actor_dt must be positive")
    offsets = (
        torch.arange(
            -1,
            num_future_frames + 2,
            device=raw_qpos.device,
            dtype=raw_qpos.dtype,
        )
        * actor_dt
    )
    x = frame[:, None] + phase[:, None] + offsets[None] * reference_fps
    x = x.maximum(clip_start[:, None]).minimum(clip_end[:, None])
    low = x.floor().long()
    high = (low + 1).minimum(clip_end[:, None])
    t = (x - low).unsqueeze(-1)
    a, b = raw_qpos[low], raw_qpos[high]
    qa, qb = _qnormalize(a[..., 3:7]), _qnormalize(b[..., 3:7])
    dot = (qa * qb).sum(-1, keepdim=True)
    qb = qb * torch.where(dot < 0, -1.0, 1.0)
    dot = dot.abs().clamp(max=1 - 1e-7)
    angle = torch.acos(dot)
    sphere = (torch.sin((1 - t) * angle) * qa + torch.sin(t * angle) * qb) / angle.sin()
    quat = _qnormalize(torch.where(dot > 0.9995, torch.lerp(qa, qb, t), sphere))
    interpolated = torch.lerp(a, b, t)
    return torch.cat((interpolated[..., :3], quat, interpolated[..., 7:]), -1), offsets


class HoloMotionObservationAdapter:
    """Named-joint, pelvis-root, and action-scale bridge around Instinct state.

    ``kinematics`` is G1Kinematics with pelvis included in its tracked names.
    Simulator action offset/scale may include randomized offsets; supply those
    per call when converting normalized Holo actions to simulator actions.
    """

    def __init__(self, joint_names, kinematics, contract=None, sim_joint_signs=None):
        if contract is None:
            path = Path(__file__).resolve().parents[1] / "configs/g1_actor_io_contract.json"
            contract = json.loads(path.read_text())
        self.names = list(contract["joint_names"])
        self.joint_names = list(joint_names)
        if len(set(joint_names)) != 29 or set(joint_names) != set(self.names):
            raise ValueError("Expected the same 29 named G1 joints")
        self.to_holo = [self.joint_names.index(n) for n in self.names]
        self.to_sim = [self.names.index(n) for n in self.joint_names]
        # Instinct reverses the waist kinematic chain when rerooting to torso,
        # while retaining the same positive axes. The three physical joint
        # coordinates therefore have the opposite sign to Holo's pelvis root.
        if sim_joint_signs is None:
            sim_joint_signs = {n: -1.0 if n.startswith("waist_") else 1.0 for n in self.joint_names}
        self.sim_joint_signs = torch.tensor([sim_joint_signs[n] for n in self.joint_names])
        if not (self.sim_joint_signs.abs() == 1).all():
            raise ValueError("sim_joint_signs must be +1 or -1")
        self.holo_joint_signs = self.sim_joint_signs[self.to_holo]
        self.kinematics = kinematics
        tracked_names = [kinematics.body_names[i] for i in kinematics.tracked_idx]
        self.pelvis_index = tracked_names.index("pelvis")
        self.default_pos = torch.tensor(contract["default_joint_pos"])
        self.action_offset = torch.tensor(contract["action_offset"])
        self.action_scale = torch.tensor(contract["action_scale"])
        self._reference_cache = {}

    def pelvis_state(self, state):
        return tuple(t[:, self.pelvis_index] for t in self.kinematics.forward(*state))

    def reference_from_motion(self, raw_motion, chunk_size=4096):
        """Convert an explicitly raw MotionLib torso reference to pelvis qpos.

        This is an independent library: callers may use another optimized
        MotionLib for reward without changing actor command data.
        """
        result = []
        for start in range(0, len(raw_motion.q), chunk_size):
            q = raw_motion.q[start : start + chunk_size]
            z = torch.zeros_like(raw_motion.root_p[start : start + chunk_size])
            state = (
                raw_motion.root_p[start : start + chunk_size],
                raw_motion.root_R[start : start + chunk_size],
                z,
                z,
                q,
                torch.zeros_like(q),
            )
            p, r, _, _ = self.pelvis_state(state)
            holo_q = q[:, self.to_holo] * self.holo_joint_signs.to(q)
            result.append(torch.cat((p, _matquat(r), holo_q), -1))
        return torch.cat(result)

    def reference_window(self, raw_motion_lib, frame, phase, origin=None):
        """Raw MotionLib + f/phase/origin -> 13-frame 50Hz pelvis reference.

        ``origin`` is the current per-env terrain origin, not a learned or
        optimized offset. Passing it preserves the actual world root height.
        For a distinct raw clip library the caller supplies that library's
        own frame/phase mapping; reward-library frame IDs cannot be reused
        blindly when clips or lengths differ.
        """
        key = id(raw_motion_lib)
        if key not in self._reference_cache:
            self._reference_cache[key] = (
                raw_motion_lib,
                self.reference_from_motion(raw_motion_lib),
            )
        raw_qpos = self._reference_cache[key][1]
        clip = raw_motion_lib.clip_id[frame]
        qpos, times = sample_holomotion_reference(
            raw_qpos,
            frame,
            phase,
            raw_motion_lib.clip_start[clip],
            raw_motion_lib.clip_end[clip],
            reference_fps=raw_motion_lib.framerate,
        )
        if origin is not None:
            qpos = torch.cat((qpos[..., :3] + origin[:, None], qpos[..., 3:]), -1)
        return qpos, times

    def terms(
        self,
        state,
        last_holo_action,
        raw_reference_qpos,
        *,
        reference_sample_time=None,
        reference_yaw_alignment_wxyz=None,
        **kwargs,
    ):
        """Derive actor terms from a differentiable predicted torso state."""
        _, rotation, _, w = self.pelvis_state(state)
        local_w = (rotation.transpose(-1, -2) * w.unsqueeze(-2)).sum(-1)
        return build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=raw_reference_qpos,
                robot_root_quat_wxyz=_matquat(rotation),
                robot_root_angvel_local=local_w,
                robot_dof_pos=state[4][:, self.to_holo] * self.holo_joint_signs.to(state[4]),
                robot_dof_vel=state[5][:, self.to_holo] * self.holo_joint_signs.to(state[5]),
                last_action=last_holo_action,
                default_dof_pos=self.default_pos.to(state[4]),
                reference_sample_time=reference_sample_time,
                reference_yaw_alignment_wxyz=reference_yaw_alignment_wxyz,
            ),
            **kwargs,
        )

    def action_to_sim(self, holo_action, sim_offset, sim_scale):
        target = (
            self.action_offset.to(holo_action) + self.action_scale.to(holo_action) * holo_action
        )
        target = target[:, self.to_sim] * self.sim_joint_signs.to(target)
        return (target - sim_offset) / sim_scale

    def action_from_sim(self, sim_action, sim_offset, sim_scale):
        target = sim_offset + sim_scale * sim_action
        target = target[:, self.to_holo] * self.holo_joint_signs.to(target)
        return (target - self.action_offset.to(sim_action)) / self.action_scale.to(sim_action)
