"""Official HoloMotion K1 actor contract on top of the local mjlab bridge.

This module deliberately does not reimplement the HoloMotion policy or PPO.  It
only translates mjlab state and a robot reference window to the 506-value actor
input exported by the official HoloMotion K1 training run, and translates the
22 actor outputs back to the canonical K1 policy joint order.

The exported actor order is read from ONNX metadata.  No positional assumption
is made about either the ONNX output or the mjlab articulation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .reference_observations import ReferenceWindow
from .robot_contract import RobotContract, get_robot_contract


def _csv_floats(value: str, *, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"ONNX metadata {name!r} is not a float list") from error
    if not result or not torch.isfinite(torch.tensor(result)).all().item():
        raise ValueError(f"ONNX metadata {name!r} must contain finite values")
    return result


def _csv_names(value: str, *, name: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(","))
    if (
        not result
        or any(not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"ONNX metadata {name!r} contains empty or duplicate names")
    return result


@dataclass(frozen=True)
class HoloMotionOnnxContract:
    """The deployment fields embedded in an official HoloMotion actor."""

    joint_names: tuple[str, ...]
    joint_stiffness: tuple[float, ...]
    joint_damping: tuple[float, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    obs_dim: int = 506
    future_frames: int = 10

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        if count != 22:
            raise ValueError(f"the K1 actor must expose 22 joints, got {count}")
        numeric_fields = (
            self.joint_stiffness,
            self.joint_damping,
            self.default_joint_pos,
            self.action_scale,
        )
        if any(len(field) != count for field in numeric_fields):
            raise ValueError("ONNX joint metadata sizes differ")
        if any(
            value <= 0
            for field in (
                self.joint_stiffness,
                self.joint_damping,
                self.action_scale,
            )
            for value in field
        ):
            raise ValueError(
                "ONNX stiffness/damping/action_scale values must be positive"
            )
        expected = 18 + 4 * count + self.future_frames * (18 + count)
        if self.obs_dim != expected:
            raise ValueError(
                f"K1 HoloMotion observation must be {expected}, got {self.obs_dim}"
            )
        get_robot_contract("k1_22dof").check_joint_set(
            self.joint_names, source="HoloMotion ONNX metadata"
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        obs_dim: int = 506,
        future_frames: int = 10,
    ) -> HoloMotionOnnxContract:
        required = (
            "joint_names",
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "action_scale",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError(f"ONNX metadata is missing {missing}")
        return cls(
            joint_names=_csv_names(metadata["joint_names"], name="joint_names"),
            joint_stiffness=_csv_floats(
                metadata["joint_stiffness"], name="joint_stiffness"
            ),
            joint_damping=_csv_floats(metadata["joint_damping"], name="joint_damping"),
            default_joint_pos=_csv_floats(
                metadata["default_joint_pos"], name="default_joint_pos"
            ),
            action_scale=_csv_floats(metadata["action_scale"], name="action_scale"),
            obs_dim=obs_dim,
            future_frames=future_frames,
        )

    @classmethod
    def from_onnx(cls, path: str | Path) -> HoloMotionOnnxContract:
        """Read the graph input width and deployment metadata without inference."""
        import onnx

        model = onnx.load(str(Path(path).expanduser()), load_external_data=False)
        inputs = {item.name: item for item in model.graph.input}
        if "obs" not in inputs:
            raise ValueError("HoloMotion ONNX graph has no 'obs' input")
        dims = inputs["obs"].type.tensor_type.shape.dim
        if len(dims) != 2 or not dims[1].dim_value:
            raise ValueError("HoloMotion ONNX 'obs' must have a static feature width")
        metadata = {item.key: item.value for item in model.metadata_props}
        return cls.from_metadata(metadata, obs_dim=int(dims[1].dim_value))

    def tensor(self, name: str, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            getattr(self, name), device=like.device, dtype=like.dtype
        )


def make_holomotion_k1_mjlab_cfg(
    asset_root: str | Path,
    contract: HoloMotionOnnxContract,
    *,
    num_envs: int = 1,
    physics_dt: float = 0.005,
    decimation: int = 4,
):
    """Create a K1 mjlab config with HoloMotion action and 200 Hz sim semantics.

    The generic K1 smoke environment terminates on absolute root height.  The
    official motion task instead terminates on gravity-projection divergence,
    which depends on the active reference and is therefore applied by the task
    adapter/evaluator rather than mjlab's reference-free manager.
    """
    from .envs.mjlab_k1 import (
        K1ControlSpec,
        make_k1_mjlab_cfg,
        official_xml_path,
        read_k1_joint_specs,
    )

    xml_specs = {
        item.name: item for item in read_k1_joint_specs(official_xml_path(asset_root))
    }
    controls = []
    for index, name in enumerate(contract.joint_names):
        source = xml_specs[name]
        kp = contract.joint_stiffness[index]
        kd = contract.joint_damping[index]
        scale = contract.action_scale[index]
        exported_default = contract.default_joint_pos[index]
        default = min(max(exported_default, source.lower), source.upper)
        if abs(exported_default - default) > 0.02:
            raise ValueError(
                f"ONNX default position for {name} exceeds the MJCF limit by more than 0.02 rad"
            )
        span = source.upper - source.lower
        lower = min(source.lower + 0.05 * span, default)
        upper = max(source.upper - 0.05 * span, default)
        # The K1 HoloMotion config defines scale = 0.25 * effort / stiffness.
        effort = 4.0 * scale * kp
        controls.append(
            K1ControlSpec(
                name=name,
                kp=kp,
                kd=kd,
                effort=effort,
                armature=source.armature,
                action_scale=scale,
                default_pos=default,
                soft_lower=lower,
                soft_upper=upper,
                zero_target=default,
            )
        )
    cfg = make_k1_mjlab_cfg(
        asset_root,
        num_envs=num_envs,
        physics_dt=physics_dt,
        decimation=decimation,
        control_profile=controls,
    )
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    from .holomotion_mjlab_rewards import (
        HOLOMOTION_AUX_CONTACT_SENSOR,
        K1_AUX_CONTACT_BODY_NAMES,
        K1_BODY_NAMES,
    )
    from .envs.mjlab_physics import CONTACT_SENSOR_NAME

    allowed_contacts = {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_elbow_yaw_link",
        "right_elbow_yaw_link",
    }
    undesired_contact_bodies = tuple(
        name for name in K1_BODY_NAMES if name not in allowed_contacts
    )
    if len(undesired_contact_bodies) != 19:
        raise RuntimeError("K1 undesired-contact body contract must contain 19 bodies")

    cfg.scene.sensors = (
        ContactSensorCfg(
            name="holomotion_undesired_contacts",
            primary=ContactMatch(
                mode="body",
                pattern=undesired_contact_bodies,
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="maxforce",
            num_slots=1,
            history_length=decimation,
        ),
        ContactSensorCfg(
            name=HOLOMOTION_AUX_CONTACT_SENSOR,
            primary=ContactMatch(
                mode="body",
                pattern=K1_AUX_CONTACT_BODY_NAMES,
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="maxforce",
            num_slots=1,
            track_air_time=True,
        ),
        # Deployable input for DynamicsModel.  Keep this separate from the
        # auxiliary-label sensor above: the physical encoder needs the actual
        # contact point, normal and force, not only a binary contact flag.
        ContactSensorCfg(
            name=CONTACT_SENSOR_NAME,
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal", "tangent"),
            reduce="maxforce",
            global_frame=True,
            num_slots=1,
        ),
    )
    cfg.terminations.pop("fallen", None)
    return cfg


@dataclass(frozen=True)
class HoloMotionRobotFrame:
    """One batched mjlab state before HoloMotion observation assembly."""

    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_quat_wxyz: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor


def _unit(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-12)


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


def _quat_apply_inverse(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_multiply(_quat_multiply(_quat_inverse(q), pure), _unit(q))[..., 1:]


def _angular_velocity_wxyz(
    q0: torch.Tensor, q1: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """Match HoloMotion's shortest-arc quaternion finite difference."""
    relative = _unit(_quat_multiply(q1, _quat_inverse(q0)))
    relative = torch.where(relative[..., :1] < 0.0, -relative, relative)
    scalar = relative[..., :1]
    vector = relative[..., 1:]
    magnitude = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = torch.where(
        magnitude > 1.0e-7,
        2.0 * torch.atan2(magnitude, scalar) / magnitude.clamp_min(1.0e-12),
        torch.full_like(magnitude, 2.0),
    )
    return vector * scale / dt


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = _unit(q).unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = _unit(q).unbind(-1)
    return torch.stack(
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


def _indices(source: Sequence[str], target: Sequence[str], *, label: str) -> list[int]:
    source_names = tuple(source)
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"{label} source joint names contain duplicates")
    lookup = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target if name not in lookup]
    if missing:
        raise ValueError(f"{label} is missing joints {missing}")
    return [lookup[name] for name in target]


class HoloMotionActorObservationBuilder:
    """Build the exact 506-D K1 actor vector used by HoloMotion v1.4."""

    def __init__(self, contract: HoloMotionOnnxContract) -> None:
        self.contract = contract
        self.robot: RobotContract = get_robot_contract("k1_22dof")

    def _validate_frame(self, frame: HoloMotionRobotFrame) -> int:
        batch = frame.joint_pos.shape[0]
        count = len(frame.joint_names)
        expected = {
            "joint_pos": (batch, count),
            "joint_vel": (batch, count),
            "root_quat_wxyz": (batch, 4),
            "root_ang_vel_b": (batch, 3),
            "projected_gravity_b": (batch, 3),
        }
        for name, shape in expected.items():
            value = getattr(frame, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"robot {name} must have shape {shape}")
            if (
                value.device != frame.joint_pos.device
                or value.dtype != frame.joint_pos.dtype
            ):
                raise ValueError(f"robot {name} device/dtype differs from joint_pos")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"robot {name} contains NaN/Inf")
        self.robot.check_joint_set(frame.joint_names, source="mjlab HoloMotion frame")
        norms = torch.linalg.vector_norm(frame.root_quat_wxyz, dim=-1)
        if ((norms - 1.0).abs() > 1.0e-3).any().item():
            raise ValueError("robot root quaternion must be normalized")
        return batch

    def __call__(
        self,
        frame: HoloMotionRobotFrame,
        reference: ReferenceWindow,
        last_action_onnx: torch.Tensor,
        *,
        previous_reference_root_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = self._validate_frame(frame)
        future = self.contract.future_frames
        horizon = future + 1
        context_frames = horizon + 1
        if reference.joint_pos.shape[:2] != (batch, context_frames):
            raise ValueError(
                "reference window must contain current + "
                f"{future} actor future frames + one velocity context frame"
            )
        if (
            reference.valid_mask.shape != (batch, context_frames)
            or not reference.valid_mask[:, 0].all().item()
        ):
            raise ValueError("reference valid_mask must include a valid current frame")
        if last_action_onnx.shape != (batch, len(self.contract.joint_names)):
            raise ValueError(
                "last_action_onnx shape differs from the ONNX action contract"
            )
        if (
            last_action_onnx.device != frame.joint_pos.device
            or last_action_onnx.dtype != frame.joint_pos.dtype
        ):
            raise ValueError("last_action_onnx device/dtype differs from robot state")

        ref_idx = _indices(
            reference.joint_names, self.contract.joint_names, label="reference"
        )
        robot_idx = _indices(
            frame.joint_names, self.contract.joint_names, label="mjlab state"
        )
        ref_joint = reference.joint_pos[..., ref_idx]
        robot_joint = frame.joint_pos[..., robot_idx]
        robot_vel = frame.joint_vel[..., robot_idx]

        # HoloMotion repeats the last real reference frame at the end of a clip.
        valid_count = reference.valid_mask.long().sum(dim=1).clamp_min(1)
        last_index = (valid_count - 1).clamp_max(context_frames - 1)
        gather_index = last_index[:, None, None]
        ref_joint_last = ref_joint.gather(
            1, gather_index.expand(-1, 1, ref_joint.shape[-1])
        )
        ref_root_pos_last = reference.root_pos_w.gather(
            1, gather_index.expand(-1, 1, 3)
        )
        ref_root_quat_last = reference.root_quat_wxyz.gather(
            1, gather_index.expand(-1, 1, 4)
        )
        mask = reference.valid_mask[..., None]
        ref_joint = torch.where(mask, ref_joint, ref_joint_last)
        ref_root_pos = torch.where(mask, reference.root_pos_w, ref_root_pos_last)
        ref_root_quat = torch.where(mask, reference.root_quat_wxyz, ref_root_quat_last)
        if previous_reference_root_pose is None:
            previous_root_pos = ref_root_pos[:, 0]
            previous_root_quat = ref_root_quat[:, 0]
        else:
            if previous_reference_root_pose.shape != (batch, 7):
                raise ValueError("previous_reference_root_pose must have shape [B,7]")
            if (
                previous_reference_root_pose.device != frame.joint_pos.device
                or previous_reference_root_pose.dtype != frame.joint_pos.dtype
            ):
                raise ValueError(
                    "previous reference root pose device/dtype differs from robot state"
                )
            if not torch.isfinite(previous_reference_root_pose).all().item():
                raise ValueError("previous reference root pose contains NaN/Inf")
            previous_root_pos = previous_reference_root_pose[:, :3]
            previous_root_quat = previous_reference_root_pose[:, 3:7]

        # The official actor derives reference velocities from one past pose
        # and one extra future pose. Stored motion velocity fields are unused.
        root_pos_context = torch.cat((previous_root_pos[:, None], ref_root_pos), dim=1)
        root_quat_context = torch.cat(
            (previous_root_quat[:, None], ref_root_quat), dim=1
        )
        sample = torch.arange(1, horizon + 1, device=frame.joint_pos.device)
        lo = sample - 1
        hi = sample + 1
        dt = frame.joint_pos.new_full((batch, horizon, 1), 0.04)
        ref_lin_w = (root_pos_context[:, hi] - root_pos_context[:, lo]) / dt
        ref_ang_w = _angular_velocity_wxyz(
            root_quat_context[:, lo], root_quat_context[:, hi], dt
        )
        ref_joint = ref_joint[:, :horizon]
        ref_root_pos = ref_root_pos[:, :horizon]
        ref_root_quat = ref_root_quat[:, :horizon]

        gravity_w = ref_joint.new_tensor((0.0, 0.0, -1.0)).expand(batch, horizon, 3)
        ref_gravity = _quat_apply_inverse(ref_root_quat, gravity_w)
        ref_lin_b = _quat_apply_inverse(ref_root_quat, ref_lin_w)
        ref_ang_b = _quat_apply_inverse(ref_root_quat, ref_ang_w)
        robot_yaw = _yaw(frame.root_quat_wxyz)
        ref_yaw = _yaw(ref_root_quat)
        current_yaw_error = ref_yaw[:, 0] - robot_yaw
        future_yaw_delta = ref_yaw[:, 1:] - ref_yaw[:, :1]
        relative_future = _quat_multiply(
            _quat_inverse(frame.root_quat_wxyz)[:, None, :],
            ref_root_quat[:, 1:],
        )
        future_rot6d = _matrix_from_quat(relative_future)[..., :2].reshape(
            batch, future, 6
        )
        default = self.contract.tensor("default_joint_pos", frame.joint_pos)

        current_terms = (
            ref_gravity[:, 0],
            ref_lin_b[:, 0],
            ref_ang_b[:, 0],
            ref_joint[:, 0],
            ref_root_pos[:, 0, 2:3],
            torch.stack(
                (torch.sin(current_yaw_error), torch.cos(current_yaw_error)),
                dim=-1,
            ),
            frame.projected_gravity_b,
            frame.root_ang_vel_b,
            robot_joint - default,
            robot_vel,
            last_action_onnx,
        )
        future_terms = (
            ref_joint[:, 1:],
            ref_root_pos[:, 1:, 2:3],
            ref_gravity[:, 1:],
            ref_lin_b[:, 1:],
            ref_ang_b[:, 1:],
            torch.stack(
                (torch.sin(future_yaw_delta), torch.cos(future_yaw_delta)),
                dim=-1,
            ),
            future_rot6d,
        )
        observation = torch.cat(
            (
                *current_terms,
                *(term.reshape(batch, -1) for term in future_terms),
            ),
            dim=-1,
        )
        if observation.shape != (batch, self.contract.obs_dim):
            raise RuntimeError(
                f"assembled HoloMotion observation has {observation.shape[-1]} values, "
                f"expected {self.contract.obs_dim}"
            )
        if not torch.isfinite(observation).all().item():
            raise RuntimeError("assembled HoloMotion observation contains NaN/Inf")
        return observation

    def onnx_action_to_policy(self, action_onnx: torch.Tensor) -> torch.Tensor:
        if action_onnx.ndim != 2 or action_onnx.shape[-1] != len(
            self.contract.joint_names
        ):
            raise ValueError("HoloMotion ONNX action must be [B,22]")
        indices = _indices(
            self.contract.joint_names,
            self.robot.joint_names,
            label="ONNX action",
        )
        return action_onnx[..., indices]


class HoloMotionK1MJLabAdapter:
    """Small inference boundary around :class:`K1MJLabBridge`."""

    def __init__(self, bridge, contract: HoloMotionOnnxContract) -> None:
        from .envs.mjlab_k1 import K1MJLabBridge

        if not isinstance(bridge, K1MJLabBridge):
            raise TypeError("bridge must be K1MJLabBridge")
        self.bridge = bridge
        self.contract = contract
        self.builder = HoloMotionActorObservationBuilder(contract)
        self.last_action_onnx = torch.zeros(
            bridge.num_envs,
            len(contract.joint_names),
            device=bridge.device,
            dtype=bridge.entity.data.joint_pos.dtype,
        )
        self.previous_reference_root_pose = torch.zeros(
            bridge.num_envs,
            7,
            device=bridge.device,
            dtype=bridge.entity.data.joint_pos.dtype,
        )
        self._previous_reference_valid = torch.zeros(
            bridge.num_envs, device=bridge.device, dtype=torch.bool
        )
        self._pending_reference_root_pose: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        ids = self.bridge._ids(env_ids)
        self.bridge.reset(env_ids)
        self.last_action_onnx[ids] = 0.0
        self.previous_reference_root_pose[ids] = 0.0
        self._previous_reference_valid[ids] = False
        self._pending_reference_root_pose = None

    def observation(self, reference: ReferenceWindow) -> torch.Tensor:
        data = self.bridge.entity.data
        frame = HoloMotionRobotFrame(
            joint_names=self.bridge.joint_names,
            joint_pos=data.joint_pos,
            joint_vel=data.joint_vel,
            root_quat_wxyz=data.root_link_quat_w,
            root_ang_vel_b=data.root_link_ang_vel_b,
            projected_gravity_b=data.projected_gravity_b,
        )
        current_reference_root_pose = torch.cat(
            (reference.root_pos_w[:, 0], reference.root_quat_wxyz[:, 0]),
            dim=-1,
        )
        previous_reference_root_pose = torch.where(
            self._previous_reference_valid[:, None],
            self.previous_reference_root_pose,
            current_reference_root_pose,
        )
        observation = self.builder(
            frame,
            reference,
            self.last_action_onnx,
            previous_reference_root_pose=previous_reference_root_pose,
        )
        self._pending_reference_root_pose = current_reference_root_pose.clone()
        return observation

    def step(self, action_onnx: torch.Tensor):
        if self._pending_reference_root_pose is None:
            raise RuntimeError("observation() must be called before step()")
        if (
            action_onnx.device != self.last_action_onnx.device
            or action_onnx.dtype != self.last_action_onnx.dtype
        ):
            raise ValueError("ONNX action device/dtype differs from mjlab")
        policy_action = self.builder.onnx_action_to_policy(action_onnx)
        transition = self.bridge.step(policy_action)
        self.last_action_onnx.copy_(action_onnx)
        self.previous_reference_root_pose.copy_(self._pending_reference_root_pose)
        self._previous_reference_valid.fill_(True)
        self._pending_reference_root_pose = None
        return transition
