"""第 15A 个文件：从 mjlab 读取真实物理反馈，交给独立动力学模型。

这个文件做三件事：
  1. 给原来的 K1 mjlab 环境加上脚部真实接触传感器。
  2. 把 mjlab 的身体、脚部和外部物体状态整理成 PhysicsObservation。
  3. 提供奖励模块需要的 TrackingState / BodyTrackingState。

非常重要：
  - 脚部接触来自 MuJoCo 接触传感器，不是根据参考动作伪造。
  - 初版只有左右脚与静止地面的接触槽位；它不是任意物体接触图。
  - 接触点是碰撞求解器报告的接触位置，不是假装精确 heel/toe site。
  - 额外物体的状态来自仿真真值，只适用于仿真训练；部署时须接感知/估计器。
  - 本文件不改变旧 mjlab_k1.py、PPO 或第 14 模块。
  - 不含自测、测试数据或自动训练入口，留给统一验收。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Callable

import torch

from .mjlab_k1 import K1MJLabBridge, make_k1_mjlab_cfg
from ..models.physics import PhysicsObservation
from ..rewards.motion_tracking import TrackingState, BodyTrackingState


FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTACT_SENSOR_NAME = "k1_feet_terrain_contact"


def _contact_primary_names(sensor) -> tuple[str, ...]:
    """读取接触传感器解析后的 primary 顺序，兼容 mjlab 1.3/1.6。"""
    names = getattr(sensor, "primary_names", None)
    if names is not None:
        return tuple(names)
    # mjlab 1.3 尚未公开 primary_names，但初始化后的 slot 保留同一顺序；
    # 每个 primary 会为每个 field 重复一次，因此需要稳定去重。
    slots = getattr(sensor, "_slots", None)
    if slots is None:
        raise RuntimeError("当前 mjlab 接触传感器没有可读取的 primary 名称")
    ordered = tuple(dict.fromkeys(getattr(slot, "primary_name", None) for slot in slots))
    if not ordered or any(name is None for name in ordered):
        raise RuntimeError("mjlab 接触传感器的 primary 槽位无效")
    return ordered


@dataclass(frozen=True)
class ObjectObservationSpec:
    """描述一个需要被 Actor 观察的物体，不承担物体识别。

    输入：
        entity_name：mjlab 场景中注册的物体名字，如 'skateboard'。
        size_m：这个物体的长、宽、高，单位米；来自明确的物体配置。
    输出：创建后，K1PhysicsBridge 会在每帧读取这个物体的根位姿和速度。

    注意：这里读取仿真真值，是研究阶段的理想传感器；不能拿来证明可实机部署。
    """

    entity_name: str
    size_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        """功能：检查物体名字与几何尺寸，防止无效配置混入训练。

        输入：对象自身的 entity_name、size_m。
        输出：无；非法时抛 ValueError。
        """
        if not isinstance(self.entity_name, str) or not self.entity_name.strip():
            raise ValueError("物体 entity_name 不能为空")
        if (len(self.size_m) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (float, int))
                or not math.isfinite(v) or v <= 0 for v in self.size_m)):
            raise ValueError("size_m 必须是3个正的有限尺寸（米）")


@dataclass(frozen=True)
class PrivilegedFootContactTruth:
    """Simulator-only foot/terrain contact truth.

    None of these tensors may be routed into a deployable Actor.  They are
    intended for Teacher/Critic inputs, reward terms, and supervised targets.
    The foot axis is fixed as ``(left, right)``.
    """

    point_position_b: torch.Tensor
    point_velocity_b: torch.Tensor
    normal_b: torch.Tensor
    normal_force_n: torch.Tensor
    contact_found: torch.Tensor
    tangential_force_n: torch.Tensor
    slip_speed_m_s: torch.Tensor
    friction_utilization: torch.Tensor

    @property
    def teacher_dim(self) -> int:
        """Per-foot truth width (14) times the fixed left/right slots."""
        return 28

    def teacher_tensor(self) -> torch.Tensor:
        """Flatten exact left/right truth without mixing their identities."""
        values = torch.cat(
            (
                self.point_position_b,
                self.point_velocity_b,
                self.normal_b,
                self.normal_force_n.unsqueeze(-1) / 100.0,
                self.contact_found.unsqueeze(-1).to(self.point_position_b.dtype),
                self.tangential_force_n.unsqueeze(-1) / 100.0,
                self.slip_speed_m_s.unsqueeze(-1),
                self.friction_utilization.clamp_max(5.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        result = values.flatten(-2)
        if result.shape[-1] != self.teacher_dim or not torch.isfinite(result).all().item():
            raise ValueError("privileged foot Teacher tensor is invalid")
        return result


def make_k1_physics_cfg(asset_root: str, *, num_envs: int = 1,
                        physics_dt: float = 0.002, decimation: int = 10):
    """功能：在第11模块的 K1 环境配置上，增加真正的脚地接触测量。

    输入：
        asset_root：官方下载的 booster_assets 所在目录。
        num_envs：并行 K1 的数量 B。
        physics_dt、decimation：物理步长与每个策略动作执行的物理步数。
    输出：mjlab 的 ManagerBasedRlEnvCfg 配置对象。还没有启动仿真。

    后续：ManagerBasedRlEnv(cfg, device='cuda:0') 才会创建物理环境。
    这里不修改机器人碰撞结构，也不改 reward 配置。
    """
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    cfg = make_k1_mjlab_cfg(
        asset_root, num_envs=num_envs,
        physics_dt=physics_dt, decimation=decimation,
    )
    if any(sensor.name == CONTACT_SENSOR_NAME for sensor in cfg.scene.sensors):
        raise ValueError("接触传感器名字与现有传感器重复")
    # 只跟踪左右脚和静止地面 terrain 之间的真实接触。
    # maxforce 每只脚保留最大受力的那个接触点，pos/normal 才有清晰含义。
    # global_frame=True 让 force 变成世界坐标；必须同时请求 tangent。
    contact = ContactSensorCfg(
        name=CONTACT_SENSOR_NAME,
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce", global_frame=True,
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (contact,)
    return cfg


def _world_to_body(quat_wxyz: torch.Tensor, vector_w: torch.Tensor) -> torch.Tensor:
    """功能：把世界坐标中的速度/方向变成机器人身体坐标中的速度/方向。

    输入：
        quat_wxyz：[B,4]，根部件朝向（身体坐标旋转到世界坐标）。
        vector_w：[B,3] 或 [B,C,3]，世界坐标向量。
    输出：与 vector_w 一样形状的身体坐标向量。

    用途：让模型在机器人转向后，仍用统一的前、左、上方向理解物理反馈。
    """
    from ..reference_observations import _quat_to_matrix

    matrix = _quat_to_matrix(quat_wxyz).transpose(-1, -2)
    if vector_w.ndim == 3:
        matrix = matrix.unsqueeze(1)
    return (matrix @ vector_w.unsqueeze(-1)).squeeze(-1)


def _quat_relative(base_wxyz: torch.Tensor, target_wxyz: torch.Tensor) -> torch.Tensor:
    """功能：求物体相对于机器人的朝向，支持同批 B 个环境。

    输入：base_wxyz、target_wxyz 都为 [B,4]，顺序 w,x,y,z。
    输出：[B,4]，物体相对根部件的朝向四元数。
    """
    bw, bx, by, bz = base_wxyz.unbind(-1)
    tw, tx, ty, tz = target_wxyz.unbind(-1)
    # 根部件四元数取逆：单位四元数只需反转虚部符号。
    x, y, z = -bx, -by, -bz
    quat = torch.stack((
        bw * tw - x * tx - y * ty - z * tz,
        bw * tx + x * tw + y * tz - z * ty,
        bw * ty - x * tz + y * tw + z * tx,
        bw * tz + x * ty - y * tx + z * tw,
    ), dim=-1)
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)


class K1PhysicsBridge:
    """把原版 K1MJLabBridge 扩展成物理反馈适配器，不重写它的动作控制。

    输入：
        bridge：已经创建的 K1MJLabBridge，里面保存了 mjlab 环境。
        object_specs：可选物体配置，按给定顺序形成稳定的物体槽位。
        ground_velocity_fn：可选的世界坐标地面速度估计函数。
            输入 [B,2,3] 的接触点世界位置，输出 [B,2,3] 的地面速度。
            None 表示本模块配置的静止平面地面，地面速度为零。
    输出：
        read() → PhysicsObservation，供 DynamicsModel 和 PhysicsMLPActorCritic 使用。
        tracking_state() → TrackingState，供第13模块计算基本跟踪奖励。
    """

    def __init__(self, bridge: K1MJLabBridge, *,
                 object_specs: tuple[ObjectObservationSpec, ...] = (),
                 ground_velocity_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 check_values: bool = True) -> None:
        """功能：一次性检查接触传感器、脚部部件和物体名字是否存在。

        输入：bridge、object_specs、ground_velocity_fn、check_values。
        输出：无；保存固定索引，后续每帧不用查找名字。
        """
        if not isinstance(bridge, K1MJLabBridge):
            raise TypeError("bridge 必须是 K1MJLabBridge")
        if bridge.robot_id != "k1_22dof":
            raise ValueError("K1PhysicsBridge 仅负责 K1；G1 需单独环境适配")
        self.bridge = bridge
        self.entity = bridge.entity
        self.device = bridge.device
        self.num_envs = bridge.num_envs
        self.check_values = check_values
        self.ground_velocity_fn = ground_velocity_fn
        self.object_specs = tuple(object_specs)
        if type(check_values) is not bool:
            raise TypeError("check_values 必须是 bool")
        if ground_velocity_fn is not None and not callable(ground_velocity_fn):
            raise TypeError("ground_velocity_fn 必须是可调用函数或 None")
        if any(not isinstance(item, ObjectObservationSpec) for item in self.object_specs):
            raise TypeError("object_specs 元素必须是 ObjectObservationSpec")
        names = [item.entity_name for item in self.object_specs]
        if len(set(names)) != len(names) or "robot" in names or "terrain" in names:
            raise ValueError("物体名称不能重复、也不能把 robot/terrain 当外部物体")
        for item in self.object_specs:
            if not isinstance(item, ObjectObservationSpec):
                raise TypeError("object_specs 元素必须是 ObjectObservationSpec")
            if item.entity_name not in bridge.env.scene.entities:
                raise ValueError(f"mjlab 场景里没有外部物体 {item.entity_name!r}")
        for name in FOOT_NAMES:
            if name not in self.entity.body_names:
                raise ValueError(f"K1 缺少脚部 body {name!r}")
        self.foot_body_ids = [self.entity.body_names.index(name) for name in FOOT_NAMES]
        sensor = bridge.env.scene[CONTACT_SENSOR_NAME]
        primary = list(_contact_primary_names(sensor))
        indices = []
        for name in FOOT_NAMES:
            matches = [i for i, actual in enumerate(primary)
                       if actual == name or actual.endswith("/" + name)]
            if len(matches) != 1:
                raise ValueError(f"接触传感器没有唯一的 {name}: 实际 {primary}")
            indices.append(matches[0])
        # 不能把接触坐标系的 force 当世界坐标，也不能把多接触槽当两只脚。
        cfg = getattr(sensor, "cfg", None)
        if cfg is not None and (cfg.reduce != "maxforce" or cfg.num_slots != 1
                                or not cfg.global_frame):
            raise ValueError("物理桥要求 maxforce、num_slots=1、global_frame=True 的传感器")
        self.sensor = sensor
        self.sensor_indices = indices
        self._contact_supervision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def contact_supervision(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return training-only tangential force, slip speed and friction utilization.

        ``read()`` must be called for the current simulator state first.  These tensors are
        deliberately kept outside PhysicsObservation so a deployable Actor/Dynamics input cannot
        accidentally consume simulator-only force truth.
        """
        if self._contact_supervision is None:
            raise RuntimeError("读取接触监督前必须先读取当前物理观测")
        return tuple(value.clone() for value in self._contact_supervision)

    @torch.no_grad()
    def read_deployable(
        self, contact_estimate: torch.Tensor | None = None
    ) -> PhysicsObservation:
        """Build a student observation without reading simulator contact truth.

        Foot position/velocity are link-kinematic quantities derivable from
        encoders and the robot model.  Exact collision point, surface normal,
        contact force, slip and root linear velocity are deliberately absent.
        ``contact_estimate`` may only be supplied by an explicit deployable
        estimator/sensor adapter; it is never copied from the mjlab sensor here.
        """
        body_obs = self.bridge.observation().clone()
        data = self.entity.data
        root_pos_w = data.root_link_pos_w
        root_quat_w = data.root_link_quat_w
        root_lin_vel_w = data.root_link_vel_w[:, :3]
        foot_pose = data.body_link_pose_w[:, self.foot_body_ids]
        foot_velocity = data.body_link_vel_w[:, self.foot_body_ids, :3]
        foot_position_b = _world_to_body(
            root_quat_w, foot_pose[..., :3] - root_pos_w[:, None]
        )
        # Removing root translation leaves a link-kinematic velocity that can
        # be reconstructed from proprioception; it is not ground slip truth.
        foot_velocity_b = _world_to_body(
            root_quat_w, foot_velocity - root_lin_vel_w[:, None]
        )
        contacts = body_obs.new_zeros(self.num_envs, len(FOOT_NAMES), 12)
        contacts[..., :3] = foot_position_b
        contacts[..., 3:6] = foot_velocity_b
        if contact_estimate is not None:
            if (
                not isinstance(contact_estimate, torch.Tensor)
                or contact_estimate.shape != (self.num_envs, len(FOOT_NAMES))
                or contact_estimate.device != self.device
                or contact_estimate.dtype not in (torch.bool, body_obs.dtype)
            ):
                raise ValueError("contact_estimate must be deployable [B,2] bool/float")
            estimate = contact_estimate.to(body_obs.dtype)
            if ((estimate < 0) | (estimate > 1)).any().item():
                raise ValueError("contact_estimate must be in [0,1]")
            contacts[..., 10] = estimate
        contact_mask = torch.ones(
            self.num_envs, len(FOOT_NAMES), device=self.device, dtype=torch.bool
        )
        empty_objects = body_obs.new_zeros(self.num_envs, 0, 16)
        empty_mask = torch.zeros(self.num_envs, 0, device=self.device, dtype=torch.bool)
        core_valid = torch.ones(self.num_envs, 9 + 2*((body_obs.shape[-1]-6)//3),
                                device=self.device, dtype=torch.bool)
        core_valid[:, :3] = False
        return PhysicsObservation(
            body_obs,
            body_obs.new_zeros(self.num_envs, 3),
            contacts,
            contact_mask,
            empty_objects,
            empty_mask,
            core_valid,
        )

    @torch.no_grad()
    def privileged_contact_truth(self) -> PrivilegedFootContactTruth:
        """Read exact mjlab contacts for non-Actor training channels only."""
        _, truth = self.privileged_read()
        return truth

    @torch.no_grad()
    def privileged_read(self) -> tuple[PhysicsObservation, PrivilegedFootContactTruth]:
        """Atomically read one physical frame and its richer contact truth."""
        exact = self.read()
        tangential, slip, utilization = self.contact_supervision()
        truth = PrivilegedFootContactTruth(
            point_position_b=exact.contact_features[..., :3].clone(),
            point_velocity_b=exact.contact_features[..., 3:6].clone(),
            normal_b=exact.contact_features[..., 6:9].clone(),
            normal_force_n=exact.contact_features[..., 9].clone(),
            contact_found=exact.contact_features[..., 10].bool().clone(),
            tangential_force_n=tangential,
            slip_speed_m_s=slip,
            friction_utilization=utilization,
        )
        return exact, truth

    @torch.no_grad()
    def read(self) -> PhysicsObservation:
        """功能：读取一帧真实仿真状态，并生成第14模块的标准物理观测。

        输入：不需要参数；会读取 bridge 中 mjlab 的当前状态。
        输出：PhysicsObservation：
            body_obs：[B,72]，原版 K1 身体观测（策略关节顺序）。
            base_lin_vel_b：[B,3]，身体坐标的根部件线速度。
            contact_features：[B,2,12]，左/右脚各一个真实接触槽位。
            contact_mask：[B,2]，两个脚部槽位一直存在（接触与否看第10列）。
            object_features：[B,M,16]，M 个已登记物体的相对位姿与速度。
            object_mask：[B,M]，已登记且被观察到的物体槽位。

        使用时机：bridge.reset() 后、或 bridge.step() 后且 reset 前读取。
        本函数复制快照，不会因为之后 reset() 而修改以前保存的 rollout 数据。
        """
        body_obs = self.bridge.observation().clone()
        data = self.entity.data
        root_pos_w = data.root_link_pos_w
        root_quat_w = data.root_link_quat_w
        root_vel_w = data.root_link_vel_w
        base_lin_vel_b = _world_to_body(root_quat_w, root_vel_w[:, :3])

        # 接触传感器报告与地面的真实碰撞，而不是从参考轨迹判断是否接触。
        c = self.sensor.data
        if any(getattr(c, key) is None for key in ("found", "force", "pos", "normal")):
            raise RuntimeError("接触传感器缺少 found/force/pos/normal 字段")
        found = c.found[:, self.sensor_indices] > 0
        raw_pos = c.pos[:, self.sensor_indices]
        raw_force = c.force[:, self.sensor_indices]
        raw_normal = c.normal[:, self.sensor_indices]
        foot_pose = data.body_link_pose_w[:, self.foot_body_ids]
        foot_velocity = data.body_link_vel_w[:, self.foot_body_ids]
        # 非接触时，用脚部部件位置作为“测量槽位”的坐标，不假装它是接触点。
        point_pos = torch.where(found.unsqueeze(-1), raw_pos, foot_pose[..., :3])
        # 刚体上任意点的速度 = 部件原点线速度 + 角速度 × 相对部件原点位置。
        point_vel = foot_velocity[..., :3] + torch.cross(
            foot_velocity[..., 3:], point_pos - foot_pose[..., :3], dim=-1,
        )
        if self.ground_velocity_fn is None:
            ground_vel_w = torch.zeros_like(point_vel)
        else:
            ground_vel_w = self.ground_velocity_fn(point_pos)
            if not isinstance(ground_vel_w, torch.Tensor) or ground_vel_w.shape != point_vel.shape:
                raise ValueError("ground_velocity_fn 必须返回 [B,2,3] 张量")
            if ground_vel_w.device != self.device or ground_vel_w.dtype != point_vel.dtype:
                raise ValueError("ground_velocity_fn 输出设备/类型错误")
        relative_velocity_b = _world_to_body(root_quat_w, point_vel - ground_vel_w)
        relative_pos_b = _world_to_body(root_quat_w, point_pos - root_pos_w[:, None])
        # normal 方向由传感器确定，只关心法向力大小；切向摩擦力不冒充法向力。
        normal_w = torch.where(found.unsqueeze(-1), raw_normal, 0.0)
        normal_len = torch.linalg.vector_norm(normal_w, dim=-1, keepdim=True)
        normal_unit = normal_w / normal_len.clamp_min(1e-8)
        signed_normal_force = (raw_force * normal_unit).sum(-1)
        normal_force = signed_normal_force.abs()
        normal_force = torch.where(found, normal_force, 0.0)
        tangential_force_w = raw_force - signed_normal_force.unsqueeze(-1) * normal_unit
        tangential_force = torch.linalg.vector_norm(tangential_force_w, dim=-1)
        relative_velocity_w = point_vel - ground_vel_w
        tangential_velocity_w = relative_velocity_w - (
            relative_velocity_w * normal_unit).sum(-1, keepdim=True) * normal_unit
        slip_speed = torch.linalg.vector_norm(tangential_velocity_w, dim=-1)
        tangential_force = torch.where(found, tangential_force, 0.0)
        slip_speed = torch.where(found, slip_speed, 0.0)
        # Static contact only supplies a lower bound on mu; at Coulomb slip this ratio approaches mu.
        utilization = torch.where(found, tangential_force / normal_force.clamp_min(1.0), 0.0)
        self._contact_supervision = (
            tangential_force.clone(), slip_speed.clone(), utilization.clone())
        normal_b = _world_to_body(root_quat_w, normal_unit)
        contact_features = torch.cat((
            relative_pos_b,
            relative_velocity_b,
            normal_b,
            normal_force.unsqueeze(-1),
            found.unsqueeze(-1).to(body_obs.dtype),
            torch.ones_like(found.unsqueeze(-1), dtype=body_obs.dtype),
            # 传感器始终可用；没有接触时真实接触力为0，不等于缺失传感器。
        ), dim=-1)
        contact_mask = torch.ones(self.num_envs, len(FOOT_NAMES),
                                  device=self.device, dtype=torch.bool)

        object_rows = []
        for item in self.object_specs:
            object_data = self.bridge.env.scene[item.entity_name].data
            object_pose = object_data.root_link_pose_w
            object_vel = object_data.root_link_vel_w
            relative_pos = _world_to_body(root_quat_w, object_pose[:, :3] - root_pos_w)
            relative_quat = _quat_relative(root_quat_w, object_pose[:, 3:7])
            relative_lin = _world_to_body(root_quat_w, object_vel[:, :3] - root_vel_w[:, :3])
            relative_ang = _world_to_body(root_quat_w, object_vel[:, 3:] - root_vel_w[:, 3:])
            size = body_obs.new_tensor(item.size_m).expand(self.num_envs, 3)
            object_rows.append(torch.cat((relative_pos, relative_quat,
                                          relative_lin, relative_ang, size), dim=-1))
        if object_rows:
            object_features = torch.stack(object_rows, dim=1)
        else:
            object_features = body_obs.new_zeros(self.num_envs, 0, 16)
        object_mask = torch.ones(self.num_envs, len(object_rows),
                                 device=self.device, dtype=torch.bool)
        output = PhysicsObservation(body_obs, base_lin_vel_b.clone(),
                                    contact_features.clone(), contact_mask,
                                    object_features.clone(), object_mask)
        if self.check_values:
            for name, value in (("body_obs", output.body_obs),
                                ("base_lin_vel_b", output.base_lin_vel_b),
                                ("contact_features", output.contact_features),
                                ("object_features", output.object_features)):
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"当前 mjlab 物理观测 {name} 含 NaN/Inf")
        return output

    @torch.no_grad()
    def tracking_state(self) -> TrackingState:
        """功能：读取第13模块奖励使用的真实关节与根部件状态。

        输入：无；直接读取 bridge.entity.data。
        输出：TrackingState，其中 joint_pos [B,N]，root_pos_w [B,3] 等。
        用途：与 MotionSampler 的当前参考比较，计算跟踪奖励。
        """
        data = self.entity.data
        vel = data.root_link_vel_w
        return TrackingState(
            joint_names=tuple(self.entity.joint_names),
            joint_pos=data.joint_pos.clone(),
            joint_vel=data.joint_vel.clone(),
            root_pos_w=data.root_link_pos_w.clone(),
            root_quat_wxyz=data.root_link_quat_w.clone(),
            root_lin_vel_w=vel[:, :3].clone(),
            root_ang_vel_w=vel[:, 3:].clone(),
        )

    @torch.no_grad()
    def body_tracking_state(self) -> BodyTrackingState:
        """功能：读取全身各个真实 link 的位置、朝向和线速度。

        输入：无；使用 mjlab 当前的所有 body_link 状态。
        输出：BodyTrackingState，body_names 对应 [B,M,3/4] 的第 M 轴。
        注意：body 原点不是鞋底 site，不能拿它冒充足底接触位置。
        """
        data = self.entity.data
        pose = data.body_link_pose_w
        vel = data.body_link_vel_w
        return BodyTrackingState(
            body_names=tuple(self.entity.body_names),
            pos_w=pose[..., :3].clone(),
            quat_wxyz=pose[..., 3:7].clone(),
            lin_vel_w=vel[..., :3].clone(),
        )
