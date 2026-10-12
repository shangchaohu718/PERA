"""第 13 个模块：给机器人“模仿参考动作”打分。

这个文件解决什么问题？
    第 12 个模块告诉我们：机器人此时应该处于什么姿态。
    mjlab 告诉我们：机器人现在真的处于什么姿态。
    本文件比较两者，计算每台机器人的奖励 [B]，交给后续 PPO 采样器。

与前面代码的关系：
    MotionSampler.reference_window()  → ReferenceWindow（目标关节/根部件）
    MotionSampler.body_reference_window() → BodyReferenceWindow（可选的身体目标）
    mjlab 的 EntityData → TrackingState / BodyTrackingState（真实状态）
    已单独准备的鞋底 site 数据 → FootSiteState / FootSiteTarget（可选的脚部目标）
                             ↓
                MotionTrackingReward.forward()
                             ↓
                RewardResult.total [B] → 后续 PPO collector

几个必须遵守的约定：
    B = 并行机器人数量，N = 关节数，H = 参考时间点数。
    M = 身体部件数量，S = 脚底测量点数量（例如左右 heel/toe）。
    坐标全部是世界坐标 w，位置米、角度弧度、速度米/秒或弧度/秒。
    四元数一律 wxyz，表示部件坐标系旋转到世界坐标系。
    本文件只评价“当前时刻”，必须读取参考窗口的第 0 个点，且它的时间偏移为 0。
    不自动推断接触、锚脚、滑脚；这些标签必须由数据准备环节明确给出。
    不把 ankle link 的中心位置冒充鞋底接触点；两种数据各有独立接口。
    此文件不修改 mjlab 内部的 reward manager，也不执行 PPO 更新。
    这是有配置参数的奖励基线，权重并未针对 K1/G1 做过实际训练标定。

生产逻辑不含自测，统一回归位于 tests/。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
import math
from typing import Sequence

import torch
from torch import nn

from ..data.motion_sampler import BodyReferenceWindow
from ..reference_observations import ReferenceWindow
from ..robot_contract import RobotContract


class FootPhase(IntEnum):
    """脚底点的动作类别，由参考数据的制作人标注，不能让奖励函数猜。

    UNKNOWN(0)：未标注。不参与接触/锚定/滑动专项奖励。
    ANCHOR(1)：参考要求这个脚底点支撑地面，并尽量保持不滑。
    SLIDE(2)：参考要求这个脚底点保持接触，同时在地面上滑动。
    SWING(3)：参考要求这个点离地摆动。
    TRANSITION(4)：交接期。不强加接触或锁脚条件。

    输入位置：FootSiteTarget.phase，[B,S]，必须是整型张量。
    输出：这些是整数标签，不是神经网络预测的概率。
    """
    UNKNOWN = 0
    ANCHOR = 1
    SLIDE = 2
    SWING = 3
    TRANSITION = 4


@dataclass(frozen=True)
class TrackingState:
    """一帧真实机器人状态：可从 mjlab 的 EntityData 读取。

    输入：
        joint_names：joint_pos/joint_vel 的列顺序，可以与策略顺序不同。
        joint_pos / joint_vel：[B,N]，真实关节角(rad) / 角速度(rad/s)。
        root_pos_w：[B,3]，实际根部件世界位置(m)。
        root_quat_wxyz：[B,4]，实际根部件朝向(wxyz)。
        root_lin_vel_w：[B,3]，实际根部件世界线速度(m/s)。
        root_ang_vel_w：[B,3]，实际根部件世界角速度(rad/s)。

    输出：只是记录数据的对象。forward() 会将关节按名字转换到策略顺序。
    注意：这些是仿真结束后的真实状态，不是 Actor 对未来的预测。
    """
    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_wxyz: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor


@dataclass(frozen=True)
class BodyTrackingState:
    """一帧真实的多个身体部件，可直接对应第 12 个模块的 body 参考。

    输入：
        body_names：每列属于哪个部件，比如 trunk、left_ankle_roll_link。
        pos_w：[B,M,3]，真实部件中心/链接原点的世界位置。
        quat_wxyz：[B,M,4]，每个部件在世界坐标中的朝向。
        lin_vel_w：[B,M,3]，每个部件原点的世界速度。

    输出：无。这些值供身体位置、朝向、速度跟踪奖励使用。
    注意：部件原点不一定是脚底。不能把它直接当作鞋底滑移测量点。
    """
    body_names: tuple[str, ...]
    pos_w: torch.Tensor
    quat_wxyz: torch.Tensor
    lin_vel_w: torch.Tensor


@dataclass(frozen=True)
class FootSiteState:
    """脚底测量点的真实物理状态，用来检查脚有没有按要求移动。

    输入：
        site_names：真实 site 名称，如 left_sole、right_sole，名字不能重复。
        pos_w：[B,S,3]，脚底点真实世界位置(m)，不是 ankle link 的位置。
        vel_w：[B,S,3]，可选；这些脚底点原点的真实世界速度(m/s)。
        quat_wxyz：[B,S,4]，可选；脚底测量坐标系的真实朝向。
        contact：[B,S]，可选 bool；物理传感器实测是否接触地面。
        prev_pos_w：[B,S,3]，可选；上一控制步相同测量点的位置。
            必须与本帧属于同一回合；机器人重置时要清除旧帧记录。
        dt_s：可选；两帧之间的真实控制周期(s)，例如 0.02。

    输出：无。forward() 会按需要读取这些物理量。
    注意：如未提供 vel_w，可以同时提供 prev_pos_w 和 dt_s 估算位移速度；
          不能用脚踝关节速度代替鞋底点的世界速度。
    """
    site_names: tuple[str, ...]
    pos_w: torch.Tensor
    vel_w: torch.Tensor | None = None
    quat_wxyz: torch.Tensor | None = None
    contact: torch.Tensor | None = None
    prev_pos_w: torch.Tensor | None = None
    dt_s: float | None = None


@dataclass(frozen=True)
class FootSiteTarget:
    """脚底测量点对应的明确参考目标，不能从身体中心位置偷换得到。

    输入：
        site_names：参考点名字，必须与 FootSiteState 的点一一对应；顺序可以不同。
        pos_w：[B,S,3]，已根据参考动作 FK 算出的鞋底点世界目标位置。
        vel_w：[B,S,3]，可选；同一鞋底点的世界目标速度。
        quat_wxyz：[B,S,4]，可选；参考鞋底点朝向。
        phase：[B,S]，可选，整数 0~4；锚定、滑动、摆动、过渡的**标注**。

    输出：无。这份数据是“参考目标”，不是脚的真实接触状态。
    重要：MotionSampler.body_reference_window() 目前提供的是身体部件轨迹；
          若要生成真正的 sole/toe/heel site 目标，必须使用同一个机器人模型
          对参考关节姿态进行 FK，并显式保存/计算 site 位姿与相位标签。
    """
    site_names: tuple[str, ...]
    pos_w: torch.Tensor
    vel_w: torch.Tensor | None = None
    quat_wxyz: torch.Tensor | None = None
    phase: torch.Tensor | None = None


@dataclass(frozen=True)
class MotionTrackingRewardConfig:
    """奖励权重与误差容忍度。权重越大，该项对总奖励的影响越大。

    输入：
        *_weight：各项奖励权重，必须 >=0；0 代表关闭该项。
        *_sigma_*：容忍度。实际误差接近 sigma 时，这项得分约为 exp(-1)。
        数值是便于启动研究的示例，不是 K1/G1 已标定的最佳值。

    输出：配置对象，供 MotionTrackingReward 使用。

    默认仅开启根部件和关节跟踪：
      - 有 body_* 参考时，可以打开 body_*_weight。
      - 有“同一鞋底 site”的参考和真实物理数据时，才能打开脚部专项权重。
      - 没有 phase 标签时，不能开启接触匹配、锚脚锁定或滑动速度奖励。
    """
    joint_pos_weight: float = 1.0
    joint_vel_weight: float = 0.2
    root_xy_weight: float = 1.0
    root_z_weight: float = 0.5
    root_rot_weight: float = 0.5
    root_lin_vel_weight: float = 0.1
    root_ang_vel_weight: float = 0.1
    body_pos_weight: float = 0.0
    body_rot_weight: float = 0.0
    body_lin_vel_weight: float = 0.0
    foot_xy_weight: float = 0.0
    foot_height_weight: float = 0.0
    foot_rot_weight: float = 0.0
    foot_slide_vel_weight: float = 0.0
    foot_anchor_still_weight: float = 0.0
    foot_contact_weight: float = 0.0

    joint_pos_sigma_rad: float = 0.25
    joint_vel_sigma_rad_s: float = 2.0
    root_xy_sigma_m: float = 0.20
    root_z_sigma_m: float = 0.10
    root_rot_sigma_rad: float = 0.35
    root_lin_vel_sigma_m_s: float = 0.70
    root_ang_vel_sigma_rad_s: float = 1.5
    body_pos_sigma_m: float = 0.20
    body_rot_sigma_rad: float = 0.40
    body_lin_vel_sigma_m_s: float = 0.80
    foot_xy_sigma_m: float = 0.08
    foot_height_sigma_m: float = 0.04
    foot_rot_sigma_rad: float = 0.35
    foot_slide_vel_sigma_m_s: float = 0.15
    foot_anchor_still_sigma_m_s: float = 0.02

    def __post_init__(self) -> None:
        """功能：配置创建时拒绝负奖励权重、负容忍度和 NaN/Inf。

        输入：所有 *_weight 和 *_sigma_* 配置项。
        输出：无；错误配置直接抛出异常，避免训练时才发现。
        """
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{item.name} 必须是实数")
            if not math.isfinite(float(value)):
                raise ValueError(f"{item.name} 不能是 NaN/Inf")
            if item.name.endswith("_weight") and value < 0:
                raise ValueError(f"{item.name} 奖励权重不能小于 0")
            if "_sigma_" in item.name and value <= 0:
                raise ValueError(f"{item.name} 容忍度必须大于 0")


@dataclass(frozen=True)
class MotionTrackingRewardResult:
    """一次奖励计算返回的东西。

    输出：
        total：[B]，每个环境最终得到的奖励，用于 PPO rollout。
        terms：字典，每项都是 [B]；存已经乘上权重的实际奖励贡献。
        metrics：字典，大多数值为 [B]；名称带 by_site 的值为 [B,S]，
                 会保留左右脚/heel/toe 的单独误差，不用平均值掩盖单脚失败。
        foot_site_names：metrics 的第 2 维 S 对应哪个脚底点；没有脚底数据时为空。
    注意：没有数据的可选奖励项不会出现在 terms 中，不用假的 0 分掩盖缺失。
    """
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]
    foot_site_names: tuple[str, ...] = ()


# ------------------------ 一、简单数学与输入检查 ------------------------


def _valid_names(names: Sequence[str], label: str) -> tuple[str, ...]:
    """功能：检查一组部件/脚底点名字是否重复或为空。

    输入：names 名字序列；label 是报错时显示的名称。
    输出：一个不重复、非空的 tuple；不合法则抛异常。
    """
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{label} 不能直接传一个字符串")
    names = tuple(names)
    if not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError(f"{label} 必须包含非空的名字")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} 中有重复名字")
    return names


def _quaternion_error_rad(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """功能：计算实际朝向与目标朝向相差多少弧度。

    输入：actual 和 target，形状相同，最后一维是 [w,x,y,z]。
    输出：去掉最后一维后的角度张量，范围 0~pi。
    原因：q 与 -q 表示同一个方向，所以要用四元数点积的绝对值。
    """
    # atan2 比直接 acos 更容易在“几乎一样”的小角度附近保持数值稳定。
    actual = actual / torch.linalg.vector_norm(actual, dim=-1, keepdim=True).clamp_min(1e-12)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-12)
    cosine = (actual * target).sum(-1).abs().clamp(0.0, 1.0)
    sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
    return 2.0 * torch.atan2(sine, cosine)


def _score(error_sq: torch.Tensor, sigma: float) -> torch.Tensor:
    """功能：把“平均平方误差”转换为越接近 1 越好的奖励。

    输入：error_sq 为 [B]，单位是误差单位的平方；sigma 是相同单位的容忍度。
    输出：[B]，每项属于 (0,1]，误差越小越接近 1。
    公式：exp(-error_sq / sigma²)；绝不通过取负奖励伪造好行为。
    """
    return torch.exp(-error_sq / (sigma * sigma))


def _mean_over_items(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """功能：只对满足条件的脚底点求平均，不让未标注的点稀释误差。

    输入：value [B,S]，每个脚底点的平方误差；mask [B,S]，True 表示计入。
    输出：mean [B] 平均平方误差；present [B] 表示当前环境有没有任何匹配点。
    注意：没有匹配点时均值设为 0，但后续奖励也必须设为 0，不能奖满分。
    """
    count = mask.sum(dim=-1)
    mean = (value * mask.to(value.dtype)).sum(dim=-1) / count.clamp_min(1)
    return mean, count > 0


def _masked_score(value: torch.Tensor, mask: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """功能：把带脚部阶段筛选条件的误差变成奖励。

    输入：value [B,S]；mask [B,S]；sigma 为容忍度。
    输出：reward [B]、present [B]。若本帧没有该阶段，奖励严格为 0。
    """
    mean, present = _mean_over_items(value, mask)
    return torch.where(present, _score(mean, sigma), torch.zeros_like(mean)), present


# ------------------------ 二、计算所有奖励 ------------------------


class MotionTrackingReward(nn.Module):
    """把当前真实状态和同一时刻的目标姿态放在一起打分。

    输入：
        robot：RobotContract，指出需要跟踪的是 K1 还是某个 G1 配置。
        cfg：奖励权重和尺度，不提供就使用仅关节/根跟踪的基础配置。
        check_values：是否逐项验证 NaN/Inf 和单位四元数；GPU 会有额外开销。
    输出：一个可调用模块。调用 forward(...) 返回 MotionTrackingRewardResult。

    特别注意：
        这个类本身没有学习参数，不通过反向传播训练奖励。
        它只负责给本步“已经执行出来的真实物理结果”打分。
        不保存滑步进度；完整 slide 位移、左右交替、全片节奏要由评估器另外统计。
    """

    def __init__(self, robot: RobotContract,
                 cfg: MotionTrackingRewardConfig | None = None,
                 *, check_values: bool = True) -> None:
        """功能：保存机器人合同和配置，准备计算奖励。

        输入：robot 指定关节名单；cfg 指定权重；check_values 控制严格数值检查。
        输出：无；创建的对象可在后续每个控制步重复使用。
        """
        super().__init__()
        if not isinstance(robot, RobotContract):
            raise TypeError("robot 必须是 RobotContract")
        if cfg is None:
            cfg = MotionTrackingRewardConfig()
        if not isinstance(cfg, MotionTrackingRewardConfig):
            raise TypeError("cfg 必须是 MotionTrackingRewardConfig")
        if type(check_values) is not bool:
            raise TypeError("check_values 必须是 bool")
        self.robot = robot
        self.cfg = cfg
        self.check_values = check_values

    def _tensor(
        self, name: str, tensor: torch.Tensor, shape: tuple[int, ...],
        *, device: torch.device, dtype: torch.dtype, is_bool: bool = False,
        is_int: bool = False,
    ) -> None:
        """功能：阻止不同环境、不同精度或错误形状的数据混在一起算奖励。

        输入：name 是字段名；tensor 是数据；shape 是期望大小；
              device/dtype 是与参考动作相同的设备/精度；is_bool/is_int 是类型要求。
        输出：无；不符合合同则抛异常。
        """
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} 必须是 {shape} 的张量")
        if tensor.device != device:
            raise ValueError(f"{name} 的设备与参考动作不同")
        expected_dtypes = (torch.int32, torch.int64) if is_int else (
            (torch.bool,) if is_bool else (dtype,)
        )
        if tensor.dtype not in expected_dtypes:
            raise TypeError(f"{name} 的数据类型错误，必须是 {expected_dtypes}")
        if self.check_values and not is_bool and not is_int and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} 有 NaN 或 Inf")

    def _unit_quat(self, name: str, quat: torch.Tensor) -> None:
        """功能：检查输入的四元数是不是近似长度 1，防止朝向奖励失真。

        输入：name 是字段名；quat 形状 [...,4]。
        输出：无；错误的四元数导致报错。不负责修补坏姿态。
        """
        if self.check_values:
            norm = torch.linalg.vector_norm(quat, dim=-1)
            if ((norm - 1.0).abs() > 1e-3).any().item():
                raise ValueError(f"{name} 的四元数长度应接近 1")

    def _require(self, condition: bool, description: str) -> None:
        """功能：当奖励被打开但缺关键数据时直接拒绝，不假装给了奖励。

        输入：condition 表示数据是否齐全；description 是缺什么的说明。
        输出：无；条件不成立时抛异常。
        """
        if not condition:
            raise ValueError(description)

    @torch.no_grad()
    def forward(
        self,
        state: TrackingState,
        reference: ReferenceWindow,
        *,
        bodies: BodyTrackingState | None = None,
        body_reference: BodyReferenceWindow | None = None,
        feet: FootSiteState | None = None,
        foot_reference: FootSiteTarget | None = None,
    ) -> MotionTrackingRewardResult:
        """功能：比较 mjlab 当前真实状态与参考轨迹，计算 PPO 本步奖励。

        输入：
            state：TrackingState，[B,N] 关节+ [B,3/4] 根部件真实状态。
            reference：MotionSampler.reference_window() 的输出，[B,H,N] 等。
                **第0个参考时间点必须是当前时刻**，其 time_offsets_s[:,0] 为0。
            bodies：可选 BodyTrackingState，真实部件位姿。
            body_reference：可选 BodyReferenceWindow，参考身体部件轨迹。
                两者均提供后，才可开启 body_* 奖励。
            feet：可选 FootSiteState，mjlab 真实鞋底 site 的位置/接触。
            foot_reference：可选 FootSiteTarget，参考鞋底 site 的位置/阶段标签。
                两者均提供后，才可开启脚部专项奖励。

        输出：MotionTrackingRewardResult：
            total：[B]，最终奖励；交给后续 PPO collector 保存。
            terms：字典，已乘权重的每项奖励 [B]。
            metrics：字典，通常是 [B]；foot_*_by_site 是 [B,S] 的逐脚底点误差。
            foot_site_names：逐脚底点指标的真实列名，方便单独验收左右脚。

        使用顺序：mjlab 执行一步 → sampler.advance() → 获取当前 reference →
                  读取 mjlab 此刻的真实状态 → 调用本函数 → 保存 total。
        注意：mjlab 第11模块环境当前 rewards={}；因此不能使用 bridge.step()
              返回的零 reward 冒充此函数计算的追踪奖励。
        """
        if not isinstance(state, TrackingState) or not isinstance(reference, ReferenceWindow):
            raise TypeError("state 必须是 TrackingState，reference 必须是 ReferenceWindow")
        self.robot.check_joint_set(state.joint_names, source="真实物理状态")
        self.robot.check_exact_order(reference.joint_names, source="奖励参考动作")
        if reference.root_name != self.robot.root_name:
            raise ValueError("奖励参考的 root_name 与机器人合同不匹配")

        # 不能像神经网络那样读取未来 H 个点来计算本时刻奖励。
        # 这里只有索引 0，也就是当前这一控制步的参考目标。
        q_ref = reference.joint_pos
        if not isinstance(q_ref, torch.Tensor) or q_ref.ndim != 3 or q_ref.shape[-1] != self.robot.dof:
            raise ValueError("reference.joint_pos 必须是 [B,H,N]")
        b, h, n = q_ref.shape
        if b < 1 or h < 1 or q_ref.dtype not in (torch.float32, torch.float64):
            raise ValueError("参考 batch、H 必须非空，且只能使用 float32/float64")
        dev, dtype = q_ref.device, q_ref.dtype
        self._tensor("reference.joint_pos", reference.joint_pos, (b, h, n), device=dev, dtype=dtype)
        self._tensor("reference.valid_mask", reference.valid_mask, (b, h), device=dev, dtype=dtype, is_bool=True)
        self._tensor("reference.time_offsets_s", reference.time_offsets_s, (b, h), device=dev, dtype=dtype)
        if not reference.valid_mask[:, 0].all().item():
            raise ValueError("参考第0帧不是有效帧：应该提前结束动作并重置")
        if (reference.time_offsets_s[:, 0].abs() > 1e-5).any().item():
            raise ValueError("奖励必须读取当前参考，future_offsets_s 第一个值应为0")

        # 严格检查当前目标/当前真实状态。未来目标由数据采样器自己验证。
        for name, val, width in (
            ("reference.joint_vel", reference.joint_vel, n),
            ("reference.root_pos_w", reference.root_pos_w, 3),
            ("reference.root_quat_wxyz", reference.root_quat_wxyz, 4),
            ("reference.root_lin_vel_w", reference.root_lin_vel_w, 3),
            ("reference.root_ang_vel_w", reference.root_ang_vel_w, 3),
        ):
            self._tensor(name, val, (b, h, width), device=dev, dtype=dtype)
        for name, value, width in (
            ("state.joint_pos", state.joint_pos, n),
            ("state.joint_vel", state.joint_vel, n),
            ("state.root_pos_w", state.root_pos_w, 3),
            ("state.root_quat_wxyz", state.root_quat_wxyz, 4),
            ("state.root_lin_vel_w", state.root_lin_vel_w, 3),
            ("state.root_ang_vel_w", state.root_ang_vel_w, 3),
        ):
            self._tensor(name, value, (b, width), device=dev, dtype=dtype)
        self._unit_quat("当前 root", state.root_quat_wxyz)
        self._unit_quat("参考 root", reference.root_quat_wxyz[:, 0])

        # 真实的关节角可能是 mjlab 顺序；统一按关节名字变成策略顺序。
        actual_q = self.robot.to_policy_order(state.joint_pos, state.joint_names)
        actual_v = self.robot.to_policy_order(state.joint_vel, state.joint_names)
        target_q = reference.joint_pos[:, 0]
        target_v = reference.joint_vel[:, 0]
        cfg = self.cfg
        total = target_q.new_zeros(b)
        terms: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        def add(name: str, reward_score: torch.Tensor, weight: float) -> None:
            """功能：把一个 0~1 的奖励乘权重并加入最终总分。

            输入：name 是项目名；reward_score [B]；weight 是配置里的权重。
            输出：无；同时更新 total 和 terms[name]。
            """
            nonlocal total
            if weight > 0:
                contribution = reward_score * weight
                terms[name] = contribution
                total = total + contribution

        # 关节角：不用角度取模，以免把关节物理限位两端误认为接近。
        joint_pos_sq = (actual_q - target_q).square().mean(dim=-1)
        metrics["joint_pos_rmse_rad"] = torch.sqrt(joint_pos_sq)
        add("joint_pos", _score(joint_pos_sq, cfg.joint_pos_sigma_rad), cfg.joint_pos_weight)

        # 关节速度：动作节奏不对，也应该体现到奖励中。
        joint_vel_sq = (actual_v - target_v).square().mean(dim=-1)
        metrics["joint_vel_rmse_rad_s"] = torch.sqrt(joint_vel_sq)
        add("joint_vel", _score(joint_vel_sq, cfg.joint_vel_sigma_rad_s), cfg.joint_vel_weight)

        # 根部件水平位置：不仅要站稳，还要沿着参考方向移动正确的距离。
        root_xy_sq = (state.root_pos_w[:, :2] - reference.root_pos_w[:, 0, :2]).square().sum(-1)
        metrics["root_xy_error_m"] = torch.sqrt(root_xy_sq)
        add("root_xy", _score(root_xy_sq, cfg.root_xy_sigma_m), cfg.root_xy_weight)

        # 根部件高度：与水平位置分开，便于诊断下蹲或跳跃偏差。
        root_z_sq = (state.root_pos_w[:, 2] - reference.root_pos_w[:, 0, 2]).square()
        metrics["root_height_error_m"] = torch.sqrt(root_z_sq)
        add("root_z", _score(root_z_sq, cfg.root_z_sigma_m), cfg.root_z_weight)

        # 根部件朝向：使用四元数角度差，不直接相减 wxyz 四个数字。
        root_rot = _quaternion_error_rad(state.root_quat_wxyz, reference.root_quat_wxyz[:, 0])
        metrics["root_rot_error_rad"] = root_rot
        add("root_rot", _score(root_rot.square(), cfg.root_rot_sigma_rad), cfg.root_rot_weight)

        # 根部件世界速度：保证实际移动节奏更接近目标。
        root_lin_sq = (state.root_lin_vel_w - reference.root_lin_vel_w[:, 0]).square().sum(-1)
        root_ang_sq = (state.root_ang_vel_w - reference.root_ang_vel_w[:, 0]).square().sum(-1)
        metrics["root_lin_vel_error_m_s"] = torch.sqrt(root_lin_sq)
        metrics["root_ang_vel_error_rad_s"] = torch.sqrt(root_ang_sq)
        add("root_lin_vel", _score(root_lin_sq, cfg.root_lin_vel_sigma_m_s), cfg.root_lin_vel_weight)
        add("root_ang_vel", _score(root_ang_sq, cfg.root_ang_vel_sigma_rad_s), cfg.root_ang_vel_weight)

        # 身体部件奖励：只在两套真实、同名的 body 数据同时可用时计算。
        body_enabled = any((cfg.body_pos_weight, cfg.body_rot_weight, cfg.body_lin_vel_weight))
        self._require(not body_enabled or (bodies is not None and body_reference is not None),
                      "开启了 body 跟踪，但未同时提供真实 bodies 和 body_reference")
        if bodies is not None or body_reference is not None:
            self._require(bodies is not None and body_reference is not None,
                          "身体真实数据和参考数据必须成对提供")
            self._body_rewards(bodies, body_reference, reference, b, h, dev, dtype,
                               add, metrics)

        # 脚底点奖励：只有得到真正同名的 sole site 数据才可使用。
        foot_enabled = any((cfg.foot_xy_weight, cfg.foot_height_weight,
                            cfg.foot_rot_weight, cfg.foot_slide_vel_weight,
                            cfg.foot_anchor_still_weight, cfg.foot_contact_weight))
        self._require(not foot_enabled or (feet is not None and foot_reference is not None),
                      "已开启鞋底奖励，但未提供真实 foot site 和同名参考 foot site")
        if feet is not None or foot_reference is not None:
            self._require(feet is not None and foot_reference is not None,
                          "真实鞋底点与参考鞋底点必须成对提供")
            self._foot_rewards(feet, foot_reference, b, dev, dtype, add, metrics)

        # 这里返回的是控制步的奖励，而不是自动插入 mjlab 的 reward_manager。
        if self.check_values and not torch.isfinite(total).all().item():
            raise ValueError("汇总奖励出现 NaN/Inf")
        metrics["total_reward"] = total.clone()
        return MotionTrackingRewardResult(
            total=total, terms=terms, metrics=metrics,
            foot_site_names=tuple(foot_reference.site_names) if foot_reference is not None else (),
        )

    def _body_rewards(
        self, actual: BodyTrackingState, target: BodyReferenceWindow,
        reference: ReferenceWindow, b: int, h: int, dev: torch.device,
        dtype: torch.dtype, add, metrics: dict[str, torch.Tensor],
    ) -> None:
        """功能：奖励身体部件的位置、朝向、速度接近参考。

        输入：actual 是真实 body [B,M,...]；target 是第12模块的 [B,H,M,...]；
              reference 供检查两者时间点是否一致；b/h/dev/dtype 是当前批次合同；
              add 是 forward 内部的加分函数；metrics 存误差。
        输出：无；通过 add 更新最终奖励，通过 metrics 记录诊断数据。
        注意：只使用 target 的当前第0帧；按照身体部件名字重新排序。
        """
        if not isinstance(actual, BodyTrackingState) or not isinstance(target, BodyReferenceWindow):
            raise TypeError("身体真实/参考数据类型错误")
        actual_names = _valid_names(actual.body_names, "真实 body_names")
        target_names = _valid_names(target.body_names, "参考 body_names")
        missing = set(target_names) - set(actual_names)
        if missing:
            raise ValueError(f"真实 body 缺少参考部件：{sorted(missing)}")
        m, actual_m = len(target_names), len(actual_names)
        index = torch.tensor([actual_names.index(name) for name in target_names],
                             device=dev, dtype=torch.long)
        self._tensor("body_reference.valid_mask", target.valid_mask, (b, h),
                     device=dev, dtype=dtype, is_bool=True)
        if not torch.equal(target.valid_mask, reference.valid_mask):
            raise ValueError("body_reference 与 root 参考的时间有效性不一致")
        for label, obj, nitems, is_sequence in (
            ("body.actual", actual, actual_m, False),
            ("body.target", target, m, True),
        ):
            lead = (b, h, nitems) if is_sequence else (b, nitems)
            for name, width in (("pos_w", 3), ("quat_wxyz", 4), ("lin_vel_w", 3)):
                self._tensor(f"{label}.{name}", getattr(obj, name), (*lead, width),
                             device=dev, dtype=dtype)
        self._unit_quat("body.actual.quat_wxyz", actual.quat_wxyz)
        self._unit_quat("body.target.quat_wxyz(当前)", target.quat_wxyz[:, 0])
        p = actual.pos_w.index_select(-2, index)
        q = actual.quat_wxyz.index_select(-2, index)
        v = actual.lin_vel_w.index_select(-2, index)
        target_p, target_q, target_v = target.pos_w[:, 0], target.quat_wxyz[:, 0], target.lin_vel_w[:, 0]
        cfg = self.cfg
        pos_sq = (p - target_p).square().sum(-1).mean(-1)
        rot_sq = _quaternion_error_rad(q, target_q).square().mean(-1)
        vel_sq = (v - target_v).square().sum(-1).mean(-1)
        metrics["body_pos_rmse_m"] = torch.sqrt(pos_sq)
        metrics["body_rot_rmse_rad"] = torch.sqrt(rot_sq)
        metrics["body_lin_vel_rmse_m_s"] = torch.sqrt(vel_sq)
        add("body_pos", _score(pos_sq, cfg.body_pos_sigma_m), cfg.body_pos_weight)
        add("body_rot", _score(rot_sq, cfg.body_rot_sigma_rad), cfg.body_rot_weight)
        add("body_lin_vel", _score(vel_sq, cfg.body_lin_vel_sigma_m_s), cfg.body_lin_vel_weight)

    def _foot_rewards(
        self, actual: FootSiteState, target: FootSiteTarget,
        b: int, dev: torch.device, dtype: torch.dtype, add,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        """功能：按真实鞋底 site 的位置和显式阶段标签计算专项奖励。

        输入：actual 是物理鞋底点 [B,S,...]；target 是同名参考鞋底点 [B,S,...]；
              b/dev/dtype 是当前批次合同；add 保存加权奖励；metrics 保存误差。
        输出：无；更新总奖励与 metrics。

        处理方式：
            位置/高度可以始终跟踪。
            ANCHOR 时要求鞋底实际水平速度小，减少锚脚滑移。
            SLIDE 时要求鞋底水平速度接近参考滑动速度，而不是锁死滑脚。
            ANCHOR/SLIDE 期望接触，SWING 期望离地，TRANSITION 不强加接触。
        """
        if not isinstance(actual, FootSiteState) or not isinstance(target, FootSiteTarget):
            raise TypeError("鞋底真实/参考数据类型错误")
        actual_names = _valid_names(actual.site_names, "真实 site_names")
        target_names = _valid_names(target.site_names, "参考 site_names")
        if set(actual_names) != set(target_names):
            raise ValueError("真实与参考鞋底点名称不完全相同，不能把脚踝位置当成鞋底")
        s = len(target_names)
        # 将真实 site 顺序调整到参考顺序；不必强行要求仿真器本来就是这个顺序。
        index = torch.tensor([actual_names.index(name) for name in target_names],
                             device=dev, dtype=torch.long)
        self._tensor("foot.actual.pos_w", actual.pos_w, (b, s, 3), device=dev, dtype=dtype)
        self._tensor("foot.target.pos_w", target.pos_w, (b, s, 3), device=dev, dtype=dtype)
        if actual.vel_w is not None:
            self._tensor("foot.actual.vel_w", actual.vel_w, (b, s, 3), device=dev, dtype=dtype)
        if actual.prev_pos_w is not None:
            self._tensor("foot.actual.prev_pos_w", actual.prev_pos_w, (b, s, 3), device=dev, dtype=dtype)
        if target.vel_w is not None:
            self._tensor("foot.target.vel_w", target.vel_w, (b, s, 3), device=dev, dtype=dtype)
        if actual.quat_wxyz is not None:
            self._tensor("foot.actual.quat_wxyz", actual.quat_wxyz, (b, s, 4), device=dev, dtype=dtype)
            self._unit_quat("foot.actual.quat_wxyz", actual.quat_wxyz)
        if target.quat_wxyz is not None:
            self._tensor("foot.target.quat_wxyz", target.quat_wxyz, (b, s, 4), device=dev, dtype=dtype)
            self._unit_quat("foot.target.quat_wxyz", target.quat_wxyz)
        if actual.contact is not None:
            self._tensor("foot.actual.contact", actual.contact, (b, s),
                         device=dev, dtype=dtype, is_bool=True)
        if target.phase is not None:
            self._tensor("foot.target.phase", target.phase, (b, s),
                         device=dev, dtype=dtype, is_int=True)
            if ((target.phase < 0) | (target.phase > 4)).any().item():
                raise ValueError("foot.phase 只能是 UNKNOWN/ANCHOR/SLIDE/SWING/TRANSITION")
        if actual.dt_s is not None:
            if isinstance(actual.dt_s, bool) or not isinstance(actual.dt_s, (float, int)):
                raise TypeError("foot.dt_s 必须是正数")
            if not math.isfinite(float(actual.dt_s)) or actual.dt_s <= 0:
                raise ValueError("foot.dt_s 必须是正有限数")

        pos = actual.pos_w.index_select(1, index)
        cfg = self.cfg
        xy_sq = (pos[..., :2] - target.pos_w[..., :2]).square().sum(-1).mean(-1)
        height_sq = (pos[..., 2] - target.pos_w[..., 2]).square().mean(-1)
        # 平均值便于 PPO 画曲线，但验收必须保留每个脚底点的独立误差。
        # 例如左脚脚跟对了、右脚脚尖错了，不能只看平均值。
        metrics["foot_xy_rmse_m"] = torch.sqrt(xy_sq)
        metrics["foot_height_rmse_m"] = torch.sqrt(height_sq)
        metrics["foot_xy_error_by_site_m"] = torch.linalg.vector_norm(
            pos[..., :2] - target.pos_w[..., :2], dim=-1
        )
        metrics["foot_height_error_by_site_m"] = (
            pos[..., 2] - target.pos_w[..., 2]
        ).abs()
        add("foot_xy", _score(xy_sq, cfg.foot_xy_sigma_m), cfg.foot_xy_weight)
        add("foot_height", _score(height_sq, cfg.foot_height_sigma_m), cfg.foot_height_weight)

        if cfg.foot_rot_weight > 0:
            self._require(actual.quat_wxyz is not None and target.quat_wxyz is not None,
                          "开启 foot_rot 必须提供真实与参考鞋底点朝向")
        if actual.quat_wxyz is not None and target.quat_wxyz is not None:
            rot_sq = _quaternion_error_rad(
                actual.quat_wxyz.index_select(1, index), target.quat_wxyz
            ).square().mean(-1)
            metrics["foot_rot_rmse_rad"] = torch.sqrt(rot_sq)
            metrics["foot_rot_error_by_site_rad"] = _quaternion_error_rad(
                actual.quat_wxyz.index_select(1, index), target.quat_wxyz
            )
            add("foot_rot", _score(rot_sq, cfg.foot_rot_sigma_rad), cfg.foot_rot_weight)

        # 可以不提供实际速度；但要做滑移/滑速专项奖励时必须有速度信息。
        vel = None
        if actual.vel_w is not None:
            vel = actual.vel_w.index_select(1, index)
        elif actual.prev_pos_w is not None and actual.dt_s is not None:
            old_pos = actual.prev_pos_w.index_select(1, index)
            vel = (pos - old_pos) / float(actual.dt_s)
        needs_vel = cfg.foot_slide_vel_weight > 0 or cfg.foot_anchor_still_weight > 0
        self._require(not needs_vel or vel is not None,
                      "滑脚/锚脚速度奖励需要 foot.vel_w 或 prev_pos_w + dt_s")
        needs_phase = (cfg.foot_slide_vel_weight > 0 or cfg.foot_anchor_still_weight > 0
                       or cfg.foot_contact_weight > 0)
        self._require(not needs_phase or target.phase is not None,
                      "滑脚、锚脚、接触奖励必须有真实 phase 标注，不能推测")

        if target.phase is not None:
            phases = target.phase
            metrics["foot_phase_by_site"] = phases.clone()
            anchor = phases == FootPhase.ANCHOR
            slide = phases == FootPhase.SLIDE
            swing = phases == FootPhase.SWING
            known_contact = anchor | slide | swing
            metrics["anchor_site_fraction"] = anchor.to(dtype).mean(-1)
            metrics["slide_site_fraction"] = slide.to(dtype).mean(-1)
            metrics["contact_label_fraction"] = known_contact.to(dtype).mean(-1)
            if vel is not None:
                xy_vel_sq = vel[..., :2].square().sum(-1)
                anchor_mean_sq, anchor_present = _mean_over_items(xy_vel_sq, anchor)
                metrics["anchor_site_speed_m_s"] = torch.where(
                    anchor_present, torch.sqrt(anchor_mean_sq), torch.zeros_like(anchor_mean_sq)
                )
                metrics["anchor_speed_by_site_m_s"] = torch.where(
                    anchor, torch.sqrt(xy_vel_sq), torch.zeros_like(xy_vel_sq)
                )
                score, _ = _masked_score(xy_vel_sq, anchor, cfg.foot_anchor_still_sigma_m_s)
                add("foot_anchor_still", score, cfg.foot_anchor_still_weight)
                if cfg.foot_slide_vel_weight > 0:
                    self._require(target.vel_w is not None,
                                  "开启 slide_velocity 必须提供参考鞋底点速度 foot_reference.vel_w")
                if target.vel_w is not None:
                    xy_slide_err_sq = (vel[..., :2] - target.vel_w[..., :2]).square().sum(-1)
                    slide_mean_sq, slide_present = _mean_over_items(xy_slide_err_sq, slide)
                    metrics["slide_vel_rmse_m_s"] = torch.where(
                        slide_present, torch.sqrt(slide_mean_sq), torch.zeros_like(slide_mean_sq)
                    )
                    metrics["slide_vel_error_by_site_m_s"] = torch.where(
                        slide, torch.sqrt(xy_slide_err_sq), torch.zeros_like(xy_slide_err_sq)
                    )
                    score, _ = _masked_score(xy_slide_err_sq, slide, cfg.foot_slide_vel_sigma_m_s)
                    add("foot_slide_vel", score, cfg.foot_slide_vel_weight)
            if cfg.foot_contact_weight > 0:
                self._require(actual.contact is not None,
                              "开启接触奖励必须提供真正的 contact 物理传感器结果")
            if actual.contact is not None:
                actual_contact = actual.contact.index_select(1, index)
                expected_contact = anchor | slide
                right_contact = (actual_contact == expected_contact) & known_contact
                score = right_contact.to(dtype).sum(-1) / known_contact.sum(-1).clamp_min(1)
                # 没有明确的接触标签时不奖励；交接期及 UNKNOWN 不强制接触。
                score = torch.where(known_contact.any(-1), score, torch.zeros_like(score))
                metrics["contact_match_fraction"] = score
                metrics["contact_match_by_site"] = right_contact
                add("foot_contact_match", score, cfg.foot_contact_weight)
