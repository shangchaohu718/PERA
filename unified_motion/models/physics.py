"""物理研究基础：观测合同、目标编码、独立动力学预测/记忆，以及轻量MLP策略。

为什么把它们放在一个文件？
    为了让一个研究思路有清楚的完整入口，而不是碎片化新增十几个文件。

最关键的时间顺序：
    观测 o_t + 过去交互历史 → 动力学隐变量 z_t → 策略给出动作 a_t。
    动力学模型使用 (o_t, z_t, a_t) 预测下一次观测 o_(t+1)。
    只有执行 a_t 之后，才能用真实 o_(t+1) 训练预测器。
    绝对不能把未来真实状态作为 t 时刻 Actor 的输入。

输入都必须是机器人可测量或可估计的量：
    仿真真值的摩擦系数、真实物体质量等不是部署时可直接观察的量；
    如果要用它们做监督，可以单独放在训练器/教师，不进入 Actor 输入。

本文件不会连接实际传感器、不会自动发明滑板接触目标、不会运行 PPO。
生产模块不内嵌自测；回归集中在 tests/，用 python tools/evaluate.py 执行。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import functional as F

from ..robot_contract import RobotContract
from ..reference_observations import ReferenceObservation
from .encoder import EncodedMotion
from .fixed_normalization import (
    SCHEMA as NORMALIZATION_SCHEMA,
    LINEAR_VELOCITY_M_S,
    body_observation,
    contact_goal,
    contact_observation,
    object_goal,
    object_observation,
    reference_observation,
)

# 下面这些常量是统一接口中的固定列数，不能随意改而不更新数据管线。
CONTACT_DIM = 12
OBJECT_DIM = 16
CONTACT_GOAL_DIM = 7
OBJECT_GOAL_DIM = 6


@dataclass
class PhysicsObservation:
    """机器人当前可观察到的物理交互信息。

    功能：让 MLP、MoE 和独立动力学模型读取同一种物理数据格式。
    输入：
        body_obs: [B, 6+3N] 或 [B,T,6+3N]，来自旧 ObservationBuilder。
            前6列是身体角速度和重力方向；接着分别是 N 个关节角偏移、
            N 个关节速度和 N 个上一步原始动作（策略关节顺序）。
        base_lin_vel_b: [...,3]，身体坐标下的机身线速度；来自估计器。
        contact_features: [...,C,12]，C 个稳定编号的接触测量点：
            0:3  测量点相对根部件的位置，单位米，身体坐标。
            3:6  该测量点相对接触对象的速度，单位米/秒，身体坐标。
            6:9  接触面的法线单位方向，身体坐标；未接触时可为0。
            9    估计法向接触力，牛顿；不可测量时填0。
            10   是否接触：1接触，0未接触（可使用实机接触估计器）。
            11   力是否有测量/估计来源：1有，0没有。
        contact_mask: [...,C]，True=这个槽位存在，不等于正在接触。
        object_features: [...,M,16]，M 个稳定编号的物体：
            0:3   物体相对根部件的位置，米，身体坐标。
            3:7   物体相对于根部件的朝向四元数，wxyz。
            7:10  物体相对机身的线速度，米/秒，身体坐标。
            10:13 物体相对机身的角速度，弧度/秒，身体坐标。
            13:16 物体外形尺寸（长宽高），米。
        object_mask: [...,M]，True=该槽位有实际可见/可估计物体。

    输出：本对象自己不做网络计算；交给 PhysicsEncoder / DynamicsModel。

    重要：同一环境的 C/M 槽位身份必须在一次回合中保持稳定。
    暂时没有接触点或物体时，C/M 可以为0；不要把不存在的数据伪造为真。
    这不是相机识别模块，外部适配器负责把真实传感器结果整理到这里。
    """

    body_obs: torch.Tensor
    base_lin_vel_b: torch.Tensor
    contact_features: torch.Tensor
    contact_mask: torch.Tensor
    object_features: torch.Tensor
    object_mask: torch.Tensor

    @classmethod
    def from_k1_bridge(
        cls, bridge, *,
        contact_features: torch.Tensor,
        contact_mask: torch.Tensor,
        object_features: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> PhysicsObservation:
        """功能：把第11模块 K1 mjlab 桥里的真实机身状态包装成物理观测。

        输入：
            bridge：已经 bridge.reset() 的 K1MJLabBridge。
            contact_features/contact_mask：必须来自真正的接触点检测或估计。
            object_features/object_mask：必须来自外部物体状态估计；没有物体
                时明确传 [B,0,16] 和 [B,0] 空张量。
        输出：PhysicsObservation，身体状态与线速度从 mjlab 当前帧读取。
        注意：目前的第11模块没有完整鞋底/物体传感器配置，这里不会伪造接触。
              这些输入必须由 mjlab 的传感器适配器真正产生。
        """
        if getattr(bridge, "robot_id", None) != "k1_22dof":
            raise ValueError("from_k1_bridge 仅连接现有第11模块的 K1 环境")
        body_obs = bridge.observation()
        base_vel = bridge.entity.data.root_link_lin_vel_b
        return cls(body_obs, base_vel, contact_features, contact_mask,
                   object_features, object_mask)

    def at(self, t: int) -> PhysicsObservation:
        """功能：从一段 [B,T,...] 的历史里，取出第 t 帧 [B,...]。

        输入：t 为从0开始的时间序号。
        输出：该时间点的 PhysicsObservation，不会改变原始数据。
        用途：可以用同一批数据做整段训练或逐帧推理。
        """
        if self.body_obs.ndim != 3:
            raise ValueError("at(t) 需要输入 [B,T,...] 历史数据")
        if type(t) is not int or t < 0 or t >= self.body_obs.shape[1]:
            raise IndexError("时间 t 超过了历史长度")
        return PhysicsObservation(
            self.body_obs[:, t], self.base_lin_vel_b[:, t],
            self.contact_features[:, t], self.contact_mask[:, t],
            self.object_features[:, t], self.object_mask[:, t],
        )


@dataclass
class ControlGoal:
    """给控制器的目标：允许没有完整动作参考。

    输入（这三组至少有一组对当前真实时间点有效）：
        motion: 旧版 ReferenceObservation，形状 [...,H,16+2N]，可选。
        contact_features: [...,C,7]，与 PhysicsObservation 的接触槽位对齐：
            0:3 希望接触点相对根部件到达的位置（米）。
            3:6 希望接触点相对目标物体的速度（米/秒）。
            6   接触要求：1=应接触，0=应离开，-1=当前不约束接触。
        contact_mask: [...,C]，True=对此接触点指定了目标。
        object_features: [...,M,6]，相对位置(3)+相对线速度(3)目标。
        object_mask: [...,M]，True=对此物体指定了目标。

    输出：目标容器，由 GoalEncoder 转成网络特征。
    注意：这个文件不会自行生成合理的接触顺序。上层任务规划器需要提供目标。
    """

    motion: ReferenceObservation | None = None
    contact_features: torch.Tensor | None = None
    contact_mask: torch.Tensor | None = None
    object_features: torch.Tensor | None = None
    object_mask: torch.Tensor | None = None


@dataclass
class PhysicsEncoding:
    """物理观测编码后的结果。

    context：[...,D]，当前机身与接触/物体整体情况。
    contact_tokens：[...,C,D]，每个接触点自己的特征。
    object_tokens：[...,M,D]，每个外部物体自己的特征。
    这些都是神经网络特征，并不是物理引擎输出的真实参数。
    """

    context: torch.Tensor
    contact_tokens: torch.Tensor
    object_tokens: torch.Tensor


@dataclass
class DynamicsEncoding:
    """独立动力学模型从历史中形成的估计。

    latent：[...,Z]，历史交互形成的动力学隐变量。
    context：[...,D]，当前帧物理特征。
    hidden：GRU 最后一帧的记忆 [B,R]，供后续接着处理时间序列。
    prediction_error：[... ,9+2N]，上一时刻预测与本帧实际观测的差，
        没有可对齐的上一预测时全零。它会参与历史记忆的更新。
    """

    latent: torch.Tensor
    context: torch.Tensor
    hidden: torch.Tensor
    prediction_error: torch.Tensor
    contact_tokens: torch.Tensor | None = None
    object_tokens: torch.Tensor | None = None
    prediction_context: torch.Tensor | None = None

    def physics_encoding(self) -> PhysicsEncoding | None:
        """Return the already-computed scene encoding when this value came from DynamicsModel.

        PolicyInput intentionally stores only context/latent/error, so reconstructed PPO inputs
        return ``None`` here.  Online collection and dynamics supervision retain the tokens and can
        avoid running PhysicsEncoder a second time for the same observation.
        """
        if (self.contact_tokens is None or self.object_tokens is None
                or self.prediction_context is None):
            return None
        return PhysicsEncoding(self.prediction_context, self.contact_tokens, self.object_tokens)


@dataclass
class DynamicsMemory:
    """实时推理时保存的每个机器人环境的历史记忆。

    hidden：[B,R]，上一帧 GRU 的状态。
    env_ids：环境编号，禁止改变环境顺序后偷用原记忆。
    owner：创建它的模型版本标记；训练更新权重后必须作废。
    predicted_core：上一帧做出动作后预测的下一状态 [B,9+2N]，可能为None。
    prediction_valid：上次预测对哪些环境有效 [B]，可能为None。
    下一次 step() 会与真实状态比较，形成预测误差后消费掉这份预测。
    """

    hidden: torch.Tensor
    env_ids: tuple[int, ...]
    owner: object
    predicted_core: torch.Tensor | None = None
    prediction_valid: torch.Tensor | None = None
    # 上一次动作产生的接触/物体预测，供下一帧比较；不是未来真实观测。
    interaction_prediction: DynamicsPrediction | None = None


@dataclass
class DynamicsPrediction:
    """给定当前状态和当前动作后，对下一时刻做的预测。

    next_core：[...,9+2N]，身体线速度3、角速度3、重力方向3、
        关节角偏移N、关节速度N；不包括上一步动作，因为下一帧上一动作已知。
    next_contact_logits：[...,C]，下一帧各接触点是否接触的未归一化分数。
    next_object_pos：[...,M,3]，下一帧物体相对身体的位置。
    next_object_lin_vel：[...,M,3]，下一帧物体相对身体的线速度。
    next_object_ang_vel：[...,M,3]，下一帧物体相对身体的角速度。
    next_object_quat：[...,M,4]，下一帧物体相对根部件的 wxyz 朝向。
    """

    next_core: torch.Tensor
    next_contact_logits: torch.Tensor
    next_object_pos: torch.Tensor
    next_object_lin_vel: torch.Tensor
    next_object_ang_vel: torch.Tensor
    next_object_quat: torch.Tensor
    # 预测时存在的槽位。下一帧只比较前后都有效、身份稳定的槽位。
    contact_mask: torch.Tensor | None = None
    object_mask: torch.Tensor | None = None
    # Optional log variance for normalized core-state prediction error.
    next_core_logvar: torch.Tensor | None = None


@dataclass
class DynamicsLoss:
    """动力学预测的监督损失；total 用于反向传播，terms 用于记录诊断。"""
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


@dataclass
class PhysicsPolicyOutput:
    """MLP 策略的一步输出；mean/std 是 PPO 概率分布，不是 PD 目标。

    mean/std：[B,N]，N 为机器人受控关节数。
    value：[B]，Critic 的预测，仅用于训练。
    joint_names：原始动作所对应的策略关节顺序。
    """

    mean: torch.Tensor
    std: torch.Tensor
    value: torch.Tensor
    joint_names: tuple[str, ...]

    def distribution(self) -> Independent:
        """功能：将每个关节的高斯分布组成整台机器人的联合动作分布。

        输入：对象内部保存的 mean/std。
        输出：Independent(Normal(...))，log_prob 会在关节维求和为 [B]。
        """
        return Independent(Normal(self.mean, self.std), 1)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """功能：从 Actor 的分布中采样原始动作。

        输入：不需要其他输入。
        输出：action [B,N] 和它的 log_prob [B]。
        用途：PPO 必须保存这份原始 action，不能用 PD 目标或被裁剪的角度替代。
        """
        dist = self.distribution()
        action = dist.sample()
        return action, dist.log_prob(action)


# ------------------------ 通用的数据检查和安全池化 ------------------------


def _check_tensor(name: str, x: torch.Tensor, shape: tuple[int, ...],
                  device: torch.device, dtype: torch.dtype,
                  *, boolean: bool = False, check_values: bool = True) -> None:
    """功能：阻止环境数、时间数或类型不一致的数据被意外广播。

    输入：张量 x、期望形状 shape、期望设备/类型。
    输出：无；不匹配时给出明确报错。
    """
    if not isinstance(x, torch.Tensor) or tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} 必须是形状 {shape} 的 Tensor")
    if x.device != device or x.dtype != (torch.bool if boolean else dtype):
        raise ValueError(f"{name} 的设备或 dtype 不匹配")
    if check_values and not boolean and not torch.isfinite(x).all().item():
        raise ValueError(f"{name} 包含 NaN 或 Inf")


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """功能：只对真实槽位的特征求平均，不让不存在的物体影响网络。

    输入：x [...,K,D]，mask [...,K]；K 可以是0。
    输出：[...,D]；没有真实槽位时返回全0。
    """
    if x.shape[-2] == 0:
        return x.new_zeros((*x.shape[:-2], x.shape[-1]))
    safe = torch.where(mask.unsqueeze(-1), x, 0.0)
    counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    return safe.sum(dim=-2) / counts


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """功能：提取接触/物体集合中每个隐藏特征的最大有效值。

    输入：x [...,K,D]、mask [...,K]。
    输出：[...,D]；没有真实槽位时全0，避免空张量 max 报错。
    说明：与平均特征搭配，避免多个接触点的信息完全被平均淹没。
    """
    if x.shape[-2] == 0:
        return x.new_zeros((*x.shape[:-2], x.shape[-1]))
    low = torch.finfo(x.dtype).min
    peaks = torch.where(mask.unsqueeze(-1), x, low).amax(dim=-2)
    return torch.where(mask.any(dim=-1, keepdim=True), peaks, 0.0)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """功能：创建两层小 MLP，用于数据编码或预测。

    输入：输入宽度、中间宽度、输出宽度。
    输出：nn.Sequential 网络；输出张量前面的 B/T 轴不变。
    """
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, out_dim))


def _check_observation(obs: PhysicsObservation, robot: RobotContract,
                       device: torch.device, dtype: torch.dtype,
                       check_values: bool) -> tuple[int, ...]:
    """功能：严格验证一帧或一段物理观测的所有张量。

    输入：obs、机器人合同、模型设备/精度。
    输出：leading，即 (B,) 或 (B,T)，用于构造其余张量。
    说明：不同模型/不同机器人不能只因张量宽度碰巧相同就随意互换。
    """
    if not isinstance(obs, PhysicsObservation) or not isinstance(obs.body_obs, torch.Tensor):
        raise TypeError("输入必须是 PhysicsObservation")
    b = obs.body_obs
    if b.ndim not in (2, 3) or any(size == 0 for size in b.shape):
        raise ValueError("body_obs 必须是非空 [B,O] 或 [B,T,O]")
    leading = tuple(b.shape[:-1])
    _check_tensor("body_obs", b, (*leading, 6 + 3*robot.dof), device, dtype,
                  check_values=check_values)
    _check_tensor("base_lin_vel_b", obs.base_lin_vel_b, (*leading, 3), device, dtype,
                  check_values=check_values)
    for name, x, expected_last in (
        ("contact_features", obs.contact_features, CONTACT_DIM),
        ("object_features", obs.object_features, OBJECT_DIM),
    ):
        if not isinstance(x, torch.Tensor) or x.ndim != b.ndim + 1 or x.shape[-1] != expected_last:
            raise ValueError(f"{name} 必须是 [...,槽位数量,{expected_last}]")
        if tuple(x.shape[:-2]) != leading:
            raise ValueError(f"{name} 的 B/T 轴与 body_obs 不一致")
        _check_tensor(name, x, (*leading, x.shape[-2], expected_last), device, dtype,
                      check_values=check_values)
    _check_tensor("contact_mask", obs.contact_mask,
                  (*leading, obs.contact_features.shape[-2]), device, dtype,
                  boolean=True, check_values=check_values)
    _check_tensor("object_mask", obs.object_mask,
                  (*leading, obs.object_features.shape[-2]), device, dtype,
                  boolean=True, check_values=check_values)
    if check_values:
        c, m = obs.contact_features, obs.contact_mask
        if c.shape[-2]:
            flags = c[..., 10]
            if ((flags[m] < 0) | (flags[m] > 1)).any().item():
                raise ValueError("contact_features 第10列必须在0~1之间")
            if ((c[..., 11][m] < 0) | (c[..., 11][m] > 1)).any().item():
                raise ValueError("接触力可用标志必须在0~1之间")
        obj, mask = obs.object_features, obs.object_mask
        if obj.shape[-2] and mask.any().item():
            quat = obj[..., 3:7][mask]
            if ((torch.linalg.vector_norm(quat, dim=-1)-1).abs() > 1e-3).any().item():
                raise ValueError("真实物体的相对四元数必须接近单位长度")
    return leading


def _canonical_quaternion(q: torch.Tensor) -> torch.Tensor:
    """给等价的 q/-q 选同一个符号，不修改它代表的真实朝向。

    输入：[...,4] 四元数。输出：相同形状；零补位仍为零。
    选择绝对值最大分量为正，避免 w 恰好为零时无法区分符号。
    """
    index = q.abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(q.gather(-1, index) < 0, -1.0, 1.0)
    return q * sign


def _safe_unit(value: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """预测方向长度几乎为零时，沿用上一真实方向，不能把零向量称为单位方向。

    输入：value/fallback 相同形状 [...,K]；输出：有限、长度为1的向量。
    fallback 本身为零（物体补位）时使用第一轴作为无意义的占位方向。
    """
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    fallback_norm = torch.linalg.vector_norm(fallback, dim=-1, keepdim=True)
    identity = torch.zeros_like(fallback); identity[..., 0] = 1.0
    unit_fallback = torch.where(fallback_norm > 1e-6, fallback / fallback_norm.clamp_min(1e-6), identity)
    return torch.where(norm > 1e-6, value / norm.clamp_min(1e-6), unit_fallback)


def _core_state(obs: PhysicsObservation, robot: RobotContract) -> torch.Tensor:
    """功能：取出“真正会受物理动作影响”的机身与关节状态。

    输入：PhysicsObservation，其中 body_obs 已经是策略关节顺序。
    输出：[... ,9+2N]，顺序是机身线速度3 + 角速度3 + 重力3 +
          关节位置N + 关节速度N。
    原因：body_obs 末尾的 previous_action 是已知输入，不需要模型再去预测。
    """
    return torch.cat((obs.base_lin_vel_b, obs.body_obs[..., :6+2*robot.dof]), dim=-1)


# ------------------------ 第一部分：物理场景编码 ------------------------


class PhysicsEncoder(nn.Module):
    """从一帧或多帧物理交互中提取网络可使用的特征。

    输入：PhysicsObservation（身体状态、接触点、外部物体）。
    输出：PhysicsEncoding，包含整体 context 与每个槽位的 token。
    用途：让同一套动力学模型处理不同数量的物体/接触点。
    """

    def __init__(self, robot: RobotContract, *, width: int = 128,
                 hidden_dim: int = 256, check_values: bool = True,
                 scene_encoder: str = "set", graph_layers: int = 2,
                 contact_object_indices: Sequence[int] | None = None) -> None:
        """功能：创建可训练的身体、接触和物体编码器。

        输入：robot 关节合同；width 特征宽度；hidden_dim MLP 宽度。
        输出：无；创建神经网络参数供优化器学习。
        """
        super().__init__()
        if not isinstance(robot, RobotContract):
            raise TypeError("robot 必须是 RobotContract")
        if any(type(v) is not int or v < 2 for v in (width, hidden_dim)):
            raise ValueError("width/hidden_dim 必须是>=2的整数")
        self.robot, self.width, self.check_values = robot, width, check_values
        self.body_net = _mlp(6 + 3*robot.dof + 3, hidden_dim, width)
        self.contact_net = _mlp(CONTACT_DIM, hidden_dim, width)
        self.object_net = _mlp(OBJECT_DIM, hidden_dim, width)
        self.merge = nn.Sequential(nn.Linear(width*5, width), nn.LayerNorm(width))
        if scene_encoder not in ("set", "gnn"):
            raise ValueError("scene_encoder 必须是 set 或 gnn")
        if type(graph_layers) is not int or graph_layers < 1:
            raise ValueError("graph_layers 必须是正整数")
        self.scene_encoder = scene_encoder
        self.graph = None
        if scene_encoder == "gnn":
            if contact_object_indices is None:
                raise ValueError("GNN必须明确提供contact_object_indices，不能按槽位数猜物体关系")
            from .graph_encoder import PhysicsRelationGraph
            self.graph = PhysicsRelationGraph(width, contact_object_indices, n_layers=graph_layers)
        elif contact_object_indices is not None:
            raise ValueError("scene_encoder=set时不使用GNN映射，请删除无效配置")

    def forward(self, obs: PhysicsObservation) -> PhysicsEncoding:
        """功能：编码当前身体、每个接触点和每个物体，再形成整体物理特征。

        输入：obs，可为一帧 [B,...] 或一段历史 [B,T,...]。
        输出：context [...,width]；contact_tokens [...,C,width]；
              object_tokens [...,M,width]。
        说明：这里不会跨时间学习，历史因果关系交给下面的 DynamicsModel。
        """
        w = self.body_net[0].weight
        lead = _check_observation(obs, self.robot, w.device, w.dtype, self.check_values)
        body = self.body_net(torch.cat((
            body_observation(obs.body_obs, self.robot),
            obs.base_lin_vel_b / LINEAR_VELOCITY_M_S,
        ), dim=-1))
        contact_input = torch.where(obs.contact_mask.unsqueeze(-1), obs.contact_features, 0.0)
        object_input = torch.where(obs.object_mask.unsqueeze(-1), obs.object_features, 0.0)
        # q 与 -q 是同一个方向。用绝对值最大分量的符号统一表示，
        # 防止数据文件仅仅改变四元数符号，就让物理网络误以为物体变了。
        quat = _canonical_quaternion(object_input[..., 3:7])
        object_input = torch.cat((object_input[..., :3], quat, object_input[..., 7:]), dim=-1)
        object_input = torch.where(obs.object_mask.unsqueeze(-1), object_input, 0.0)
        contact = self.contact_net(contact_observation(contact_input, self.robot))
        objects = self.object_net(object_observation(object_input))
        contact = torch.where(obs.contact_mask.unsqueeze(-1), contact, 0.0)
        objects = torch.where(obs.object_mask.unsqueeze(-1), objects, 0.0)
        if self.graph is not None:
            # GNN只替换物理空间特征处理；不改变独立动力学GRU的时间含义。
            # 节点类型、关系映射由配置给出，绝不按最大接触力临时重新编号。
            body, contact, objects = self.graph(body.to(w.dtype), contact.to(w.dtype),
                                                objects.to(w.dtype), obs.contact_mask, obs.object_mask)
        summary = torch.cat((body,
                             _masked_mean(contact, obs.contact_mask),
                             _masked_max(contact, obs.contact_mask),
                             _masked_mean(objects, obs.object_mask),
                             _masked_max(objects, obs.object_mask)), dim=-1)
        return PhysicsEncoding(self.merge(summary), contact, objects)


# ------------------------ 第二部分：历史动力学辨识 ------------------------


class DynamicsModel(nn.Module):
    """独立动力学模型：从历史形成隐变量，预测当前动作的后果。

    功能：
        1. 编码接触/物体和身体状态。
        2. 用 GRU 记住此前发生的物理交互，估计 z_t。
        3. 给定当前原始动作 a_t，预测下个时刻的物理状态。

    注意：GRU 只是学习工具，z_t 并不保证等于真实摩擦系数或质量。
    如果训练数据中没有物理变化，这个模块也无法无中生有地学会适应。
    """

    def __init__(self, robot: RobotContract, *, physics_width: int = 128,
                 memory_width: int = 128, latent_width: int = 64,
                 hidden_dim: int = 256, check_values: bool = True,
                 interaction_feedback: bool = False,
                 scene_encoder: str = "set", graph_layers: int = 2,
                 contact_object_indices: Sequence[int] | None = None,
                 system_id_dim: int = 0, teacher_student: bool = False,
                 prediction_horizons: Sequence[int] = (1,),
                 uncertainty: bool = False) -> None:
        """功能：创建物理编码器、历史记忆和下一状态预测网络。

        输入：robot 关节合同；physics_width 物理特征宽度；
              memory_width 历史记忆宽度；latent_width 动力学隐变量宽度；
              hidden_dim 各小 MLP 的中间宽度。
              scene_encoder：set保留原集合编码；gnn先沿配置关系交换节点信息。
              graph_layers：消息传递轮数；contact_object_indices逐接触指定对象编号，
                  -1表示静态世界。必须与环境真实槽位一致，不能自动猜。
        输出：DynamicsModel 对象，本身不启动采样或训练。
        本版独立动力学历史仍用GRU；新LSTM/GRU选项用于Actor时间主干。
        """
        super().__init__()
        if any(type(v) is not int or v < 2 for v in
               (physics_width, memory_width, latent_width, hidden_dim)):
            raise ValueError("模型所有宽度必须是 >=2 的整数")
        self.robot = robot
        self.physics_width, self.memory_width = physics_width, memory_width
        self.latent_width, self.check_values = latent_width, check_values
        self.hidden_dim = hidden_dim
        if type(system_id_dim) is not int or system_id_dim < 0:
            raise ValueError("system_id_dim必须是非负整数")
        if type(teacher_student) is not bool or (teacher_student and not system_id_dim):
            raise ValueError("teacher_student必须是bool，且需要非零system_id_dim")
        self.system_id_dim = system_id_dim
        self.teacher_student = teacher_student
        if type(uncertainty) is not bool:
            raise TypeError("uncertainty必须是bool")
        self.uncertainty = uncertainty
        horizons = tuple(prediction_horizons)
        if (not horizons or horizons[0] != 1 or tuple(sorted(set(horizons))) != horizons
                or any(type(value) is not int or value < 1 for value in horizons)):
            raise ValueError("prediction_horizons必须是以1开头的严格递增正整数")
        self.prediction_horizons = horizons
        self.encoder = PhysicsEncoder(robot, width=physics_width,
                                      hidden_dim=hidden_dim, check_values=check_values,
                                      scene_encoder=scene_encoder, graph_layers=graph_layers,
                                      contact_object_indices=contact_object_indices)
        # 当前物理特征 + 上一原始动作 + 上次预测误差。
        # 预测误差能帮助策略知道：同一个动作在当前环境下为何有偏差。
        self.gru = nn.GRUCell(physics_width + robot.dof + (9+2*robot.dof), memory_width)
        self.latent_head = _mlp(memory_width, hidden_dim, latent_width)
        self.system_id_head = (_mlp(latent_width, hidden_dim, system_id_dim)
                               if system_id_dim else None)
        self.system_id_logvar_head = (
            _mlp(latent_width, hidden_dim, system_id_dim)
            if uncertainty and system_id_dim else None
        )
        self.teacher_encoder = (_mlp(system_id_dim, hidden_dim, latent_width)
                                if teacher_student else None)
        self.teacher_decoder = (_mlp(latent_width, hidden_dim, system_id_dim)
                                if teacher_student else None)
        # 预测器需要当前物理状态 + 动力学隐变量 + 当前准备执行的动作。
        self.action_net = _mlp(robot.dof, hidden_dim, physics_width)
        predict_in = (9 + 2*robot.dof) + latent_width + physics_width + physics_width
        self.state_predictor = _mlp(predict_in, hidden_dim, 9 + 2*robot.dof)
        self.state_logvar_predictor = (
            _mlp(predict_in, hidden_dim, 9 + 2*robot.dof) if uncertainty else None
        )
        self.future_state_predictors = nn.ModuleDict({
            str(horizon): _mlp(predict_in, hidden_dim, 9 + 2*robot.dof)
            for horizon in horizons if horizon > 1
        })
        self.future_action_nets = nn.ModuleDict({
            str(horizon): _mlp(horizon*robot.dof, hidden_dim, physics_width)
            for horizon in horizons if horizon > 1
        })
        self.contact_predictor = _mlp(physics_width + latent_width + physics_width,
                                      hidden_dim, 1)
        # 不仅预测物体位置和线速度，还要预测转动：滑板/平台很依赖角速度。
        self.object_predictor = _mlp(physics_width + latent_width + physics_width,
                                     hidden_dim, 13)
        if type(interaction_feedback) is not bool:
            raise TypeError("interaction_feedback 必须为 bool")
        self.interaction_feedback = interaction_feedback
        # 14维：接触概率残差2、可比较比例1、物体位置/线速/角速残差各3、朝向误差1、可比较比例1。
        # 只编码已经发生的误差，不预测后又偷看未来。
        self.feedback_net = (_mlp(14, hidden_dim, physics_width)
                             if interaction_feedback else None)
        self._owner = object()

    def get_extra_state(self) -> dict:
        """保存动力学接口，避免同维度、不同机器人或关节顺序被误加载。

        输入：无。输出：随 state_dict 保存的纯 Python 字典，不包含运行时记忆。
        """
        state = dict(schema=2, robot=(self.robot.root_name, self.robot.joint_names),
                    widths=(self.physics_width, self.memory_width, self.latent_width, self.hidden_dim),
                    object_rotation="largest_component_positive", timestep="fixed_by_task",
                    interaction_feedback=self.interaction_feedback)
        # 默认set路径保留R19元数据，避免无关功能升级破坏原权重。
        if self.encoder.graph is not None:
            state["scene_graph"] = self.encoder.graph.get_extra_state()
            state["graph_layers"] = self.encoder.graph.gnn.n_layers
        if self.system_id_dim:
            state["system_id_dim"] = self.system_id_dim
        if self.teacher_student:
            state["teacher_student"] = True
        if self.uncertainty:
            state["uncertainty"] = True
        state["input_normalization"] = NORMALIZATION_SCHEMA
        if self.prediction_horizons != (1,):
            state["prediction_horizons"] = self.prediction_horizons
        return state

    def identify_privileged(self, latent: torch.Tensor) -> torch.Tensor:
        """从可部署观测历史的latent预测归一化物理真值，不读取真值本身。"""
        if self.system_id_head is None:
            raise RuntimeError("DynamicsModel未开启系统辨识头")
        if latent.shape[-1] != self.latent_width:
            raise ValueError("latent宽度与系统辨识头不匹配")
        prediction = self.system_id_head(latent)
        if self.check_values and not torch.isfinite(prediction).all().item():
            raise ValueError("系统辨识输出包含NaN/Inf")
        return prediction

    def identify_privileged_distribution(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return privileged mean and calibrated log variance for confidence gating."""
        mean = self.identify_privileged(latent)
        if self.system_id_logvar_head is None:
            raise RuntimeError("DynamicsModel未开启不确定度头")
        logvar = self.system_id_logvar_head(latent).clamp(-6.0, 3.0)
        if self.check_values and not torch.isfinite(logvar).all().item():
            raise ValueError("系统辨识不确定度包含NaN/Inf")
        return mean, logvar

    def encode_privileged_teacher(self, privileged: torch.Tensor) -> torch.Tensor:
        """训练期Teacher：把仿真真值编码为z_priv；调用方负责阻断到学生的梯度。"""
        if self.teacher_encoder is None:
            raise RuntimeError("DynamicsModel未开启Teacher Encoder")
        if privileged.shape[-1] != self.system_id_dim or not torch.isfinite(privileged).all().item():
            raise ValueError("Teacher privileged输入宽度错误或包含NaN/Inf")
        return torch.tanh(self.teacher_encoder(privileged))

    def reconstruct_privileged(self, teacher_latent: torch.Tensor) -> torch.Tensor:
        """从z_priv重建物理真值，防止Teacher latent塌缩成常数。"""
        if self.teacher_decoder is None:
            raise RuntimeError("DynamicsModel未开启Teacher Decoder")
        if teacher_latent.shape[-1] != self.latent_width:
            raise ValueError("Teacher latent宽度错误")
        return self.teacher_decoder(teacher_latent)

    def set_extra_state(self, state: dict) -> None:
        """加载时核对模型定义，并废弃旧 GRU 记忆。输入：保存的元信息；输出：无。"""
        if state != self.get_extra_state():
            raise ValueError("动力学 checkpoint 的关节顺序或接口版本不匹配")
        self.invalidate_memories()

    def invalidate_memories(self) -> None:
        """功能：声明优化器已经修改动力学网络权重，所有旧推理记忆失效。

        输入：无。输出：无。
        调用时机：每次更新本模块权重之后；训练器应清理/重新建立记忆。
        """
        self._owner = object()

    def interaction_error(self, obs: PhysicsObservation,
                          prediction: DynamicsPrediction | None,
                          available: torch.Tensor) -> torch.Tensor:
        """把已经发生的接触/物体预测错误压成14个可比较的数字。

        输入：obs 当前真实单帧或整段；prediction 是之前预测当前时刻的结果；
              available [...] 为同回合且时间连续的标记。
        输出：[...,14]，归一化并限幅的误差，不包含未知摩擦或真实物体质量。
        槽位由任务注册表固定；同一回合不允许更换对象身份。mask不是置信度估计。
        暂无传感器不确定性模型：这里仅用有效性和限幅，不能称为校准过的可信度。
        """
        lead = obs.body_obs.shape[:-1]
        out = obs.body_obs.new_zeros((*lead, 14))
        if prediction is None:
            return out
        c, m = obs.contact_mask.shape[-1], obs.object_mask.shape[-1]
        if prediction.contact_mask is None or prediction.object_mask is None:
            raise ValueError("交互预测必须记录预测时的槽位mask")
        if (prediction.contact_mask.shape != obs.contact_mask.shape or
                prediction.object_mask.shape != obs.object_mask.shape):
            raise ValueError("同回合接触或物体槽位改变，不能比较旧预测")
        cm = obs.contact_mask & prediction.contact_mask & available[..., None]
        om = obs.object_mask & prediction.object_mask & available[..., None]
        def avg(x, mask):
            clean = torch.where(mask[..., None], x, 0.0)
            return clean.sum(-2) / mask.sum(-1).clamp_min(1)[..., None]
        if c:
            error = obs.contact_features[..., 10] - prediction.next_contact_logits.sigmoid()
            out[..., :2] = avg(torch.stack((error, error.abs()), -1), cm)
            out[..., 2] = cm.to(out.dtype).mean(-1)
        if m:
            errors = torch.cat((
                (obs.object_features[..., :3] - prediction.next_object_pos) / 0.10,
                (obs.object_features[..., 7:10] - prediction.next_object_lin_vel) / 1.0,
                (obs.object_features[..., 10:13] - prediction.next_object_ang_vel) / 2.0,
            ), -1).clamp(-5, 5)
            out[..., 3:12] = avg(errors, om)
            q1 = _canonical_quaternion(obs.object_features[..., 3:7])
            q2 = _canonical_quaternion(prediction.next_object_quat)
            dot = (q1*q2).sum(-1).abs().clamp(0, 1)
            # 用姿态差量而不是直接减四元数，q/-q不会制造假错误。
            out[..., 12:13] = avg((1-dot.square())[..., None], om)
            out[..., 13] = om.to(out.dtype).mean(-1)
        out = torch.where(available[..., None], out, 0.0)
        if not torch.isfinite(out).all().item():
            raise ValueError("接触/物体反馈包含非有限误差")
        return out.detach()

    def _feedback_context(self, context: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """输入当前编码和已发生的14维误差，输出加入交互反馈的同宽度特征。

        减去零输入时的编码，确保没有有效预测时不凭空增加偏置。
        参数仍由下一状态监督学习，不由PPO修改。
        """
        if self.feedback_net is None:
            return context
        return context + self.feedback_net(error) - self.feedback_net(torch.zeros_like(error))

    def _advance_hidden(self, context: torch.Tensor, previous_action: torch.Tensor,
                        prediction_error: torch.Tensor, hidden: torch.Tensor,
                        episode_start: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """功能：计算本帧的历史记忆，同时防止不同回合互相串数据。

        输入：当前物理 context [B,D]，上一动作 [B,N]，
              prediction_error [B,9+2N]：上次预测与本次真实值之差，
              hidden [B,R]，episode_start [B]，valid [B]。
        输出：更新后的记忆 [B,R]；补位环境保持旧记忆不动。
        """
        reset_hidden = torch.where(episode_start.unsqueeze(-1), 0.0, hidden)
        # 归一化误差，避免关节速度的数值尺度盖过细微接触反馈。
        n = self.robot.dof
        error_scales = prediction_error.new_tensor(
            [1.0]*9 + [0.3]*n + [3.0]*n
        )
        stable_error = (prediction_error / error_scales).clamp(-5.0, 5.0)
        proposal = self.gru(torch.cat((context, previous_action, stable_error), dim=-1),
                            reset_hidden)
        return torch.where(valid.unsqueeze(-1), proposal, hidden)

    def forward_sequence(self, obs: PhysicsObservation, *,
                         episode_start: torch.Tensor,
                         valid_mask: torch.Tensor | None = None,
                         initial_hidden: torch.Tensor | None = None,
                         previous_predicted_core: torch.Tensor | None = None,
                         previous_prediction_valid: torch.Tensor | None = None,
                         previous_interaction_prediction: DynamicsPrediction | None = None,
                         ) -> DynamicsEncoding:
        """功能：在训练时顺序处理一段历史，让物理特征具有因果记忆。

        输入：
            obs：全部张量都为 [B,T,...]，T 按真实时间从旧到新。
            episode_start：[B,T]，True 表示该帧是重置后的新回合第一帧。
            valid_mask：[B,T]，True 表示真实帧；尾部补位写 False。
            initial_hidden：[B,R]，可选，只有从回合中途切序列时才传入。
            previous_predicted_core：[B,T,9+2N]，可选。每个 t 保存
                t-1 时执行动作后，预测 t 时刻真实状态的结果。
            previous_prediction_valid：[B,T]，表示哪个位置有可用的上一预测。
                从 rollout 采集记录中读取，不可从未来真实状态反推填充。
        输出：DynamicsEncoding：latent [B,T,Z]、context [B,T,D]、hidden [B,R]、
              prediction_error [B,T,9+2N]。
        重要：没有 initial_hidden 时，所有环境的第0帧必须是回合起点。
              使用 PPO 时不要把一个 rollout 随意打乱顺序。
        """
        w = self.gru.weight_ih
        lead = _check_observation(obs, self.robot, w.device, w.dtype, self.check_values)
        if len(lead) != 2:
            raise ValueError("forward_sequence 只接受 [B,T,...] 的历史")
        b, t = lead
        _check_tensor("episode_start", episode_start, (b,t), w.device, w.dtype, boolean=True)
        if valid_mask is None:
            valid_mask = torch.ones(b, t, dtype=torch.bool, device=w.device)
        _check_tensor("valid_mask", valid_mask, (b,t), w.device, w.dtype, boolean=True)
        if (valid_mask[:,1:] & ~valid_mask[:,:-1]).any().item():
            raise ValueError("时间补位只能在最后")
        if (episode_start & ~valid_mask).any().item():
            raise ValueError("补位帧不能标为新回合")
        if initial_hidden is None:
            if not episode_start[:,0].all().item():
                raise ValueError("没有 initial_hidden 时，首帧必须是各自回合起点")
            hidden = w.new_zeros((b, self.memory_width))
        else:
            _check_tensor("initial_hidden", initial_hidden, (b,self.memory_width), w.device,
                          w.dtype, check_values=self.check_values)
            hidden = initial_hidden
        if (previous_predicted_core is None) != (previous_prediction_valid is None):
            raise ValueError("上一预测张量和对应的有效标志必须同时提供，或者同时省略")
        core_width = 9 + 2*self.robot.dof
        if previous_predicted_core is not None:
            _check_tensor("previous_predicted_core",previous_predicted_core,
                          (b,t,core_width),w.device,w.dtype,check_values=self.check_values)
            _check_tensor("previous_prediction_valid",previous_prediction_valid,
                          (b,t),w.device,w.dtype,boolean=True)
        enc = self.encoder(obs)
        contexts = enc.context
        if self.interaction_feedback:
            available = (torch.zeros_like(valid_mask) if previous_prediction_valid is None
                         else previous_prediction_valid & valid_mask & ~episode_start)
            if available.any().item() and previous_interaction_prediction is None:
                raise ValueError("启用交互反馈时，监督重放必须保存之前的接触/物体预测")
            contexts = self._feedback_context(contexts, self.interaction_error(
                obs, previous_interaction_prediction, available))
        z_list, err_list = [], []
        for step in range(t):
            # 上一步原始 action 在 body_obs 的最后 N 列，属于当前可知信息。
            previous_action = obs.body_obs[:,step,-self.robot.dof:]
            if previous_predicted_core is None:
                error = obs.body_obs.new_zeros((b,core_width))
            else:
                available = (previous_prediction_valid[:,step] & valid_mask[:,step]
                             & ~episode_start[:,step])
                error = _core_state(obs.at(step),self.robot) - previous_predicted_core[:,step]
                error = torch.where(available.unsqueeze(-1),error,0.0)
            hidden = self._advance_hidden(contexts[:,step], previous_action,
                                          error, hidden, episode_start[:,step], valid_mask[:,step])
            z = self.latent_head(hidden)
            z_list.append(torch.where(valid_mask[:,step,None], z, 0.0))
            err_list.append(error)
        return DynamicsEncoding(torch.stack(z_list, dim=1),
                                torch.where(valid_mask.unsqueeze(-1), contexts, 0.0),
                                hidden, torch.stack(err_list,dim=1),
                                enc.contact_tokens, enc.object_tokens, enc.context)

    @torch.no_grad()
    def step(self, obs: PhysicsObservation, *, episode_start: torch.Tensor,
             env_ids: Sequence[int], memory: DynamicsMemory | None = None,
             ) -> tuple[DynamicsEncoding, DynamicsMemory]:
        """功能：在实时采样时处理一帧，保留各环境的物理记忆。

        输入：
            obs：形状 [B,...] 的物理观测。
            episode_start：[B]，本次观测是否已经进入新回合。
            env_ids：B 个环境的固定编号；顺序必须始终相同。
            memory：上次调用返回的记忆，首次调用时传 None。
        输出：DynamicsEncoding 的 latent [B,Z]、context [B,D]；以及新 memory。
        限制：必须 .eval()；如果首次没有记忆，所有环境必须从新回合开始。
              更新模型权重后调用 invalidate_memories() 并重建记忆。
        """
        if self.training:
            raise RuntimeError("DynamicsModel.step 只用于 eval 模式采样")
        w = self.gru.weight_ih
        leading = _check_observation(obs, self.robot, w.device, w.dtype, self.check_values)
        if len(leading) != 1:
            raise ValueError("step 只接受 [B,...] 的单帧数据")
        b = leading[0]
        env_ids = tuple(env_ids)
        if len(env_ids) != b or len(set(env_ids)) != b or any(type(i) is not int for i in env_ids):
            raise ValueError("env_ids 必须是 B 个不重复的整数")
        _check_tensor("episode_start", episode_start, (b,), w.device, w.dtype, boolean=True)
        if memory is None:
            if not episode_start.all().item():
                raise ValueError("第一帧必须是回合起点；中途采样请提供真实历史")
            hidden = w.new_zeros((b,self.memory_width))
        else:
            if not isinstance(memory, DynamicsMemory) or memory.owner is not self._owner:
                raise ValueError("动力学历史记忆属于其他模型或者已经过期")
            if memory.env_ids != env_ids:
                raise ValueError("环境编号或行顺序变化，禁止复用旧记忆")
            _check_tensor("memory.hidden", memory.hidden, (b,self.memory_width),
                          w.device, w.dtype, check_values=self.check_values)
            hidden = memory.hidden
        if memory is not None and (memory.predicted_core is None) != (memory.prediction_valid is None):
            raise ValueError("memory 中的预测和有效标记不完整")
        if memory is None or memory.predicted_core is None:
            error = obs.body_obs.new_zeros((b,9+2*self.robot.dof))
        else:
            _check_tensor("memory.predicted_core",memory.predicted_core,
                          (b,9+2*self.robot.dof),w.device,w.dtype,check_values=self.check_values)
            _check_tensor("memory.prediction_valid",memory.prediction_valid,(b,),
                          w.device,w.dtype,boolean=True)
            available = memory.prediction_valid & ~episode_start
            error = torch.where(available.unsqueeze(-1),
                                _core_state(obs,self.robot)-memory.predicted_core,0.0)
        enc = self.encoder(obs)
        context = enc.context
        if self.interaction_feedback:
            available = (torch.zeros_like(episode_start) if memory is None or memory.prediction_valid is None
                         else memory.prediction_valid & ~episode_start)
            pred = None if memory is None else memory.interaction_prediction
            if available.any().item() and pred is None:
                raise ValueError("交互反馈缺少上一预测，请清理旧版运行时记忆")
            context = self._feedback_context(context, self.interaction_error(obs, pred, available))
        hidden = self._advance_hidden(context, obs.body_obs[...,-self.robot.dof:],
                                      error, hidden, episode_start,
                                      torch.ones(b,dtype=torch.bool,device=w.device))
        z = self.latent_head(hidden)
        # 上一预测已经消费；本帧选完动作后，需要手动存入新的预测。
        return (DynamicsEncoding(z, context, hidden, error,
                                 enc.contact_tokens, enc.object_tokens, enc.context),
                DynamicsMemory(hidden, env_ids, self._owner))

    @torch.no_grad()
    def remember_prediction(self, memory: DynamicsMemory,
                            prediction: DynamicsPrediction, *,
                            valid: torch.Tensor | None = None) -> DynamicsMemory:
        """功能：本帧 Actor 已决定 action 后，保存模型对下一帧的预测。

        输入：memory 是刚从 step() 得到的记忆；prediction 由 predict_next() 产生；
              valid [B] 表示哪些环境的预测可在下一步比较，默认全 True。
        输出：一个新的 DynamicsMemory，下次 step() 就能计算真实预测误差。
        注意：预测必须在当前动作确定后计算，不能在下一帧用真实结果倒填。
        """
        w = self.gru.weight_ih
        if not isinstance(memory, DynamicsMemory) or memory.owner is not self._owner:
            raise ValueError("记忆所属模型不正确或者已过期")
        b = memory.hidden.shape[0]
        _check_tensor("memory.hidden",memory.hidden,(b,self.memory_width),
                      w.device,w.dtype,check_values=self.check_values)
        _check_tensor("prediction.next_core",prediction.next_core,
                      (b,9+2*self.robot.dof),w.device,w.dtype,check_values=self.check_values)
        if valid is None:
            valid = torch.ones(b,dtype=torch.bool,device=w.device)
        _check_tensor("valid",valid,(b,),w.device,w.dtype,boolean=True)
        from dataclasses import fields
        saved = None
        if self.interaction_feedback:
            if prediction.contact_mask is None or prediction.object_mask is None:
                raise ValueError("交互预测缺少预测时有效性mask")
            saved = DynamicsPrediction(**{
                f.name: (None if getattr(prediction, f.name) is None
                         else getattr(prediction, f.name).detach().clone())
                for f in fields(prediction)
            })
        return DynamicsMemory(memory.hidden, memory.env_ids, self._owner,
                              prediction.next_core.detach().clone(), valid.clone(), saved)

    def predict_next(self, obs: PhysicsObservation, action: torch.Tensor,
                     latent: torch.Tensor, *,
                     physics_encoding: PhysicsEncoding | None = None) -> DynamicsPrediction:
        """功能：给定当前真实状态、准备执行的动作和物理记忆，预测下个状态。

        输入：
            obs：[B,...] 或 [B,T,...]，当前真实状态。
            action：[...,N]，当前策略准备执行的原始动作（不能是参考角度）。
            latent：[...,Z]，DynamicsModel 在当前时刻得到的隐变量。
            physics_encoding：同一 obs 已由 step/forward_sequence 算出的编码；提供时
                直接复用，避免 PhysicsEncoder 对同一帧重复前向。
        输出：DynamicsPrediction：下一时刻机身/关节状态、接触概率分数、物体状态。
        注意：预测结果只能用于学习/辅助决策；它不是 mjlab 的真实物理结果。
              如果要把预测用于 Actor，必须保证使用的 action 是当前可知的候选动作。
        """
        w = self.gru.weight_ih
        lead = _check_observation(obs, self.robot, w.device, w.dtype, self.check_values)
        _check_tensor("action", action, (*lead,self.robot.dof), w.device, w.dtype,
                      check_values=self.check_values)
        _check_tensor("latent", latent, (*lead,self.latent_width), w.device, w.dtype,
                      check_values=self.check_values)
        physics = self.encoder(obs) if physics_encoding is None else physics_encoding
        if not isinstance(physics, PhysicsEncoding):
            raise TypeError("physics_encoding 必须是 PhysicsEncoding 或 None")
        c, m = obs.contact_features.shape[-2], obs.object_features.shape[-2]
        _check_tensor("physics_encoding.context", physics.context,
                      (*lead,self.physics_width), w.device, w.dtype,
                      check_values=self.check_values)
        _check_tensor("physics_encoding.contact_tokens", physics.contact_tokens,
                      (*lead,c,self.physics_width), w.device, w.dtype,
                      check_values=self.check_values)
        _check_tensor("physics_encoding.object_tokens", physics.object_tokens,
                      (*lead,m,self.physics_width), w.device, w.dtype,
                      check_values=self.check_values)
        action_code = self.action_net(action)
        core = _core_state(obs, self.robot)
        net_input = torch.cat((core, latent, physics.context, action_code), dim=-1)
        delta = self.state_predictor(net_input)
        core_logvar = (
            None if self.state_logvar_predictor is None
            else self.state_logvar_predictor(net_input).clamp(-6.0, 3.0)
        )
        next_core = core + delta
        # 重力方向是单位向量。这里防止预测器产生无穷大的长度。
        gravity = _safe_unit(next_core[...,6:9], core[...,6:9])
        next_core = torch.cat((next_core[...,:6],gravity,next_core[...,9:]),dim=-1)
        contacts = torch.cat((physics.contact_tokens,
                              latent.unsqueeze(-2).expand(*lead, physics.contact_tokens.shape[-2], self.latent_width),
                              action_code.unsqueeze(-2).expand(*lead, physics.contact_tokens.shape[-2], self.physics_width)),
                             dim=-1)
        objects = torch.cat((physics.object_tokens,
                             latent.unsqueeze(-2).expand(*lead, physics.object_tokens.shape[-2], self.latent_width),
                             action_code.unsqueeze(-2).expand(*lead, physics.object_tokens.shape[-2], self.physics_width)),
                            dim=-1)
        contact_logits = self.contact_predictor(contacts).squeeze(-1)
        object_change = self.object_predictor(objects)
        object_pos = obs.object_features[...,0:3] + object_change[...,0:3]
        object_vel = obs.object_features[...,7:10] + object_change[...,3:6]
        object_ang_vel = obs.object_features[...,10:13] + object_change[...,6:9]
        # q 和 -q 表示同一朝向；这里规范单位长度，避免预测四元数逐渐变形。
        old_quat = _canonical_quaternion(obs.object_features[...,3:7])
        quat_raw = old_quat + object_change[...,9:13]
        object_quat = _canonical_quaternion(_safe_unit(quat_raw, old_quat))
        return DynamicsPrediction(next_core, contact_logits, object_pos,
                                  object_vel, object_ang_vel, object_quat,
                                  obs.contact_mask.clone(), obs.object_mask.clone(),
                                  core_logvar)

    def predict_future_core(self, obs: PhysicsObservation, action_sequence: torch.Tensor,
                            latent: torch.Tensor, *, horizon: int,
                            physics_encoding: PhysicsEncoding | None = None) -> torch.Tensor:
        """Directly predict core state ``horizon`` control steps ahead.

        This is an auxiliary training head, not an autoregressive simulator.  It consumes the
        policy commands recorded over the interval and therefore amplifies slow
        contact/dynamics effects without recursively feeding model errors back as observations.
        """
        if horizon not in self.prediction_horizons or horizon <= 1:
            raise ValueError("horizon必须是模型配置中大于1的预测跨度")
        w = self.gru.weight_ih
        lead = _check_observation(obs, self.robot, w.device, w.dtype, self.check_values)
        _check_tensor("action_sequence", action_sequence,
                      (*lead,horizon,self.robot.dof), w.device, w.dtype,
                      check_values=self.check_values)
        _check_tensor("latent", latent, (*lead,self.latent_width), w.device, w.dtype,
                      check_values=self.check_values)
        physics = self.encoder(obs) if physics_encoding is None else physics_encoding
        if not isinstance(physics, PhysicsEncoding):
            raise TypeError("physics_encoding 必须是 PhysicsEncoding 或 None")
        c, m = obs.contact_features.shape[-2], obs.object_features.shape[-2]
        _check_tensor("physics_encoding.context", physics.context,
                      (*lead,self.physics_width), w.device, w.dtype,
                      check_values=self.check_values)
        _check_tensor("physics_encoding.contact_tokens", physics.contact_tokens,
                      (*lead,c,self.physics_width), w.device, w.dtype,
                      check_values=self.check_values)
        _check_tensor("physics_encoding.object_tokens", physics.object_tokens,
                      (*lead,m,self.physics_width), w.device, w.dtype,
                      check_values=self.check_values)
        # Preserve command order: a gait's left/right sequence cannot be represented by a mean.
        # Each configured horizon owns a fixed-width sequence encoder and prediction head.
        action_code = self.future_action_nets[str(horizon)](
            action_sequence.reshape(*lead,horizon*self.robot.dof))
        core = _core_state(obs, self.robot)
        delta = self.future_state_predictors[str(horizon)](
            torch.cat((core, latent, physics.context, action_code), dim=-1))
        predicted = core + delta
        gravity = _safe_unit(predicted[...,6:9], core[...,6:9])
        return torch.cat((predicted[...,:6], gravity, predicted[...,9:]), dim=-1)

    def prediction_loss(self, prediction: DynamicsPrediction,
                        next_observation: PhysicsObservation, *,
                        valid_transition: torch.Tensor) -> DynamicsLoss:
        """功能：拿动作执行后的真实结果，监督训练独立动力学模型。

        输入：
            prediction：predict_next 在执行动作前算出的预测。
            next_observation：执行动作之后、任何 reset 之前读取的真实下一状态。
            valid_transition：[B] 或 [B,T]，哪些预测与真实结果确实相邻。
                如果下一观测已被 auto-reset 替换，必须把该 transition 设 False。
        输出：DynamicsLoss.total 标量，可用 backward()；terms 为各项诊断损失。
        说明：比较真实观测，不依赖购买额外动力学数据集。
        """
        w = self.gru.weight_ih
        lead = _check_observation(next_observation, self.robot, w.device, w.dtype,
                                  self.check_values)
        _check_tensor("valid_transition", valid_transition, lead, w.device, w.dtype,
                      boolean=True)
        expected_core = (*lead,9 + 2*self.robot.dof)
        _check_tensor("prediction.next_core",prediction.next_core,expected_core,w.device,w.dtype,
                      check_values=self.check_values)
        if prediction.next_core_logvar is not None:
            _check_tensor("prediction.next_core_logvar", prediction.next_core_logvar,
                          expected_core, w.device, w.dtype,
                          check_values=self.check_values)
        c = next_observation.contact_features.shape[-2]
        m = next_observation.object_features.shape[-2]
        _check_tensor("prediction.next_contact_logits",prediction.next_contact_logits,
                      (*lead,c),w.device,w.dtype,check_values=self.check_values)
        for name,x in (("next_object_pos",prediction.next_object_pos),
                       ("next_object_lin_vel",prediction.next_object_lin_vel),
                       ("next_object_ang_vel",prediction.next_object_ang_vel)):
            _check_tensor("prediction."+name,x,(*lead,m,3),w.device,w.dtype,
                          check_values=self.check_values)
        _check_tensor("prediction.next_object_quat",prediction.next_object_quat,
                      (*lead,m,4),w.device,w.dtype,check_values=self.check_values)
        # 分段归一化防止关节角、角速度等不同量纲差异过大。
        n = self.robot.dof
        true = _core_state(next_observation,self.robot)
        active = valid_transition.to(w.dtype)
        if not valid_transition.any().item():
            raise ValueError("没有任何有效状态转移，不能训练动力学模型")
        count = active.sum()
        terms = {}
        for key, begin, end, unit_scale in (
            ("base_linear_velocity", 0,3,1.0),
            ("base_angular_velocity", 3,6,1.0),
            ("gravity_direction", 6,9,1.0),
            ("joint_position", 9,9+n,0.3),
            ("joint_velocity", 9+n,9+2*n,3.0),
        ):
            normalized = (prediction.next_core[...,begin:end]-true[...,begin:end])/unit_scale
            if prediction.next_core_logvar is None:
                error = normalized.square().mean(-1)
            else:
                logvar = prediction.next_core_logvar[...,begin:end]
                error = 0.5 * (normalized.square()*torch.exp(-logvar)+logvar).mean(-1)
            terms[key] = (error*active).sum()/count
        if c > 0:
            cmask = next_observation.contact_mask & valid_transition.unsqueeze(-1)
            labels = next_observation.contact_features[...,10]
            closs = F.binary_cross_entropy_with_logits(prediction.next_contact_logits,labels,
                                                        reduction="none")
            terms["contact"] = (closs*cmask).sum()/cmask.sum().clamp_min(1)
        else:
            terms["contact"] = prediction.next_core.new_zeros(())
        if m > 0:
            omask = next_observation.object_mask & valid_transition.unsqueeze(-1)
            target_pos = next_observation.object_features[...,0:3]
            target_vel = next_observation.object_features[...,7:10]
            target_ang_vel = next_observation.object_features[...,10:13]
            target_quat = next_observation.object_features[...,3:7]
            pos_err = (prediction.next_object_pos-target_pos).square().mean(-1)
            vel_err = (prediction.next_object_lin_vel-target_vel).square().mean(-1)
            ang_err = (prediction.next_object_ang_vel-target_ang_vel).square().mean(-1)
            # q 与 -q 朝向相同；用点积的平方避免四元数变号影响训练。
            quat_cos = (prediction.next_object_quat*target_quat).sum(dim=-1).abs().clamp(0,1)
            rot_err = 1 - quat_cos.square()
            denom = omask.sum().clamp_min(1)
            terms["object_position"] = (pos_err*omask).sum()/denom
            terms["object_velocity"] = (vel_err*omask).sum()/denom
            terms["object_ang_velocity"] = (ang_err*omask).sum()/denom
            terms["object_orientation"] = (rot_err*omask).sum()/denom
        else:
            terms["object_position"] = prediction.next_core.new_zeros(())
            terms["object_velocity"] = prediction.next_core.new_zeros(())
            terms["object_ang_velocity"] = prediction.next_core.new_zeros(())
            terms["object_orientation"] = prediction.next_core.new_zeros(())
        # 第一版对各类标准化损失等权，后续通过训练数据实测再调节。
        total = sum(terms.values())
        return DynamicsLoss(total, terms)


# ------------------------ 第三部分：任务目标编码（无 MoCap 也能给目标） ------------------------


class GoalEncoder(nn.Module):
    """把完整动作参考、接触目标和物体目标合成同一种策略目标特征。

    功能：可以三选一，也可以混合；不要求每个新任务都有完整 MoCap。
    注意：能读取目标不等于会自动规划目标；零样本能力仍要靠训练验证。
    """

    def __init__(self, robot: RobotContract, *, width: int = 128,
                 hidden_dim: int = 256, check_values: bool = True) -> None:
        """功能：创建三种目标各自的小编码器和统一融合层。

        输入：robot 关节合同、width 输出特征宽度、hidden_dim 中间宽度。
        输出：一个可学习的 GoalEncoder。
        """
        super().__init__()
        self.robot, self.width, self.check_values = robot, width, check_values
        self.motion_net = _mlp(16 + 2*robot.dof, hidden_dim, width)
        self.contact_net = _mlp(CONTACT_GOAL_DIM + 7, hidden_dim, width)
        self.object_net = _mlp(OBJECT_GOAL_DIM + 6, hidden_dim, width)
        self.merge = _mlp(3*width+3, hidden_dim, width)

    def forward(self, physics: PhysicsObservation, goal: ControlGoal) -> torch.Tensor:
        """功能：把目标与当前真实物理状态对比，生成目标和误差特征。

        输入：physics 当前观测；goal 是动作参考/接触目标/物体目标。
        输出：[...,width]；同一时刻只使用此刻已知的目标。
        说明：接触位置误差和相对速度误差是逐槽位计算的，避免混淆对象。
        """
        if not isinstance(goal, ControlGoal):
            raise TypeError("goal 必须是 ControlGoal")
        x = physics.body_obs
        lead = tuple(x.shape[:-1]); dev = x.device; dtype = x.dtype
        values = []
        flags = []
        if goal.motion is None:
            values.append(x.new_zeros((*lead,self.width))); flags.append(x.new_zeros(lead))
        else:
            ref = goal.motion
            if not isinstance(ref, ReferenceObservation) or not isinstance(ref.features, torch.Tensor):
                raise TypeError("motion 必须是 ReferenceObservation")
            if ref.features.ndim != x.ndim+1 or ref.features.shape[-2] < 1:
                raise ValueError("motion.features 必须比身体多一个 H 时间点轴")
            h = ref.features.shape[-2]
            _check_tensor("motion.features",ref.features,(*lead,h,16+2*self.robot.dof),
                          dev,dtype,check_values=self.check_values)
            _check_tensor("motion.valid_mask",ref.valid_mask,(*lead,h),dev,dtype,boolean=True)
            normalized=reference_observation(ref.features,self.robot)
            encoded = self.motion_net(torch.where(ref.valid_mask.unsqueeze(-1),normalized,0.0))
            values.append(_masked_mean(encoded,ref.valid_mask))
            flags.append(ref.valid_mask.any(dim=-1).to(dtype))
        if goal.contact_features is None:
            if goal.contact_mask is not None:
                raise ValueError("有 contact_mask 但没有 contact_features")
            values.append(x.new_zeros((*lead,self.width))); flags.append(x.new_zeros(lead))
        else:
            c = physics.contact_features.shape[-2]
            _check_tensor("contact_goal",goal.contact_features,(*lead,c,CONTACT_GOAL_DIM),
                          dev,dtype,check_values=self.check_values)
            _check_tensor("contact_goal_mask",goal.contact_mask,(*lead,c),dev,dtype,boolean=True)
            if (goal.contact_mask & ~physics.contact_mask).any().item():
                raise ValueError("不能给不存在的接触槽位布置目标")
            desired = contact_goal(goal.contact_features)
            real = contact_observation(physics.contact_features,self.robot)
            errors = torch.cat((desired[...,:3]-real[...,:3],
                                desired[...,3:6]-real[...,3:6],
                                torch.where(desired[...,6:7] < 0, 0.0,
                                            desired[...,6:7]-real[...,10:11])),dim=-1)
            parts = torch.cat((desired,errors),dim=-1)
            encoded = self.contact_net(torch.where(goal.contact_mask.unsqueeze(-1),parts,0.0))
            values.append(_masked_mean(encoded,goal.contact_mask))
            flags.append(goal.contact_mask.any(dim=-1).to(dtype))
        if goal.object_features is None:
            if goal.object_mask is not None:
                raise ValueError("有 object_mask 但没有 object_features")
            values.append(x.new_zeros((*lead,self.width)));flags.append(x.new_zeros(lead))
        else:
            m=physics.object_features.shape[-2]
            _check_tensor("object_goal",goal.object_features,(*lead,m,OBJECT_GOAL_DIM),
                          dev,dtype,check_values=self.check_values)
            _check_tensor("object_goal_mask",goal.object_mask,(*lead,m),dev,dtype,boolean=True)
            if (goal.object_mask & ~physics.object_mask).any().item():
                raise ValueError("物体目标指向不可观察的物体，请先由规划器处理")
            desired=object_goal(goal.object_features)
            real=object_observation(physics.object_features)
            errors=torch.cat((desired[...,:3]-real[...,:3],
                              desired[...,3:]-real[...,7:10]),dim=-1)
            encoded=self.object_net(torch.where(goal.object_mask.unsqueeze(-1),
                                                 torch.cat((desired,errors),dim=-1),0.0))
            values.append(_masked_mean(encoded,goal.object_mask))
            flags.append(goal.object_mask.any(dim=-1).to(dtype))
        present=torch.stack(flags,dim=-1)
        if self.check_values and not present.any(dim=-1).all().item():
            raise ValueError("每个真实控制时刻至少需要一类有效目标")
        return self.merge(torch.cat((*values,present),dim=-1))


# ------------------------ 第四部分：MLP 控制分支与 MoE 条件化桥 ------------------------


class PhysicsMLPActorCritic(nn.Module):
    """轻量控制基线：使用物理隐变量 z 和任务目标输出动作/价值。

    输入：PhysicsObservation + DynamicsEncoding + ControlGoal。
    输出：PhysicsPolicyOutput（高斯动作分布参数、Critic value）。
    重要：DynamicsModel 独立训练。训练本 Actor 时默认让 dynamics 特征 detach，
          避免 PPO 策略损失悄悄改坏动力学预测器。Critic 的特权输入不进入 Actor。
    """

    def __init__(self, robot: RobotContract, *, physics_width: int = 128,
                 latent_width: int = 64, hidden_dim: int = 256,
                 privileged_dim: int = 0, init_std: float = 1.0,
                 check_values: bool = True) -> None:
        """功能：建立 MLP Actor 和独立 Critic，并保留 G1/K1 各自的输出维数。

        输入：机器人合同；DynamicsModel 的 physics_width/latent_width；
              privileged_dim 为 Critic 专用额外状态维数，Actor 看不到。
        输出：可训练 Actor/Critic。不同机器人分别创建一个实例。
        """
        super().__init__()
        if not isinstance(robot, RobotContract):
            raise TypeError("robot 必须是 RobotContract")
        if type(privileged_dim) is not int or privileged_dim < 0:
            raise ValueError("privileged_dim 必须为非负整数")
        if not isinstance(init_std,(int,float)) or isinstance(init_std,bool) or not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std 必须是正有限实数")
        if any(type(v) is not int or v < 2 for v in (physics_width, latent_width, hidden_dim)):
            raise ValueError("physics_width、latent_width、hidden_dim 必须是>=2的整数")
        if type(check_values) is not bool:
            raise TypeError("check_values 必须是 bool")
        self.hidden_dim = hidden_dim
        self.robot, self.privileged_dim = robot, privileged_dim
        self.physics_width, self.latent_width = physics_width, latent_width
        self.check_values=check_values
        self.goal_encoder=GoalEncoder(robot,width=physics_width,hidden_dim=hidden_dim,
                                      check_values=check_values)
        in_dim=2*physics_width+latent_width
        self.actor_net=nn.Sequential(_mlp(in_dim,hidden_dim,hidden_dim),nn.SiLU())
        self.mean_head=nn.Linear(hidden_dim,robot.dof)
        nn.init.normal_(self.mean_head.weight,mean=0.0,std=0.01)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std=nn.Parameter(torch.full((robot.dof,),math.log(init_std)))
        # Critic 的训练损失不会向 Actor 主干或 DynamicsModel 反传。
        self.value_net=nn.Sequential(_mlp(in_dim+privileged_dim,hidden_dim,hidden_dim),
                                     nn.SiLU(),nn.Linear(hidden_dim,1))

    def get_extra_state(self) -> dict:
        """保存身体和目标接口定义，而不仅保存同尺寸权重。

        输入：无。输出：随 state_dict 保存的字典；旧版缺此字段必须显式迁移。
        """
        return dict(schema=3, robot=(self.robot.root_name, self.robot.joint_names),
                    widths=(self.physics_width, self.latent_width, self.hidden_dim),
                    privileged_dim=self.privileged_dim, goal="motion_contact_object_v1",
                    input_normalization=NORMALIZATION_SCHEMA)

    def set_extra_state(self, state: dict) -> None:
        """核对载入权重所属的机器人和特征接口；不一致就拒绝。

        输入：checkpoint 里的定义字典。输出：无；绝不按同样列数猜关节语义。
        """
        if state != self.get_extra_state():
            raise ValueError("MLP 策略 checkpoint 的机器人或输入定义不匹配")

    def forward(self, physics: PhysicsObservation, dynamics: DynamicsEncoding,
                goal: ControlGoal, *, privileged: torch.Tensor | None = None,
                detach_dynamics: bool = True) -> PhysicsPolicyOutput:
        """功能：使用当前物理状态、动力学适应结果与目标，计算动作和价值。

        输入：physics 是 [B,...] 的当前观测；dynamics 的 latent [B,Z] 和 context [B,D]；
              goal 可以是完整动作或接触/物体目标；privileged 只给 Critic。
        输出：mean/std [B,N] 与 value [B]，仍未真正执行到机器人。
        说明：这是单帧 Actor；需要 GRU 记忆时先调用 DynamicsModel.step()。
        """
        w=self.mean_head.weight
        lead=_check_observation(physics,self.robot,w.device,w.dtype,self.check_values)
        if len(lead) not in (1,2):
            raise ValueError("MLP Actor.forward 只接受 [B,...] 或 [B,T,...]")
        _check_tensor("dynamics.context",dynamics.context,(*lead,self.physics_width),
                      w.device,w.dtype,check_values=self.check_values)
        _check_tensor("dynamics.latent",dynamics.latent,(*lead,self.latent_width),
                      w.device,w.dtype,check_values=self.check_values)
        if self.privileged_dim:
            _check_tensor("privileged",privileged,(*lead,self.privileged_dim),
                          w.device,w.dtype,check_values=self.check_values)
        elif privileged is not None:
            raise ValueError("没有开启 Critic 特权观测，不能额外传入 privileged")
        task=self.goal_encoder(physics,goal)
        # PPO 训练初期可以冻结独立动力学模型；它自己的预测损失单独训练。
        phy = dynamics.context.detach() if detach_dynamics else dynamics.context
        latent=dynamics.latent.detach() if detach_dynamics else dynamics.latent
        shared=torch.cat((phy,latent,task),dim=-1)
        mean=self.mean_head(self.actor_net(shared))
        std=self.log_std.exp().expand_as(mean)
        if self.check_values and (not torch.isfinite(mean).all().item() or
                                  not torch.isfinite(std).all().item() or (std<=0).any().item()):
            raise ValueError("Actor 动作分布非法，不能交给 PPO 或环境")
        critic_input=torch.cat((shared.detach(),privileged),dim=-1) if self.privileged_dim else shared.detach()
        value=self.value_net(critic_input).squeeze(-1)
        if self.check_values and not torch.isfinite(value).all().item():
            raise ValueError("Critic 输出非有限值")
        return PhysicsPolicyOutput(mean,std,value,self.robot.joint_names)


class PhysicsMoEConditioner(nn.Module):
    """把独立动力学特征注入已经编码的动作状态，供旧 MoE 主干读取。

    功能：不重新实现 Expert，也不替换现有 HoloMotion 风格 Transformer。
    输入：EncodedMotion、DynamicsEncoding。
    输出：新的 EncodedMotion；只改变 state_token，不碰 reference_tokens。
    注意：旧 MultiRobotActorCritic 当前在内部自行调用 Encoder，尚无插入接口。
          正式接入 MoE 时需要对 Actor 的调用链做一次明确的 v2 接口升级。
          它不是已经完成的可训练 MoE-PPO 闭环。
    """

    def __init__(self, d_model: int, physics_width: int,
                 latent_width: int, hidden_dim: int = 256) -> None:
        """功能：创建输入宽度对齐的小网络。

        输入：原 Transformer 宽度、物理特征宽度、z 宽度、MLP 中间宽度。
        输出：可学习的物理条件化注入层。
        """
        super().__init__()
        self.d_model=d_model
        self.map=_mlp(physics_width+latent_width,hidden_dim,d_model)
        # 从零开始，不改变旧策略在初始化时的原有特征。
        nn.init.zeros_(self.map[-1].weight)
        nn.init.zeros_(self.map[-1].bias)

    def forward(self, encoded: EncodedMotion,
                dynamics: DynamicsEncoding) -> EncodedMotion:
        """功能：给已有的身体 token 增加物理条件，不重复融合参考动作。

        输入：encoded.state_token [...,D]，dynamics.context [...,P]、latent [...,Z]。
        输出：新的 EncodedMotion，形状与旧版完全相同。
        说明：初始输出和旧版相同，之后可由训练逐渐学会利用物理信息。
        """
        x=encoded.state_token
        if x.shape[:-1]!=dynamics.context.shape[:-1] or x.shape[:-1]!=dynamics.latent.shape[:-1]:
            raise ValueError("身体状态与动力学特征的环境/时间轴不同")
        if x.shape[-1]!=self.d_model:
            raise ValueError("EncodedMotion 的隐藏维数与 Conditioner 不一致")
        residual=self.map(torch.cat((dynamics.context,dynamics.latent),dim=-1))
        return EncodedMotion(encoded.robot_id,x+residual,
                             encoded.reference_tokens,encoded.reference_mask,
                             encoded.router_context)
