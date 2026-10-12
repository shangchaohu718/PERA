"""新代码统一从这里导入。避免再混用编号模块、v2 文件和历史 PPO。

本文件只提供公开入口与模型创建函数，不启动仿真，也不在导入时开始训练。
网络层仍复用原有 Attention、ReferenceFusion、MoE，不重新实现数学公式。
"""
from collections.abc import Mapping
import math

from .robot_contract import RobotContract, get_robot_contract, register_robot
from .observations import RobotState, ObservationBuilder
from .reference_observations import ReferenceObservation, ReferenceWindow, ReferenceObservationBuilder
from .data.motion_sampler import MotionClip, MotionLibrary, MotionSampler
from .rewards.motion_tracking import MotionTrackingReward, MotionTrackingRewardConfig
from .models.physics import (
    PhysicsObservation, ControlGoal, DynamicsEncoding, DynamicsMemory,
    DynamicsModel, PhysicsMLPActorCritic,
)
from .models.physics_policy import PhysicsPolicyConfig, PhysicsConditionedActorCritic
from .algorithms.common import PPOConfig, compute_gae
from .algorithms.sequence_training import (
    TaskTransition, PhysicsTask, MotionTrackingTask, SequenceRollout,
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater,
)


def build_policy(
    robots: Mapping[str, RobotContract], *, backbone: str = "mlp",
    d_model: int = 128, hidden_dim: int = 256, n_layers: int = 2,
    context_len: int = 32, num_experts: int = 4,
    physics_width: int = 128, latent_width: int = 64, init_std: float = 1.0,
    recurrent_hidden_dim: int | None = None, max_replay_steps: int = 4096,
    lru_state_dim: int | None = None, lru_r_min: float | None = None,
    lru_r_max: float | None = None, lru_max_phase: float | None = None,
    lru_scan: str | None = None,
    privileged_dims: Mapping[str, int] | None = None,
    check_values: bool = True,
):
    """创建六种策略之一：MLP、Transformer、MoE、GRU、LSTM、LRU，共用采样与PPO。

    输入：
        robots：{机器人ID: 关节合同}。MLP 当前一次只注册一款机器人；
                Dense/MoE 可注册多款，但每次采样批次仍只包含一种机器人。
        backbone：mlp / transformer（旧名dense）/ moe / gru / lstm / lru，每次只创建一种时间处理路线。
        d_model：时序策略的隐藏宽度，当前工厂固定 Q=4头/KV=2头，宽度须能被8整除。
        hidden_dim：输入编码器/MLP 的中间宽度，不是关节数量。
        n_layers：Transformer或循环Actor的层数；MLP不使用。
        context_len：仅用于Transformer。GRU/LSTM推理递推h/c，不受此窗口截断。
        recurrent_hidden_dim：循环Actor每层的记忆宽度，None使用d_model。
        max_replay_steps：循环PPO完整历史的保护上限，超限报错而不是默默忘记。
        lru_state_dim：LRU每层复数记忆宽度，None使用d_model；实际占2倍实数。
        lru_r_min/lru_r_max：只控制初始lambda半径，默认0.9/0.999。
        lru_max_phase：初始相位上限（弧度），默认2*pi。
        lru_scan：parallel（可反传前缀扫描）或sequential（逐帧对照）。
        GNN是独立DynamicsModel中的scene_encoder='gnn'，不算时间主干。
        num_experts：MoE 专家总数，固定每个 token 选择2个，至少需要2个专家。
        physics_width/latent_width：必须与 DynamicsModel 输出一致。
        init_std：原始动作高斯分布的初始标准差；不是关节角范围。
        check_values：模型内部逐层有限值检查；关闭可减少GPU同步，但训练器的
            reward、value、loss、KL和梯度非有限检查仍然保留。
    输出：PhysicsMLPActorCritic 或 PhysicsConditionedActorCritic，可交给统一 Collector/PPO。
    注意：这只是创建网络，随机权重没有运动技能；仿真动作缩放在环境中另行定义。
    """
    if not isinstance(robots, Mapping) or not robots:
        raise ValueError("robots 必须是非空映射")
    if any(not isinstance(k, str) or not k or not isinstance(v, RobotContract)
           for k, v in robots.items()):
        raise TypeError("robots 需要 {非空字符串: RobotContract}")
    if type(check_values) is not bool:
        raise TypeError("check_values 必须是 bool")
    if backbone == "transformer":
        backbone = "dense"
    lru_values = (lru_state_dim, lru_r_min, lru_r_max, lru_max_phase, lru_scan)
    if backbone != "lru" and any(v is not None for v in lru_values):
        raise ValueError("lru_*参数仅用于LRU；不能给Transformer配置后静默忽略")
    if backbone not in ("gru", "lstm") and recurrent_hidden_dim is not None:
        raise ValueError("recurrent_hidden_dim只用于GRU/LSTM，不能静默忽略")
    if backbone == "mlp":
        if len(robots) != 1:
            raise ValueError("MLP 基线一次只创建一种机器人；多身体共享主干使用 Dense/MoE")
        rid, robot = next(iter(robots.items()))
        dims = {} if privileged_dims is None else dict(privileged_dims)
        if set(dims) - set(robots):
            raise ValueError("privileged_dims 包含未注册的机器人")
        return PhysicsMLPActorCritic(robot, physics_width=physics_width,
                                    latent_width=latent_width, hidden_dim=hidden_dim,
                                    privileged_dim=dims.get(rid, 0), init_std=init_std,
                                    check_values=check_values)
    if backbone not in ("dense", "moe", "gru", "lstm", "lru"):
        raise ValueError("backbone 只能是 mlp、transformer/dense、moe、gru、lstm、lru")
    return PhysicsConditionedActorCritic(robots, PhysicsPolicyConfig(
        backbone=backbone, d_model=d_model, input_hidden_dim=hidden_dim, n_layers=n_layers,
        n_heads=4, n_kv_heads=2, context_len=context_len, dense_hidden_dim=hidden_dim,
        num_fine_experts=num_experts, num_shared_experts=1, top_k=2,
        expert_hidden_dim=hidden_dim, router_hidden_dim=d_model,
        physics_width=physics_width, latent_width=latent_width,
        critic_hidden_dim=hidden_dim, init_std=init_std,
        recurrent_hidden_dim=recurrent_hidden_dim, max_replay_steps=max_replay_steps,
        lru_state_dim=lru_state_dim, lru_r_min=.9 if lru_r_min is None else lru_r_min,
        lru_r_max=.999 if lru_r_max is None else lru_r_max,
        lru_max_phase=2*math.pi if lru_max_phase is None else lru_max_phase,
        lru_scan="parallel" if lru_scan is None else lru_scan,
        check_values=check_values,
    ), privileged_dims=privileged_dims)

# R19任务层：不在导入时启动仿真。
from .tasks.interaction import (
    ContactPair, RelationState, PlanTarget, PlanStage, ContactPlanTask,
    DomainScheduler, RandomizedTask,
)

# R20插口：低层模块可独立使用；正式训练仍走统一的build_policy。
from .models.recurrent import RecurrentCore, RecurrentState
from .models.graph_encoder import GraphBatch, MessagePassingEncoder, PhysicsRelationGraph

# R21：真正的线性循环插口，不是缓存淘汰算法。
from .models.lru import LRUCore, LRULayer, LRUState
