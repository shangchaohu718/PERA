"""时序物理训练：把任务、独立动力学模型和三类策略接成同一个闭环。

这是第17阶段的正式入口，取代“MLP 单帧 PPO”去训练 Transformer 的错误用法。
输入来自真实任务适配器，不读取或生成任何隐藏的示范数据。

时间约定：
    o_t -> 动力学 z_t -> 策略采样 a_t -> 环境执行 -> 真实 o_(t+1)
    -> 奖励与 final-state 价值 -> 保存经验 -> 必要时重置。
    预测下一状态用已决定的 a_t；不能提前把真实 o_(t+1) 喂给 Actor。

历史约定：
    Transformer 每层读取 C 帧、共 L 层，保留至少 L*(C-1) 个原始输入前缀。
    PPO 只打乱环境，不打乱时间；前缀参与计算，但不参与策略/价值损失。
    更新策略后重建 K/V；更新动力学后重放真实历史，重建 GRU 和预测，不重置环境。
    这是正确性优先的实现；没有宣称完成 GPU 吞吐优化或机器人控制验收。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Protocol, Sequence

import torch
from torch import nn

from .common import PPOConfig, compute_gae
from .dynamics_history import DynamicsHistory
from ..robot_contract import RobotContract
from ..reference_observations import ReferenceObservation
from ..models.physics import (
    ControlGoal, DynamicsEncoding, DynamicsMemory, DynamicsModel, DynamicsPrediction,
    PhysicsEncoding, PhysicsMLPActorCritic, PhysicsObservation, _core_state,
)
from ..models.physics_policy import PhysicsConditionedActorCritic, PhysicsSequenceOutput

Policy = PhysicsMLPActorCritic | PhysicsConditionedActorCritic


def clone_tree(value: Any) -> Any:
    """复制经验中的全部张量，切断梯度和仿真器共享内存。

    输入：张量、数据类、字典、元组或普通元信息。输出：同结构的独立快照。
    必须在环境下一次 step/reset 之前调用，否则旧经验可能被原地覆盖。
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: clone_tree(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    return copy.deepcopy(value)


def stack_tree(items: Sequence[Any], dim: int = 1) -> Any:
    """将 T 份同结构的单帧数据堆成 [B,T,...]。

    输入：非空列表；可选字段必须在这段序列中保持同一种结构。
    输出：保留字段名字的数据类。切换任务时用有效 mask，不临时删除某个字段。
    """
    if not items:
        raise ValueError("不能堆叠空经验")
    first = items[0]
    if any(type(x) is not type(first) for x in items):
        raise ValueError("序列中数据结构变化；用固定槽位及 mask 表示任务切换")
    if isinstance(first, torch.Tensor):
        return torch.stack(tuple(items), dim=dim)
    if is_dataclass(first):
        return type(first)(**{f.name: stack_tree([getattr(x, f.name) for x in items], dim)
                              for f in fields(first)})
    if first is None:
        return None
    if any(x != first for x in items):
        raise ValueError("序列中的非张量元信息不一致")
    return first


def select_tree(value: Any, env_ids: torch.Tensor) -> Any:
    """按第0维选择环境，保留环境内部完整时间顺序。

    输入：含 [B,...] 张量的数据类和 [K] 整数索引。输出：对应 [K,...] 数据。
    """
    if isinstance(value, torch.Tensor):
        return value.index_select(0, env_ids)
    if is_dataclass(value):
        return type(value)(**{f.name: select_tree(getattr(value, f.name), env_ids) for f in fields(value)})
    return value


def at_time(value: Any, t: int) -> Any:
    """从 [B,T,...] 数据树取第 t 帧 [B,...]；不改变原始数据。

    输入：时间序列数据类和有效时间编号。输出：单帧数据，供 MLP 或缓存重放。
    """
    if isinstance(value, torch.Tensor):
        return value[:, t]
    if is_dataclass(value):
        return type(value)(**{f.name: at_time(getattr(value, f.name), t) for f in fields(value)})
    return value


def slice_time(value: Any, start: int, stop: int | None) -> Any:
    """Slice a [B,T,...] tensor tree without changing its dataclass structure."""
    if isinstance(value, torch.Tensor):
        return value[:, start:stop]
    if is_dataclass(value):
        return type(value)(**{
            field.name: slice_time(getattr(value, field.name), start, stop)
            for field in fields(value)
        })
    return value


def parameter_stamp(model: nn.Module) -> tuple[int, ...]:
    """记录参数身份和原地修改版本，用来拒绝过期 rollout，而不是判断模型性能。

    输入：网络。输出：内存内使用的版本元组；不写入 checkpoint。
    optimizer.step()/load_state_dict() 会改变版本。禁止用 .data 偷改权重绕过检测。
    """
    return tuple(number for p in model.parameters() for number in (id(p), p._version))


@dataclass
class TaskTransition:
    """环境执行后的结果，不包含 reset 后观测。

    reward/terminated/truncated 均为 [B]。terms 是可选分项奖励字典。
    真正终止和外部时间截断不能同时为真；最终状态仍须由 task.read() 读取。
    """
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terms: dict[str, torch.Tensor] | None = None
    metrics: dict[str, torch.Tensor] | None = None


class PhysicsTask(Protocol):
    """采样器对环境的最小要求；mjlab 与 CPU 接口测评均实现这个协议。

    生产任务必须 auto_reset=False。模型不会根据这个接口自动获得滑板接触计划。
    """
    robot_id: str
    robot: RobotContract
    num_envs: int
    device: torch.device
    env_ids: tuple[int, ...]
    auto_reset: bool

    @property
    def episode_start(self) -> torch.Tensor:
        """输出 [B] bool：当前读到的状态是否是新回合第一帧。"""
        ...

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """输入待重置环境编号；输出无。None 重置全部，其他环境不能被移动。"""
        ...

    def read(self) -> PhysicsObservation:
        """无额外输入；输出当前 [B,...] 物理观测，不做步进或重置。"""
        ...

    def goal(self) -> ControlGoal:
        """无额外输入；输出当前时刻的合法目标，不能读取未来真实状态。"""
        ...

    def step(self, raw_action: torch.Tensor) -> TaskTransition:
        """输入策略顺序原始动作 [B,N]；输出执行结果，结束行暂不重置。"""
        ...


class MotionTrackingTask:
    """把现有物理桥、动作采样器和奖励接为 PhysicsTask，不重写它们。

    输入：physics_bridge/sampler/reward_fn。输出：供统一采样器使用的任务对象。
    目前真实物理桥仍是 K1；G1 的网络接口可测，但实物资产桥需要单独接入。
    """
    auto_reset = False

    def __init__(self, physics_bridge: Any, sampler: Any, reward_fn: Any,
                 *, scale_reward_by_dt: bool = False,
                 actor_observation: str = "full", privileged_learning: bool = False,
                 extended_contact_supervision: bool = False,
                 reference_state_init: bool = False,
                 random_reference_start: bool = False,
                 minimum_reference_remaining_s: float = 0.0) -> None:
        """保存已有模块并核对机器人、频率和奖励所需数据。

        scale_reward_by_dt=False 延续旧版外部奖励尺度；开启后总奖励和分项一起乘 dt。
        """
        self.physics_bridge, self.sampler, self.reward_fn = physics_bridge, sampler, reward_fn
        self.bridge = physics_bridge.bridge
        self.robot_id, self.robot = self.bridge.robot_id, self.bridge.robot
        self.num_envs, self.device, self.env_ids = self.bridge.num_envs, self.bridge.device, self.bridge.env_ids
        self.scale_reward_by_dt = scale_reward_by_dt
        if type(scale_reward_by_dt) is not bool:
            raise TypeError("scale_reward_by_dt 必须是 bool")
        if actor_observation not in ("full", "deployable"):
            raise ValueError("actor_observation必须是full或deployable")
        if type(privileged_learning) is not bool:
            raise TypeError("privileged_learning必须是bool")
        if type(extended_contact_supervision) is not bool:
            raise TypeError("extended_contact_supervision必须是bool")
        if extended_contact_supervision and not privileged_learning:
            raise ValueError("扩展接触监督只允许进入特权训练通道")
        if type(reference_state_init) is not bool:
            raise TypeError("reference_state_init必须是bool")
        if type(random_reference_start) is not bool:
            raise TypeError("random_reference_start必须是bool")
        if (isinstance(minimum_reference_remaining_s, bool)
                or not isinstance(minimum_reference_remaining_s, (int, float))
                or minimum_reference_remaining_s < 0):
            raise ValueError("minimum_reference_remaining_s必须是非负数")
        if actor_observation == "deployable" and not privileged_learning:
            raise ValueError("可部署参考策略需要特权Critic监督被屏蔽的物理状态")
        self.actor_observation = actor_observation
        self.privileged_learning = privileged_learning
        self.extended_contact_supervision = extended_contact_supervision
        self.reference_state_init = reference_state_init
        self.random_reference_start = random_reference_start
        self.minimum_reference_remaining_s = float(minimum_reference_remaining_s)
        if self.bridge.env.cfg.auto_reset is not False:
            raise ValueError("必须关闭自动 reset，保留最终观测")
        if sampler.library.robot != self.robot or reward_fn.robot != self.robot:
            raise ValueError("参考、奖励和仿真必须使用同一机器人合同")
        if sampler.num_envs != self.num_envs or sampler.library.device != self.device:
            raise ValueError("参考与仿真的环境数/设备不一致")
        if abs(sampler.step_dt - self.bridge.env.step_dt) > 1e-8:
            raise ValueError("参考与仿真的控制周期不一致")
        if abs(sampler.offsets_s[0].item()) > 1e-7:
            raise ValueError("第0个参考目标必须是当前时刻")
        if any(getattr(reward_fn.cfg, name) > 0 for name in (
            "foot_xy_weight", "foot_height_weight", "foot_rot_weight", "foot_slide_vel_weight",
            "foot_anchor_still_weight", "foot_contact_weight")):
            raise ValueError("当前任务未提供鞋底 site/接触阶段标签，不能开启脚部专项项")
        if any(getattr(reward_fn.cfg, name) > 0 for name in (
            "body_pos_weight", "body_rot_weight", "body_lin_vel_weight")) and not sampler.library.body_names:
            raise ValueError("身体奖励需要动作库的 body_* 数据")

    @property
    def episode_start(self) -> torch.Tensor:
        """输出桥接器的 [B] 新回合标志，不改变环境。"""
        return self.bridge.episode_start

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """重置选中的仿真环境，然后只给这些环境重新分配和对齐参考动作。

        输入：None 或 [K] 环境编号。输出：无；其他环境的轨迹时钟保持不变。
        """
        self.bridge.reset(env_ids)
        data = self.bridge.entity.data
        index = slice(None) if env_ids is None else env_ids
        self.sampler.reset(env_ids=env_ids, random_start=self.random_reference_start,
                           min_remaining_s=self.minimum_reference_remaining_s,
                           current_root_pos_w=data.root_link_pos_w[index],
                           current_root_quat_wxyz=data.root_link_quat_w[index])
        if self.reference_state_init:
            self._initialize_from_reference(env_ids)

    @torch.no_grad()
    def _initialize_from_reference(self, env_ids: torch.Tensor | None) -> None:
        """把 reset 后的物理状态同步到参考当前帧，避免从无关默认姿态硬追动作。"""
        ids = (torch.arange(self.num_envs, device=self.device, dtype=torch.long)
               if env_ids is None else torch.as_tensor(env_ids, device=self.device).long())
        if ids.numel() == 0:
            return
        reference = self.sampler.reference_window()
        joint_pos = reference.joint_pos.index_select(0, ids)[:, 0]
        joint_vel = reference.joint_vel.index_select(0, ids)[:, 0]
        root_pose = torch.cat((reference.root_pos_w.index_select(0, ids)[:, 0],
                              reference.root_quat_wxyz.index_select(0, ids)[:, 0]), -1)
        root_velocity = torch.cat((reference.root_lin_vel_w.index_select(0, ids)[:, 0],
                                  reference.root_ang_vel_w.index_select(0, ids)[:, 0]), -1)
        entity = self.bridge.entity
        simulator_pos = self.robot.to_simulator_order(joint_pos, self.bridge.joint_names)
        simulator_vel = self.robot.to_simulator_order(joint_vel, self.bridge.joint_names)
        entity.write_root_link_pose_to_sim(root_pose, env_ids=ids)
        entity.write_root_link_velocity_to_sim(root_velocity, env_ids=ids)
        entity.write_joint_state_to_sim(simulator_pos, simulator_vel, env_ids=ids)
        # env.reset() 已经 forward 过默认状态；传送到参考状态后必须重新计算运动学和传感器。
        self.bridge.env.sim.forward()
        self.bridge.env.sim.sense()

    def read(self) -> PhysicsObservation:
        """输出当前真实物理观测；快照复制由采样器统一负责。"""
        if self.actor_observation == "deployable":
            # This method does not touch the mjlab contact sensor.  In
            # particular, collision point/normal/force/slip cannot leak through
            # DynamicsModel into the Actor.
            return self.physics_bridge.read_deployable()
        return self.physics_bridge.read()

    @property
    def privileged_dim(self) -> int:
        """基础5维；可选增加每只脚的切向力、滑移速度和摩擦利用率共6维。"""
        if not self.privileged_learning:
            return 0
        return 11 if self.extended_contact_supervision else 5

    def privileged(self) -> torch.Tensor | None:
        """返回仅供Critic、系统辨识和Teacher使用的同帧仿真真值。"""
        if not self.privileged_learning:
            return None
        frame = self.physics_bridge.read()
        force = frame.contact_features[..., 9]
        # 100N只作为稳定数值尺度，不改变物理读数的来源。
        parts = [frame.base_lin_vel_b, force/100.0]
        if self.extended_contact_supervision:
            tangential_force, slip_speed, utilization = self.physics_bridge.contact_supervision()
            parts.extend((tangential_force/100.0, slip_speed, utilization.clamp_max(5.0)))
        value = torch.cat(parts, -1)
        if value.shape != (self.num_envs, self.privileged_dim) or not torch.isfinite(value).all().item():
            raise ValueError("参考任务privileged物理真值无效")
        return value

    def goal(self) -> ControlGoal:
        """把当前参考窗口转换为网络特征。输出 ControlGoal(motion=...)。"""
        data = self.bridge.entity.data
        ref = self.bridge.reference_builder(self.sampler.reference_window(),
              current_root_pos_w=data.root_link_pos_w, current_root_quat_wxyz=data.root_link_quat_w)
        return ControlGoal(motion=ref)

    def step(self, raw_action: torch.Tensor) -> TaskTransition:
        """执行一帧后再推进参考、计算奖励，绝不在这里 reset。

        输入：[B,N] 原始动作。输出：本步奖励与结束原因；片段结束是真正任务终止。
        """
        transition = self.bridge.step(raw_action)
        self.sampler.advance()
        body_reference = self.sampler.body_reference_window() if self.sampler.library.body_names else None
        result = self.reward_fn(self.physics_bridge.tracking_state(), self.sampler.reference_window(),
                 bodies=self.physics_bridge.body_tracking_state() if body_reference is not None else None,
                 body_reference=body_reference)
        terminated = transition.terminated | self.sampler.finished()
        truncated = transition.truncated & ~terminated
        scale = self.sampler.step_dt if self.scale_reward_by_dt else 1.0
        return TaskTransition(result.total * scale, terminated, truncated,
                              {k: v * scale for k, v in result.terms.items()},
                              {k: v.clone() for k, v in result.metrics.items()})


@dataclass
class PolicyInput:
    """一个控制时刻的冻结输入；用于重建策略历史，不保存 K/V 当作训练数据。

    physics/goal 为 [B,...]，context/latent/error 为 [B,P/Z/K]，starts 为 [B]。
    """
    physics: PhysicsObservation
    goal: ControlGoal
    context: torch.Tensor
    latent: torch.Tensor
    prediction_error: torch.Tensor
    starts: torch.Tensor
    privileged: torch.Tensor | None = None

    def dynamics(self) -> DynamicsEncoding:
        """组合冻结的动力学特征供策略使用；hidden 不被 Actor 读取。

        输出 DynamicsEncoding。真实 GRU 隐状态另存于 rollout.initial_hidden。
        """
        return DynamicsEncoding(self.latent, self.context, self.context.new_zeros((self.context.shape[0], 0)),
                                self.prediction_error)


@dataclass
class SequenceRollout:
    """一批已经采到的经验，保存原始动作、完整时间顺序和可选历史前缀。

    inputs/next_observation 是 [B,T,...]。reward/next_value/结束标志为 [T,B]，便于 GAE。
    prefix 是本批以前的真实输入，仅用于重算历史；没有对应的新 reward，不能参与 loss。
    consumed 防止同一份 on-policy 数据被重复更新。预测监督有独立的版本检查。
    """
    robot_id: str
    joint_names: tuple[str, ...]
    inputs: PolicyInput
    prefix: PolicyInput | None
    next_observation: PhysicsObservation
    action: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    next_value: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    initial_hidden: torch.Tensor
    previous_predicted_core: torch.Tensor
    previous_prediction_valid: torch.Tensor
    valid_transition: torch.Tensor
    reward_terms: dict[str, torch.Tensor]
    policy_stamp: tuple[int, ...]
    dynamics_stamp: tuple[int, ...]
    consumed: bool = False
    previous_interaction_prediction: DynamicsPrediction | None = None
    # PPO/log-prob always use ``action`` (the sampled command). Dynamics uses
    # this field when an actuator-delay queue executed a different command.
    executed_action: torch.Tensor | None = None
    dynamics_prefix: DynamicsHistory | None = None

    @property
    def horizon(self) -> int:
        """输出本批新采样的控制步数 T，不包含前缀。"""
        return self.action.shape[1]

    @property
    def num_envs(self) -> int:
        """输出并行环境数 B。"""
        return self.action.shape[0]

    @property
    def observation(self) -> PhysicsObservation:
        """输出本批的 [B,T,...] 物理历史，供独立动力学监督复用。"""
        return self.inputs.physics

    @property
    def episode_start(self) -> torch.Tensor:
        """输出 [B,T] 新回合标志。"""
        return self.inputs.starts


def _policy_contract(policy: Policy, robot_id: str) -> RobotContract:
    """输入策略及机器人ID；输出对应合同，避免同宽度但不同关节语义混用。"""
    if isinstance(policy, PhysicsMLPActorCritic):
        return policy.robot
    if isinstance(policy, PhysicsConditionedActorCritic) and robot_id in policy.robots:
        return policy.robots[robot_id]
    raise TypeError("策略必须是 PhysicsMLPActorCritic 或已注册该机器人的物理 Transformer")


def _history_size(policy: Policy) -> int | None:
    """输出策略需要保留的原始输入前缀长度，不包括 GRU 的历史。

    每层局部注意力向前延伸 C-1 帧，L 层需要 L*(C-1)，不能只保留一个窗口。
    MLP 无时序状态，返回0；GRU/LSTM返回None，要求保留真实回合历史。
    不能把循环隐状态当成一个有限Transformer窗口。
    """
    return 0 if isinstance(policy, PhysicsMLPActorCritic) else policy.history_requirement


def _join_input(prefix: PolicyInput | None, current: PolicyInput) -> PolicyInput:
    """将前缀与本批沿 T 轴连接，所有可选目标字段必须结构相同。

    输入：[B,P,...] 前缀或 None，以及 [B,T,...] 当前批；输出 [B,P+T,...]。
    """
    if prefix is None:
        return current
    def join(a: Any, b: Any) -> Any:
        if type(a) is not type(b):
            raise ValueError("跨批次目标结构变化；请用 mask 或显式 reset_all() 切换")
        if isinstance(a, torch.Tensor):
            return torch.cat((a, b), dim=1)
        if is_dataclass(a):
            return type(a)(**{f.name: join(getattr(a, f.name), getattr(b, f.name)) for f in fields(a)})
        return b
    return join(prefix, current)


def _check_history(starts: torch.Tensor, prefix_length: int, needed: int | None) -> None:
    """判断第一个参与 PPO 的时刻是否拥有足够前文。

    输入：[B,P+T] reset 标志、前缀长度、需要的前缀长度；输出无。
    历史不足时，必须能在已有前缀/当前首帧找到真实回合起点，否则停止。
    """
    if needed is not None and prefix_length >= needed:
        return
    if not starts[:, :prefix_length+1].any(dim=1).all().item():
        raise ValueError("时序 PPO 缺少足够历史：不能通过伪造 reset 或清零 K/V 继续")


class PhysicsSequenceCollector:
    """同一采样器支持 MLP / Dense / MoE，物理反馈来自传入的任务对象。

    输入：task、独立 dynamics、policy。输出：collect(horizon) 得到 SequenceRollout。
    更新策略后自动用前缀重建缓存；更新动力学后需显式 on_dynamics_updated()。
    """
    def __init__(self, task: PhysicsTask, dynamics: DynamicsModel, policy: Policy,
                 *, dynamics_history_steps: int = 16384, dynamics_history_bytes: int = 512*1024**2) -> None:
        """核对环境接口和模型宽度，不自动开始训练。

        输入：任务实例与两套网络。输出：可反复采样的对象。
        Critic 特权信息由可选 task.privileged() 返回，未配置时为 None。
        """
        if getattr(task, "auto_reset", None) is not False:
            raise ValueError("task 必须明确 auto_reset=False，不能丢失 final observation")
        if not isinstance(dynamics, DynamicsModel):
            raise TypeError("dynamics 必须是独立 DynamicsModel")
        if dynamics.robot != task.robot or _policy_contract(policy, task.robot_id) != task.robot:
            raise ValueError("任务、动力学与策略的机器人合同不一致")
        pwidth = policy.physics_width if isinstance(policy, PhysicsMLPActorCritic) else policy.cfg.physics_width
        zwidth = policy.latent_width if isinstance(policy, PhysicsMLPActorCritic) else policy.cfg.latent_width
        if (dynamics.physics_width, dynamics.latent_width) != (pwidth, zwidth):
            raise ValueError("独立动力学输出宽度与策略输入不一致")
        self.task, self.dynamics, self.policy = task, dynamics, policy
        self.robot, self.robot_id = task.robot, task.robot_id
        self.num_envs, self.device = task.num_envs, torch.device(task.device)
        self.env_ids = tuple(task.env_ids)
        if len(self.env_ids) != self.num_envs or len(set(self.env_ids)) != self.num_envs:
            raise ValueError("任务环境ID必须唯一且与批次一致")
        self._ready = False
        self._memory: DynamicsMemory | None = None
        self._cache = None
        self._history: list[PolicyInput] = []
        self._dynamics_stamp: tuple[int, ...] | None = None
        self._dynamics_history = DynamicsHistory(dynamics_history_steps, max_bytes=dynamics_history_bytes)

    @torch.no_grad()
    def reset_all(self) -> None:
        """开始新训练段：重置全部任务与两种记忆，不保留旧 episode 的状态。

        输入：无；输出：无。初次采样自动调用，研究中也可主动调用。
        """
        self.policy.eval(); self.dynamics.eval()
        self.task.reset()
        if not self.task.episode_start.all().item():
            raise ValueError("reset_all 后所有环境必须标记新回合")
        self._memory, self._cache = None, None
        self._history.clear()
        self._dynamics_history.clear()
        self._dynamics_stamp = parameter_stamp(self.dynamics)
        self._ready = True

    @torch.no_grad()
    def on_dynamics_updated(self) -> None:
        """用新参数重放真实观测和执行动作；不改变任务状态或回合边界。"""
        self.dynamics.invalidate_memories()
        if not self._ready:
            return
        # Old Actor inputs may precede resets that are already irrelevant to
        # every live row. Keep only the suffix covered by Dynamics replay.
        keep = min(len(self._history), len(self._dynamics_history.steps))
        history = self._history[-keep:] if keep else []
        memory, encodings = self._dynamics_history.rebuild(
            self.dynamics, self.env_ids, capture_last=keep)
        rebuilt = []
        for item, encoding in zip(history, encodings, strict=True):
            actor_dyn = (self.task.policy_dynamics(encoding, item.privileged)
                         if hasattr(self.task, "policy_dynamics") else encoding)
            rebuilt.append(replace(item, context=clone_tree(actor_dyn.context),
                latent=clone_tree(actor_dyn.latent),
                prediction_error=clone_tree(actor_dyn.prediction_error)))
        self._memory, self._history, self._cache = memory, rebuilt, None
        self._dynamics_stamp = parameter_stamp(self.dynamics)

    def _privileged(self) -> torch.Tensor | None:
        """无输入；从任务可选接口取得 Critic 专属 [B,Q] 状态并复制。"""
        fn = getattr(self.task, "privileged", None)
        return clone_tree(fn()) if fn is not None else None

    @torch.no_grad()
    def _actor_step(self, inp: PolicyInput, cache: Any):
        """执行策略单步，不执行物理步进。

        输入：当前冻结输入、上一个缓存；输出：分布/动作/log-prob/更新缓存。
        """
        if isinstance(self.policy, PhysicsMLPActorCritic):
            output = self.policy(inp.physics, inp.dynamics(), inp.goal, privileged=inp.privileged)
            action, lp = output.sample()
            return output, action, lp, None
        return self.policy.step(self.robot_id, inp.physics, inp.dynamics(), inp.goal,
               env_ids=self.env_ids, episode_start=inp.starts, cache=cache, privileged=inp.privileged)

    @torch.no_grad()
    def _rebuild_cache(self) -> None:
        """用当前策略权重重算保留的前缀；不采样动作、不推进环境。

        只在内部第一帧建立空缓存，RoPE 原点可以平移。真正 loss 时刻已有足够前文。
        """
        self._cache = None
        if isinstance(self.policy, PhysicsMLPActorCritic):
            return
        if self._history and _history_size(self.policy) is None:
            starts = torch.stack([item.starts for item in self._history], 1)
            if not starts.any(dim=1).all().item():
                raise ValueError("循环Actor前缀丢失真实回合起点，拒绝伪造记忆")
        for i, item in enumerate(self._history):
            # deterministic=True 避免重放历史时无意义地消耗动作采样随机数。
            starts = torch.ones_like(item.starts) if i == 0 else item.starts
            _, _, _, self._cache = self.policy.step(
                self.robot_id, item.physics, item.dynamics(), item.goal,
                env_ids=self.env_ids, episode_start=starts, cache=self._cache,
                privileged=item.privileged, deterministic=True)

    @torch.no_grad()
    def collect(self, horizon: int) -> SequenceRollout:
        """采集 T 步真实交互，返回可供 PPO 与动力学监督共用的经验。

        输入：正整数 horizon。输出 SequenceRollout；不进行参数更新。
        任何 final-state 价值都通过无缓存的独立 Critic 读取，不偷推进 Actor 记忆。
        """
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon 必须是正整数")
        self.policy.eval(); self.dynamics.eval()
        if not self._ready:
            self.reset_all()
        if parameter_stamp(self.dynamics) != self._dynamics_stamp:
            raise RuntimeError("动力学权重已更新；先调用 collector.on_dynamics_updated()")
        if tuple(self.task.env_ids) != self.env_ids:
            raise ValueError("环境行身份改变，禁止串用记忆")
        self._rebuild_cache()
        dynamics_prefix = self._dynamics_history.snapshot()
        prefix = stack_tree(self._history) if self._history else None
        policy_stamp = parameter_stamp(self.policy)
        inputs, next_frames, predictions, prediction_valid = [], [], [], []
        actions, executed_actions, means, stds, lps, values = [], [], [], [], [], []
        rewards, next_values, ends, cuts, metrics = [], [], [], [], []
        initial_hidden = None
        interaction_predictions = []
        pending_next_value = None
        pending_continue = None
        for step_index in range(horizon):
            frame = clone_tree(self.task.read())
            starts = clone_tree(self.task.episode_start)
            if starts.shape != (self.num_envs,) or starts.dtype != torch.bool:
                raise ValueError("episode_start 必须是 [B] bool")
            self._dynamics_history.check_capacity(starts, frame)
            before = self._memory
            pre_hidden = frame.body_obs.new_zeros((self.num_envs, self.dynamics.memory_width)) if before is None else before.hidden
            pre_prediction = frame.body_obs.new_zeros((self.num_envs, 9+2*self.robot.dof)) if before is None or before.predicted_core is None else before.predicted_core
            pre_valid = torch.zeros_like(starts) if before is None or before.prediction_valid is None else before.prediction_valid
            if initial_hidden is None:
                initial_hidden = clone_tree(pre_hidden)
            dyn, memory = self.dynamics.step(frame, episode_start=starts,
                       env_ids=self.env_ids, memory=before)
            privileged = self._privileged()
            actor_dyn = (self.task.policy_dynamics(dyn, privileged)
                         if hasattr(self.task, "policy_dynamics") else dyn)
            inp = PolicyInput(frame, clone_tree(self.task.goal()), clone_tree(actor_dyn.context),
                   clone_tree(actor_dyn.latent), clone_tree(actor_dyn.prediction_error), starts,
                   clone_tree(privileged))
            output, action, lp, self._cache = self._actor_step(inp, self._cache)
            # For a non-terminal MLP transition, V(o_{t+1}) is exactly the value already
            # produced for the next control action.  Fill the previous slot now and avoid a
            # second Dynamics+Actor forward on every environment step.
            if pending_next_value is not None:
                pending_next_value[pending_continue] = output.value[pending_continue]
                if not torch.isfinite(pending_next_value).all().item():
                    raise FloatingPointError("下一状态价值非有限")
                pending_next_value = pending_continue = None
            # 从这里以后，环境里的旧 view 可能被覆盖，以上输入已经全部做了独立快照。
            transition = self.task.step(action)
            if not isinstance(transition, TaskTransition):
                raise TypeError("task.step 必须返回 TaskTransition")
            for name, tensor in (("reward", transition.reward), ("terminated", transition.terminated),
                                 ("truncated", transition.truncated)):
                dtype = frame.body_obs.dtype if name == "reward" else torch.bool
                if not isinstance(tensor, torch.Tensor) or tensor.shape != (self.num_envs,) or tensor.device != self.device or tensor.dtype != dtype:
                    raise ValueError(f"{name} 必须为同设备、正确类型的 [B]")
            if not torch.isfinite(transition.reward).all().item():
                raise FloatingPointError("环境奖励含非有限值")
            if (transition.terminated & transition.truncated).any().item():
                raise ValueError("真正终止和时间截断不能同时为真")
            executed = getattr(self.task, "last_executed_action", action)
            if (not isinstance(executed, torch.Tensor) or executed.shape != action.shape
                    or executed.device != action.device or executed.dtype != action.dtype):
                raise ValueError("last_executed_action必须与策略动作同形状、同设备、同类型")
            executed = clone_tree(executed)
            self._dynamics_history.append(frame, starts, executed)
            predicted = self.dynamics.predict_next(
                frame, executed, dyn.latent, physics_encoding=dyn.physics_encoding())
            if self.dynamics.interaction_feedback:
                old_interaction = None if before is None else before.interaction_prediction
                if old_interaction is None:
                    old_interaction = clone_tree(predicted)
                    for field_name in ("contact_mask", "object_mask"):
                        getattr(old_interaction, field_name).zero_()
                interaction_predictions.append(clone_tree(old_interaction))
            after_action = self.dynamics.remember_prediction(memory, predicted)
            next_frame = clone_tree(self.task.read())
            if isinstance(self.policy, PhysicsMLPActorCritic):
                continuing = ~(transition.terminated | transition.truncated)
                need_value_now = transition.truncated.clone()
                if step_index == horizon - 1:
                    need_value_now |= continuing
                next_v = output.value.new_zeros((self.num_envs,))
                if need_value_now.any().item():
                    next_dyn, _ = self.dynamics.step(
                        next_frame, episode_start=torch.zeros_like(starts),
                        env_ids=self.env_ids, memory=after_action)
                    next_privileged = self._privileged()
                    next_actor_dyn = (self.task.policy_dynamics(next_dyn, next_privileged)
                                      if hasattr(self.task, "policy_dynamics") else next_dyn)
                    terminal_value = self.policy(
                        next_frame, next_actor_dyn, clone_tree(self.task.goal()),
                        privileged=next_privileged).value
                    next_v[need_value_now] = terminal_value[need_value_now]
            else:
                next_dyn, _ = self.dynamics.step(next_frame, episode_start=torch.zeros_like(starts),
                             env_ids=self.env_ids, memory=after_action)
                next_goal = clone_tree(self.task.goal())
                next_privileged = self._privileged()
                next_actor_dyn = (self.task.policy_dynamics(next_dyn, next_privileged)
                                  if hasattr(self.task, "policy_dynamics") else next_dyn)
                next_v = self.policy.value(self.robot_id, next_frame, next_actor_dyn, next_goal,
                                           privileged=next_privileged)
            next_v = torch.where(transition.terminated, 0.0, next_v)
            if not torch.isfinite(next_v).all().item():
                raise FloatingPointError("最终状态价值非有限")
            inputs.append(inp); next_frames.append(next_frame)
            predictions.append(clone_tree(pre_prediction)); prediction_valid.append(clone_tree(pre_valid))
            actions.append(clone_tree(action));executed_actions.append(executed)
            means.append(clone_tree(output.mean)); stds.append(clone_tree(output.std))
            lps.append(clone_tree(lp)); values.append(clone_tree(output.value))
            rewards.append(clone_tree(transition.reward)); next_values.append(clone_tree(next_v))
            if isinstance(self.policy, PhysicsMLPActorCritic) and step_index < horizon - 1:
                pending_next_value = next_values[-1]
                pending_continue = continuing
            ends.append(clone_tree(transition.terminated)); cuts.append(clone_tree(transition.truncated))
            metrics.append(clone_tree(transition.terms or {}))
            self._history.append(inp)
            keep = _history_size(self.policy)
            if keep is None:
                # GRU/LSTM的记忆可依赖整个回合，不能按Transformer窗口截断。
                # 只删除“每个环境最近一次真实reset”之前都已无用的共同前文。
                history_starts = torch.stack([x.starts for x in self._history], 1)
                indices = torch.arange(len(self._history), device=self.device)
                last = torch.where(history_starts, indices, -1).amax(dim=1)
                discard = int(last.min().item())
                if discard > 0:
                    self._history = self._history[discard:]
                if len(self._history) > self.policy.cfg.max_replay_steps:
                    self._ready = False
                    raise RuntimeError("循环Actor历史超过max_replay_steps；增加显式上限或缩短真实回合，不能悄悄截断")
            else:
                self._history = self._history[-keep:] if keep else []
            # 保留的是处理 o_t 后且预测了 o_(t+1) 的记忆；下轮再消费 o_(t+1)，不处理两遍。
            self._memory = after_action
            done = transition.terminated | transition.truncated
            if done.any().item():
                self.task.reset(done.nonzero(as_tuple=False).flatten())
        if pending_next_value is not None:
            raise RuntimeError("MLP next_value 延迟填充未完成")
        if policy_stamp != parameter_stamp(self.policy) or self._dynamics_stamp != parameter_stamp(self.dynamics):
            raise RuntimeError("采样期间网络权重被外部修改，整批数据作废")
        keys = set(metrics[0])
        if any(set(item) != keys for item in metrics):
            raise ValueError("奖励分项名字在一批内部变化，不能悄悄丢掉诊断")
        result = SequenceRollout(
            self.robot_id, self.robot.joint_names, stack_tree(inputs), prefix, stack_tree(next_frames),
            torch.stack(actions, 1), torch.stack(means, 1), torch.stack(stds, 1),
            torch.stack(lps, 1), torch.stack(values, 1), torch.stack(rewards), torch.stack(next_values),
            torch.stack(ends), torch.stack(cuts), initial_hidden, torch.stack(predictions, 1),
            torch.stack(prediction_valid, 1), torch.ones((self.num_envs, horizon), dtype=torch.bool, device=self.device),
            {key: torch.stack([item[key] for item in metrics]) for key in keys},
            policy_stamp, self._dynamics_stamp,
        )
        if interaction_predictions:
            result.previous_interaction_prediction = stack_tree(interaction_predictions)
        result.executed_action = torch.stack(executed_actions, 1)
        result.dynamics_prefix = dynamics_prefix
        return result


@dataclass(frozen=True)
class SequencePPOStats:
    """一次 PPO 更新的统计；old_policy_error 是更新前复算误差，不是运动误差。"""
    optimizer_steps: int
    early_stopped: bool
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    clip_fraction: float
    grad_norm: float
    old_policy_error: float


class PhysicsSequencePPO:
    """MLP、Dense、MoE 共用的 PPO 更新器，禁止打乱真实时间。

    输入：策略网络和 PPOConfig；update 输入 SequenceRollout，输出统计并更新策略。
    独立动力学不归此优化器管理；PPO 使用采样时冻结的 context/latent。
    """
    def __init__(self, policy: Policy, config: PPOConfig | None = None) -> None:
        """创建策略优化器。输入模型及可选配置；输出无。"""
        if not isinstance(policy, (PhysicsMLPActorCritic, PhysicsConditionedActorCritic)):
            raise TypeError("不支持的策略类型")
        self.policy, self.config = policy, config or PPOConfig()
        if not isinstance(self.config, PPOConfig):
            raise TypeError("config 必须是 PPOConfig")
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate, eps=self.config.adam_eps)
        self.base_learning_rate = self.config.learning_rate
        self.total_updates = 0

    def set_learning_rate_scale(self, scale: float) -> float:
        """按外层训练进度设置 PPO 学习率，返回实际学习率。"""
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1:
            raise ValueError("学习率倍率必须在(0,1]")
        value = self.base_learning_rate * float(scale)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def _evaluate(self, rollout: SequenceRollout, ids: torch.Tensor):
        """用当前权重与真实前缀重算动作分布，返回当前批的输出（不含前缀）。

        输入：rollout 和环境索引 [K]。输出：log_prob/value/entropy/mean/std/balance。
        """
        current = select_tree(rollout.inputs, ids)
        prefix = select_tree(rollout.prefix, ids) if rollout.prefix is not None else None
        prefix_len = 0 if prefix is None else prefix.starts.shape[1]
        seq = _join_input(prefix, current)
        _check_history(seq.starts, prefix_len, _history_size(self.policy))
        if isinstance(self.policy, PhysicsMLPActorCritic):
            # MLP has no temporal state: evaluate [K,T,...] in one batched forward instead of
            # launching 32 small GPU graphs from a Python loop for a typical rollout.
            output = self.policy(current.physics, current.dynamics(), current.goal,
                                 privileged=current.privileged)
            mean, std, values = output.mean, output.std, output.value
            balance = mean.new_zeros(())
        else:
            balance_mask = torch.zeros_like(seq.starts)
            balance_mask[:, prefix_len:] = True
            output = self.policy.forward_sequence(rollout.robot_id, seq.physics, seq.dynamics(), seq.goal,
                     episode_start=seq.starts, router_balance_mask=balance_mask,
                     privileged=seq.privileged, require_fresh_start=False)
            mean, std, values = (x[:, prefix_len:] for x in (output.mean, output.std, output.values))
            balance = output.balance_loss
        distribution = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
        actions = rollout.action.index_select(0, ids)
        return distribution.log_prob(actions), values, distribution.entropy(), mean, std, balance

    @torch.no_grad()
    def verify_old_policy(self, rollout: SequenceRollout) -> float:
        """在修改任何权重前核对同一个动作的概率、均值、标准差和价值。

        输入：刚采到的 rollout；输出：最大绝对复算误差。超限则拒绝训练。
        """
        error = 0.0
        for ids in torch.arange(rollout.num_envs, device=rollout.action.device).split(self.config.minibatch_envs):
            lp, value, _, mean, std, _ = self._evaluate(rollout, ids)
            for new, old, atol in (
                (lp, rollout.old_log_prob, self.config.logprob_atol),
                (value, rollout.old_value, self.config.value_atol),
                (mean, rollout.old_mean, self.config.distribution_atol),
                (std, rollout.old_std, self.config.distribution_atol)):
                e = (new - old.index_select(0, ids)).abs().max().item()
                if not torch.isfinite(new).all().item() or e > atol:
                    raise RuntimeError(f"旧策略复算失败（误差 {e:g}）；检查历史、目标和原始 action")
                error = max(error, e)
        return error

    def update(self, rollout: SequenceRollout) -> SequencePPOStats:
        """执行一次 on-policy PPO。前缀只提供上下文，不重复计算奖励或优势。

        输入：一份未消费且对应当前权重的 rollout。输出：更新统计。
        """
        if not isinstance(rollout, SequenceRollout) or rollout.consumed:
            raise ValueError("rollout 类型错误或已经消费")
        robot = _policy_contract(self.policy, rollout.robot_id)
        if robot.joint_names != rollout.joint_names:
            raise ValueError("rollout 的关节顺序与策略不一致")
        if parameter_stamp(self.policy) != rollout.policy_stamp:
            raise RuntimeError("rollout 来自旧策略或其他模型，不允许重复 on-policy 更新")
        self.policy.eval()
        error = self.verify_old_policy(rollout) if self.config.verify_old_policy else 0.0
        adv, ret = compute_gae(rollout.reward, rollout.old_value.transpose(0, 1), rollout.next_value,
                   rollout.terminated, rollout.truncated, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        adv, ret = adv.transpose(0, 1).detach(), ret.transpose(0, 1).detach()
        if self.config.normalize_advantage:
            adv = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        self.policy.train()
        rollout.consumed = True
        totals, updates, early = [0.0]*6, 0, False
        try:
            for _ in range(self.config.epochs):
                order = torch.randperm(rollout.num_envs, device=rollout.action.device)
                for ids in order.split(self.config.minibatch_envs):
                    lp, value, entropy, mean, std, balance = self._evaluate(rollout, ids)
                    old_lp = rollout.old_log_prob.index_select(0, ids)
                    ratio = (lp-old_lp).exp()
                    if not torch.isfinite(ratio).all().item():
                        raise FloatingPointError("PPO 概率比溢出，拒绝更新")
                    a, returns, old_v = adv[ids], ret[ids], rollout.old_value[ids]
                    policy_loss = -torch.minimum(ratio*a, ratio.clamp(1-self.config.clip_ratio, 1+self.config.clip_ratio)*a).mean()
                    sq = (value-returns).square()
                    if self.config.value_clip is not None:
                        clipped = old_v + (value-old_v).clamp(-self.config.value_clip, self.config.value_clip)
                        sq = torch.maximum(sq, (clipped-returns).square())
                    vloss, ent = 0.5*sq.mean(), entropy.mean()
                    old_mean, old_std = rollout.old_mean[ids], rollout.old_std[ids]
                    kl = (std.log()-old_std.log() + (old_std.square()+(old_mean-mean).square())/(2*std.square()) - 0.5).sum(-1).mean().clamp_min(0)
                    if not torch.isfinite(kl).item():
                        raise FloatingPointError("解析 KL 非有限")
                    if self.config.target_kl is not None and kl.item() > self.config.target_kl:
                        early = True; break
                    loss = policy_loss + self.config.value_coef*vloss - self.config.entropy_coef*ent + self.config.balance_coef*balance
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("PPO 总损失非有限")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
                    self.optimizer.step()
                    if hasattr(self.policy, "invalidate_caches"):
                        self.policy.invalidate_caches()
                    clip = ((ratio-1).abs()>self.config.clip_ratio).float().mean()
                    numbers = (policy_loss, vloss, ent, kl, clip, grad)
                    totals = [v + n.detach().item() for v, n in zip(totals, numbers)]
                    self.total_updates += 1; updates += 1
                if early:
                    break
        finally:
            # 即使更新失败，也不让旧运行时缓存继续使用。
            if hasattr(self.policy, "invalidate_caches"):
                self.policy.invalidate_caches()
            self.policy.eval()
        return SequencePPOStats(updates, early, *(v/max(updates, 1) for v in totals), error)

    def state_dict(self) -> dict:
        """保存策略、Adam 状态和计数。输出可交给 torch.save 的字典。

        不保存环境或 rollout；加载后必须重新采样，不能声称从同一个物理状态精确续跑。
        """
        return dict(schema=1, config=asdict(self.config), model=copy.deepcopy(self.policy.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()), updates=self.total_updates)

    def load_state_dict(self, state: dict) -> None:
        """恢复策略优化状态，并核对算法配置。输入 checkpoint 字典；输出无。"""
        if not isinstance(state, dict) or state.get("schema") != 1 or state.get("config") != asdict(self.config):
            raise ValueError("PPO checkpoint 的版本或配置不匹配")
        if type(state.get("updates")) is not int or state["updates"] < 0:
            raise ValueError("更新计数非法")
        self.policy.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_updates = state["updates"]
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(self.policy, "invalidate_caches"):
            self.policy.invalidate_caches()


@dataclass(frozen=True)
class DynamicsUpdateStats:
    """独立动力学一次监督学习的损失、梯度与累计更新次数，不是控制成功率。"""
    loss: float
    grad_norm: float
    optimizer_steps: int
    terms: dict[str, float]


def _core_prediction_terms(predicted: torch.Tensor, target: PhysicsObservation,
                           valid: torch.Tensor, robot: RobotContract,
                           prefix: str) -> dict[str, torch.Tensor]:
    """Compute the same normalized core losses for an auxiliary future horizon."""
    true = _core_state(target, robot)
    if predicted.shape != true.shape or valid.shape != true.shape[:-1]:
        raise ValueError("多步预测、目标和有效mask形状不一致")
    n = robot.dof
    result = {}
    for name, begin, end, scale in (
        ("base_linear_velocity",0,3,1.0),
        ("base_angular_velocity",3,6,1.0),
        ("gravity_direction",6,9,1.0),
        ("joint_position",9,9+n,0.3),
        ("joint_velocity",9+n,9+2*n,3.0),
    ):
        error = ((predicted[...,begin:end]-true[...,begin:end])/scale).square()
        mask = target.core_valid[..., begin:end] & valid.unsqueeze(-1)
        result[f"{prefix}_{name}"] = torch.where(mask, error, 0.).sum()/mask.sum().clamp_min(1)
    return result


@torch.no_grad()
def replay_dynamics_inputs(model, prefix, observation, episode_start, actions):
    """Refresh truncated-BPTT inputs under current weights before every epoch."""
    ids = tuple(range(actions.shape[0]))
    memory, _ = prefix.rebuild(model, ids)
    initial = (actions.new_zeros((actions.shape[0], model.memory_width))
               if memory is None else memory.hidden.clone())
    cores, valid, interactions = [], [], []
    from ..models.physics import PhysicsEncoding
    physics = model.encoder(observation)
    for step in range(actions.shape[1]):
        frame = at_time(observation, step)
        starts = episode_start[:, step]
        cores.append(actions.new_zeros((actions.shape[0], 9+2*model.robot.dof))
                     if memory is None else memory.predicted_core.clone())
        valid.append(torch.zeros_like(starts) if memory is None else memory.prediction_valid.clone())
        before = memory
        encoded = PhysicsEncoding(physics.context[:, step], physics.contact_tokens[:, step],
                                   physics.object_tokens[:, step])
        encoding, memory = model.step(frame, episode_start=starts, env_ids=ids, memory=memory,
                                      physics_encoding=encoded)
        prediction = model.predict_next(frame, actions[:, step], encoding.latent,
                                        physics_encoding=encoding.physics_encoding())
        if model.interaction_feedback:
            previous = None if before is None else before.interaction_prediction
            if previous is None:
                previous = clone_tree(prediction)
                previous.contact_mask.zero_(); previous.object_mask.zero_()
            interactions.append(previous)
        memory = model.remember_prediction(memory, prediction)
    return (initial, torch.stack(cores, 1), torch.stack(valid, 1),
            stack_tree(interactions) if interactions else None)


class DynamicsUpdater:
    """使用真实 (o_t,a_t,o_(t+1)) 训练物理预测器，不更新策略。

    输入独立 DynamicsModel；update 使用同一份 SequenceRollout。
    一次更新后旧 GRU 起始状态已经过期；重复监督前须重新构建历史或重新采样。
    """
    def __init__(self, model: DynamicsModel, *, learning_rate: float = 1e-4,
                 max_grad_norm: float = 1.0, system_id_weight: float = 0.0,
                 teacher_distillation_weight: float = 0.0,
                 teacher_reconstruction_weight: float = 0.0,
                 multi_step_weight: float = 0.0,
                 latent_slow_weight: float = 0.0,
                 epochs: int = 4) -> None:
        """建立独立 Adam。输入模型和正数超参数；输出无。"""
        import math
        if not isinstance(model, DynamicsModel):
            raise TypeError("model 必须是 DynamicsModel")
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0
               for x in (learning_rate, max_grad_norm)):
            raise ValueError("学习率和梯度上限必须是正的有限数")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("Dynamics epochs必须是正整数")
        if (isinstance(system_id_weight, bool) or not isinstance(system_id_weight, (float, int))
                or not math.isfinite(system_id_weight) or system_id_weight < 0):
            raise ValueError("system_id_weight必须是非负有限数")
        if bool(model.system_id_dim) != bool(system_id_weight):
            raise ValueError("系统辨识头与system_id_weight必须同时开启或关闭")
        for name, value in (("teacher_distillation_weight",teacher_distillation_weight),
                            ("teacher_reconstruction_weight",teacher_reconstruction_weight),
                            ("multi_step_weight",multi_step_weight),
                            ("latent_slow_weight",latent_slow_weight)):
            if (isinstance(value,bool) or not isinstance(value,(float,int))
                    or not math.isfinite(value) or value<0):
                raise ValueError(f"{name}必须是非负有限数")
        from .dynamics_objective import validate_teacher_weights
        validate_teacher_weights(model, teacher_reconstruction_weight, teacher_distillation_weight)
        if bool(multi_step_weight) != (len(model.prediction_horizons) > 1):
            raise ValueError("多步预测头和multi_step_weight必须同时开启或关闭")
        self.model, self.max_grad_norm = model, float(max_grad_norm)
        self.system_id_weight = float(system_id_weight)
        self.teacher_distillation_weight=float(teacher_distillation_weight)
        self.teacher_reconstruction_weight=float(teacher_reconstruction_weight)
        self.multi_step_weight=float(multi_step_weight)
        self.latent_slow_weight=float(latent_slow_weight)
        self.epochs=epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.base_learning_rate = float(learning_rate)
        self.steps = 0

    def set_learning_rate_scale(self, scale: float) -> float:
        """按外层训练进度设置动力学学习率，返回实际学习率。"""
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1:
            raise ValueError("学习率倍率必须在(0,1]")
        value = self.base_learning_rate * float(scale)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def update(self, rollout: SequenceRollout) -> DynamicsUpdateStats:
        """在同一批真实序列上复习 ``epochs`` 次，并返回各步平均损失。"""
        if (not isinstance(rollout, SequenceRollout)
                or parameter_stamp(self.model) != rollout.dynamics_stamp):
            raise RuntimeError("动力学监督需要对应当前模型的采样历史；旧 hidden 不能冒充新模型状态")
        if (self.epochs > 1 and rollout.dynamics_prefix is None
                and not rollout.episode_start[:,0].all().item()):
            raise ValueError(
                "Dynamics多轮复习要求序列从真实回合起点开始；"
                "请把dynamics_every设为1或把epochs设为1"
            )
        results = []
        for _ in range(self.epochs):
            if rollout.dynamics_prefix is not None or rollout.episode_start[:, 0].all().item():
                prefix = rollout.dynamics_prefix or DynamicsHistory()
                initial, core, valid, interaction = replay_dynamics_inputs(
                    self.model, prefix, rollout.observation,
                    rollout.episode_start,
                    rollout.action if rollout.executed_action is None else rollout.executed_action)
                current = replace(rollout, initial_hidden=initial,
                    previous_predicted_core=core, previous_prediction_valid=valid,
                    previous_interaction_prediction=interaction)
            else:
                current = rollout
            results.append(self._update_once(current))
        keys = set(results[0].terms)
        if any(set(item.terms) != keys for item in results):
            raise RuntimeError("Dynamics多轮复习的损失项发生变化")
        return DynamicsUpdateStats(
            sum(item.loss for item in results) / self.epochs,
            sum(item.grad_norm for item in results) / self.epochs,
            self.steps,
            {key: sum(item.terms[key] for item in results) / self.epochs for key in keys},
        )

    def _update_once(self, rollout: SequenceRollout) -> DynamicsUpdateStats:
        """更新物理模型一次，使用采样前的真实 GRU 状态进行截断反向传播。

        输入：未因动力学权重改变而过期的 rollout。输出监督损失和梯度统计。
        下一状态必须是 reset 之前读到的那一帧；终止转移仍可用于物理监督。
        """
        self.model.train()
        try:
            encoding = self.model.forward_sequence(rollout.observation,
                episode_start=rollout.episode_start, initial_hidden=rollout.initial_hidden.detach(),
                previous_predicted_core=rollout.previous_predicted_core.detach(),
                previous_prediction_valid=rollout.previous_prediction_valid,
                previous_interaction_prediction=rollout.previous_interaction_prediction)
            prediction = self.model.predict_next(
                rollout.observation,
                (rollout.action if rollout.executed_action is None
                 else rollout.executed_action).detach(),
                encoding.latent,
                physics_encoding=encoding.physics_encoding())
            loss = self.model.prediction_loss(prediction, rollout.next_observation,
                                              valid_transition=rollout.valid_transition)
            total = loss.total
            terms = dict(loss.terms)
            if self.multi_step_weight:
                horizon_losses = []
                total_steps = rollout.horizon
                encoded_physics = encoding.physics_encoding()
                if encoded_physics is None:
                    raise RuntimeError("多步预测缺少已编码物理特征")
                for horizon in self.model.prediction_horizons[1:]:
                    length = total_steps-horizon
                    if length <= 0:
                        continue
                    source = slice_time(rollout.observation,0,length)
                    target_future = slice_time(rollout.observation,horizon,horizon+length)
                    dynamics_action=(rollout.action if rollout.executed_action is None
                                     else rollout.executed_action)
                    action_sequence = torch.stack(
                        [dynamics_action[:,offset:offset+length]
                         for offset in range(horizon)], dim=2).detach()
                    source_physics = PhysicsEncoding(
                        encoded_physics.context[:,:length],
                        encoded_physics.contact_tokens[:,:length],
                        encoded_physics.object_tokens[:,:length])
                    predicted_core = self.model.predict_future_core(
                        source, action_sequence, encoding.latent[:,:length],
                        horizon=horizon, physics_encoding=source_physics)
                    valid = torch.ones((rollout.num_envs,length),dtype=torch.bool,
                                       device=rollout.action.device)
                    for offset in range(horizon):
                        valid &= rollout.valid_transition[:,offset:offset+length]
                    for offset in range(1,horizon+1):
                        valid &= ~rollout.episode_start[:,offset:offset+length]
                    if not valid.any().item():
                        continue
                    future_terms = _core_prediction_terms(
                        predicted_core,target_future,valid,self.model.robot,f"h{horizon}")
                    terms.update(future_terms)
                    horizon_losses.append(sum(future_terms.values()))
                if horizon_losses:
                    multi_step_loss = torch.stack(horizon_losses).mean()
                    terms["multi_step"] = multi_step_loss
                    total = total + self.multi_step_weight*multi_step_loss
            if self.latent_slow_weight and rollout.horizon > 1:
                same_episode = ~rollout.episode_start[:,1:]
                delta = (encoding.latent[:,1:]-encoding.latent[:,:-1]).square().mean(-1)
                slow_loss = (delta*same_episode).sum()/same_episode.sum().clamp_min(1)
                terms["latent_slow"] = slow_loss
                total = total + self.latent_slow_weight*slow_loss
            if self.system_id_weight:
                target = rollout.inputs.privileged
                if target is None or target.shape != (*encoding.latent.shape[:-1], self.model.system_id_dim):
                    raise ValueError("系统辨识需要与latent时间轴对齐的privileged目标")
                # 至少观察过一次动作响应后才监督；回合首帧不能凭空辨识物理参数。
                from .dynamics_objective import identification_loss
                mask = rollout.previous_prediction_valid & rollout.valid_transition & ~rollout.episode_start
                identified, _, _ = identification_loss(self.model, encoding.latent, target, mask,
                    reconstruction_weight=self.teacher_reconstruction_weight,
                    distillation_weight=self.teacher_distillation_weight,
                    teacher_valid=rollout.valid_transition)
                terms.update(identified)
                total = (total + self.system_id_weight*identified["system_identification"]
                         + self.teacher_reconstruction_weight*identified["teacher_reconstruction"]
                         + self.teacher_distillation_weight*identified["teacher_distillation"])
            if not torch.isfinite(total).item():
                raise FloatingPointError("动力学损失非有限")
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step(); self.steps += 1
            return DynamicsUpdateStats(total.detach().item(), grad.detach().item(), self.steps,
                                       {k: v.detach().item() for k,v in terms.items()})
        finally:
            self.model.invalidate_memories(); self.model.eval()

    def state_dict(self) -> dict:
        """输出物理模型/优化器/计数快照；不包含运行时 GRU。"""
        return dict(schema=4, model=copy.deepcopy(self.model.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()), steps=self.steps,
                    epochs=self.epochs,
                    max_grad_norm=self.max_grad_norm, system_id_weight=self.system_id_weight,
                    teacher_distillation_weight=self.teacher_distillation_weight,
                    teacher_reconstruction_weight=self.teacher_reconstruction_weight,
                    multi_step_weight=self.multi_step_weight,
                    latent_slow_weight=self.latent_slow_weight)

    def load_state_dict(self, state: dict) -> None:
        """输入保存的监督 checkpoint；验证并恢复，废弃任何旧记忆。"""
        if (not isinstance(state, dict) or state.get("schema") != 4
                or state.get("epochs") != self.epochs
                or state.get("max_grad_norm") != self.max_grad_norm
                or state.get("system_id_weight") != self.system_id_weight
                or state.get("teacher_distillation_weight") != self.teacher_distillation_weight
                or state.get("teacher_reconstruction_weight") != self.teacher_reconstruction_weight
                or state.get("multi_step_weight",0.0) != self.multi_step_weight
                or state.get("latent_slow_weight",0.0) != self.latent_slow_weight
                or type(state.get("steps")) is not int or state["steps"] < 0):
            raise ValueError("动力学更新器 checkpoint 配置不匹配")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = state["steps"]
        self.optimizer.zero_grad(set_to_none=True)
        self.model.invalidate_memories()
