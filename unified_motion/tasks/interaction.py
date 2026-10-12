"""R19：通用接触关系、外部计划执行、物理课程。

核心约定：
    相同槽位的“身体部位—接触对象”关系在整个任务实例中固定。
    每步读取实际相对运动，不用参考动作伪造接触。
    本文件不实现自主规划器，也不把一个碰撞点的力当足底总力或CoP。
    Backend负责真实物理；PlanTask负责目标/奖励；RandomizedTask负责采样条件。
    全部任务手动reset，结束前的真实状态由Collector读取。
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from typing import Any, Protocol, Sequence

import torch

from ..algorithms.sequence_training import TaskTransition
from ..models.physics import ControlGoal, PhysicsObservation
from ..reference_observations import _quat_to_matrix


def rotate(q: torch.Tensor, v: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """输入单位wxyz姿态和同批向量，返回旋转后的向量；inverse=True为世界→部件。"""
    m = _quat_to_matrix(q)
    if inverse:
        m = m.transpose(-1, -2)
    while m.ndim < v.ndim + 1:
        m = m.unsqueeze(-3)
    return (m @ v.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True)
class ContactPair:
    """一条固定接触关系，而不是临时找到的第几个碰撞。

    name：关系唯一名。body：机器人部件名。object：目标场景实体名。
    object_body：目标实体中的部件名。point_local_m：机器人body中的固定测量点。
    输入来自YAML；输出用来创建传感器、排列观测和解析目标，不自行产生碰撞。
    """
    name: str
    body: str
    object: str = "terrain"
    object_body: str = "terrain"
    point_local_m: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        for name in ("name", "body", "object", "object_body"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"接触关系{name}不能为空")
        if len(self.point_local_m) != 3 or not all(math.isfinite(x) for x in self.point_local_m):
            raise ValueError("point_local_m必须为3个有限数字")


@dataclass
class RelationState:
    """从真实后端读取的同一时刻关系信息，均为世界坐标。

    root_pos/quaternion：[B,3/4]。point_pos/relative_velocity：[B,C,3]。
    object_pos/quaternion：[B,C,3/4]，每条关系对应的对象坐标系。
    found：[B,C] bool为真实接触检测；有效观测另由PhysicsObservation的mask表示。
    relative_velocity是两个刚体在同一个空间测量点的速度差，包含omega×r。
    """
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    point_pos: torch.Tensor
    relative_velocity: torch.Tensor
    object_pos: torch.Tensor
    object_quat: torch.Tensor
    found: torch.Tensor
    normal_force: torch.Tensor


@dataclass
class PhysicalRewardState:
    """只供奖励/Critic侧使用的仿真物理真值，不会拼进Actor观测。"""
    root_ang_vel_b: torch.Tensor
    com_pos_w: torch.Tensor
    joint_pos: torch.Tensor
    soft_joint_pos_limits: torch.Tensor
    undesired_contact: torch.Tensor


@dataclass(frozen=True)
class ContactRewardConfig:
    """目标驱动接触任务的密集物理奖励；所有系数显式可消融。"""
    support_weight: float = 0.0
    upright_weight: float = 0.0
    com_transfer_weight: float = 0.0
    action_rate_weight: float = 0.0
    joint_limit_weight: float = 0.0
    undesired_contact_weight: float = 0.0
    root_ang_vel_weight: float = 0.0
    foot_slide_weight: float = 0.0
    support_force_scale_n: float = 25.0
    upright_sigma_rad: float = 0.35
    com_sigma_m: float = 0.08

    def __post_init__(self):
        weights = tuple(getattr(self, f.name) for f in fields(self)
                        if f.name.endswith("_weight"))
        positive = (self.support_force_scale_n, self.upright_sigma_rad, self.com_sigma_m)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not math.isfinite(x) or x < 0 for x in weights):
            raise ValueError("密集奖励权重必须是非负有限数")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("密集奖励尺度必须是正有限数")


def relative_point_velocity(point: torch.Tensor, a_pos: torch.Tensor,
                            a_velocity: torch.Tensor, a_omega: torch.Tensor,
                            b_pos: torch.Tensor, b_velocity: torch.Tensor,
                            b_omega: torch.Tensor) -> torch.Tensor:
    """计算测量点相对接触对象的真实速度，全部输入[...,3]，单位米/秒或弧度/秒。

    输出：v_A(point)-v_B(point)。不能只减两个body原点速度，否则转动平台上会算错。
    """
    va = a_velocity + torch.cross(a_omega, point-a_pos, dim=-1)
    vb = b_velocity + torch.cross(b_omega, point-b_pos, dim=-1)
    return va-vb


class InteractionBackend(Protocol):
    """真实/诊断后端共同协议；backend_kind必须标明是不是机器人仿真。"""
    robot_id: str
    robot: Any
    num_envs: int
    device: torch.device
    env_ids: tuple[int, ...]
    step_dt: float
    pairs: tuple[ContactPair, ...]
    backend_kind: str

    def reset(self, env_ids=None): ...
    def read(self) -> PhysicsObservation: ...
    def relations(self) -> RelationState: ...
    def reward_state(self) -> PhysicalRewardState: ...
    def step(self, action: torch.Tensor) -> TaskTransition: ...
    def set_parameters(self, env_ids, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
    def apply_force(self, force_w: torch.Tensor): ...


@dataclass(frozen=True)
class PlanTarget:
    """外部给定的一项接触目标。

    pair：关系名；position：相对目标对象的局部坐标，米。
    velocity：相对目标对象的局部速度，米/秒。contact：1要接触/0要离开/-1不约束。
    此处是目标数据，不保证任意给定目标对机器人可达；需要任务设计和物理验收。
    """
    pair: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0., 0., 0.)
    contact: int = 1

    def __post_init__(self):
        if self.contact not in (-1, 0, 1) or isinstance(self.contact, bool):
            raise ValueError("contact必须为-1/0/1")
        for v in (self.position, self.velocity):
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError("目标位置/速度必须是3个有限数字")


@dataclass(frozen=True)
class PlanStage:
    """计划的一个阶段：在容差内连续保持hold_steps才进入下一阶段，避免擦边误判。"""
    name: str
    targets: tuple[PlanTarget, ...]
    hold_steps: int = 5
    max_steps: int = 200
    position_tolerance_m: float = .03
    velocity_tolerance_m_s: float = .10
    unload_reward_weight: float = 0.0
    unload_force_scale_n: float = 25.0

    def __post_init__(self):
        if not self.name or not self.targets:
            raise ValueError("阶段必须有名字和至少一个目标")
        if len({x.pair for x in self.targets}) != len(self.targets):
            raise ValueError("阶段目标不能重复指定同一关系")
        for x in (self.hold_steps, self.max_steps):
            if type(x) is not int or x < 1:
                raise ValueError("阶段帧数必须为正整数")
        if self.hold_steps > self.max_steps:
            raise ValueError("hold_steps不能超过max_steps")
        for x in (self.position_tolerance_m, self.velocity_tolerance_m_s):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("目标容差必须为正数")
        if (not math.isfinite(self.unload_reward_weight) or self.unload_reward_weight < 0
                or not math.isfinite(self.unload_force_scale_n) or self.unload_force_scale_n <= 0):
            raise ValueError("卸载奖励权重必须非负，力尺度必须为正数")


def unloading_score(normal_force: torch.Tensor, desired_contact: torch.Tensor,
                    mask: torch.Tensor, force_scale_n: float) -> torch.Tensor:
    """把摆动接触的法向载荷变成连续 [0,1] 卸载分数；无摆动目标时为0。"""
    if normal_force.shape != desired_contact.shape or mask.shape != desired_contact.shape:
        raise ValueError("卸载奖励张量形状不一致")
    swing = mask & (desired_contact >= 0) & (desired_contact < .5)
    count = swing.sum(-1)
    value = torch.exp(-normal_force.clamp_min(0)/force_scale_n)
    return torch.where(count > 0, (value*swing).sum(-1)/count.clamp_min(1), 0.)


class ContactPlanTask:
    """外部计划→实时目标→奖励→连续达标→换阶段，接入现有统一PPO。

    输入：后端和一组固定阶段；输出：PhysicsTask接口。
    只提供明确的计划执行，不生成未知器具的最佳接触顺序。
    位置跟踪使用body固定测量点；碰撞检测可以来自该body，二者不混为一个点。
    """
    auto_reset = False

    def __init__(self, backend: InteractionBackend, stages: Sequence[PlanStage],
                 reward: ContactRewardConfig | None = None):
        self.backend = backend
        self.stages = tuple(stages)
        self.reward_config = reward or ContactRewardConfig()
        if not self.stages:
            raise ValueError("计划不能为空")
        for k in ("robot_id", "robot", "num_envs", "device", "env_ids", "backend_kind"):
            setattr(self, k, getattr(backend, k))
        self.step_dt = backend.step_dt
        self.names = tuple(p.name for p in backend.pairs)
        if len(set(self.names)) != len(self.names):
            raise ValueError("接触关系名字必须唯一")
        self._registry = tuple(backend.pairs)
        for stage in self.stages:
            if set(t.pair for t in stage.targets) - set(self.names):
                raise ValueError("计划目标引用了不存在的接触关系")
        b = self.num_envs
        self.phase = torch.zeros(b, device=self.device, dtype=torch.long)
        self.age = self.phase.clone()
        self.hold = self.phase.clone()
        self._starts = torch.ones(b, device=self.device, dtype=torch.bool)
        self.pending = torch.ones_like(self._starts)
        self.last_success = torch.zeros_like(self._starts)
        self.last_metrics: dict[str, torch.Tensor] = {}
        self._previous_action = torch.zeros(b, self.robot.dof, device=self.device)
        self._has_previous_action = torch.zeros(b, dtype=torch.bool, device=self.device)

    @property
    def episode_start(self):
        """输出[B]新回合标志，读数据不会推进任务。"""
        return self._starts

    def reset(self, env_ids=None):
        """重置选中环境与阶段计数；None为全量，不能在首次只初始化部分环境。"""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if self.pending.all().item() and env_ids is not None and not hasattr(self, "_initialized"):
            raise ValueError("第一次必须全量reset")
        self.backend.reset(env_ids)
        self.phase[ids] = 0; self.age[ids] = 0; self.hold[ids] = 0
        self._starts[ids] = True; self.pending[ids] = False
        self.last_success[ids] = False
        self._previous_action[ids] = 0
        self._has_previous_action[ids] = False
        self._initialized = True

    def read(self):
        """输出真实后端观测；对象身份变化时拒绝继续比较旧动力学预测。"""
        if tuple(self.backend.pairs) != self._registry:
            raise ValueError("任务运行时修改了接触身份；必须重建任务与记忆")
        return self.backend.read()

    def _targets(self):
        """将当前阶段的对象局部目标转换成世界目标；输出位置、相对速度、接触要求、mask。"""
        r = self.backend.relations()
        b, c, _ = r.point_pos.shape
        local = torch.zeros_like(r.point_pos); velocity = torch.zeros_like(local)
        mask = torch.zeros(b, c, dtype=torch.bool, device=self.device)
        contact = local.new_full((b, c), -1.)
        for i, stage in enumerate(self.stages):
            rows = self.phase == i
            for target in stage.targets:
                j = self.names.index(target.pair)
                local[rows, j] = local.new_tensor(target.position)
                velocity[rows, j] = local.new_tensor(target.velocity)
                contact[rows, j] = target.contact
                mask[rows, j] = True
        world = r.object_pos + rotate(r.object_quat, local)
        rel_vel_w = rotate(r.object_quat, velocity)
        return r, world, rel_vel_w, contact, mask

    def goal(self):
        """输出ControlGoal；每个目标在当前机身坐标系下表达，保持与物理观测槽位对齐。"""
        r, world, speed, contact, mask = self._targets()
        position_b = rotate(r.root_quat, world-r.root_pos[:, None], inverse=True)
        speed_b = rotate(r.root_quat, speed, inverse=True)
        return ControlGoal(contact_features=torch.cat((position_b, speed_b, contact[..., None]), -1),
                           contact_mask=mask)

    def step(self, action):
        """执行动作并评价旧阶段，达标后才更新下次目标；结束行不自动reset。

        输出TaskTransition；last_metrics保留成功、误差及阶段进度供冻结评测使用。
        阶段超时当作任务失败终止；后端人工episode时间上限保持truncated语义。
        """
        if self.pending.any().item():
            raise RuntimeError("结束环境必须reset后才能继续step")
        result = self.backend.step(action)
        r, pos, vel, contact, mask = self._targets()
        pe = torch.linalg.vector_norm(r.point_pos-pos, dim=-1)
        ve = torch.linalg.vector_norm(r.relative_velocity-vel, dim=-1)
        correct = (contact < 0) | (r.found == (contact > .5))
        n = mask.sum(-1).clamp_min(1)
        reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        score = pe.new_zeros(self.num_envs)
        unload = pe.new_zeros(self.num_envs)
        for i, stage in enumerate(self.stages):
            row = self.phase == i
            good = ((pe <= stage.position_tolerance_m) &
                    (ve <= stage.velocity_tolerance_m_s) & correct) | ~mask
            reached[row] = good.all(-1)[row]
            term = torch.exp(-(pe/stage.position_tolerance_m).square()
                             -(ve/stage.velocity_tolerance_m_s).square())
            term = term * (.25 + .75*correct.to(term.dtype))
            score[row] = ((term*mask).sum(-1)/n)[row]
            if stage.unload_reward_weight:
                shaped=unloading_score(r.normal_force,contact,mask,stage.unload_force_scale_n)
                unload[row]=(stage.unload_reward_weight*shaped)[row]

        cfg = self.reward_config
        physical = self.backend.reward_state()
        dtype = score.dtype
        # 承重目标来自当前计划中 contact=1 的关系；摆动脚(contact=0)不计入支撑力。
        support_mask = mask & (contact > .5)
        support_force = (r.normal_force.clamp_min(0)*support_mask).sum(-1)
        total_force = (r.normal_force.clamp_min(0)*mask).sum(-1)
        support_fraction = support_force/(total_force+1e-6)
        support_score = support_fraction*(1-torch.exp(-support_force/cfg.support_force_scale_n))
        support_reward = cfg.support_weight*support_score

        root_up = rotate(r.root_quat, r.root_pos.new_tensor((0., 0., 1.)).expand_as(r.root_pos))
        tilt = torch.acos(root_up[:, 2].clamp(-1., 1.))
        upright_score = torch.exp(-(tilt/cfg.upright_sigma_rad).square())
        upright_reward = cfg.upright_weight*upright_score

        support_count = support_mask.sum(-1).clamp_min(1)
        support_target = (pos*support_mask[..., None]).sum(-2)/support_count[:, None]
        com_error = torch.linalg.vector_norm(physical.com_pos_w[:, :2]-support_target[:, :2], dim=-1)
        com_score = torch.exp(-(com_error/cfg.com_sigma_m).square())
        com_reward = cfg.com_transfer_weight*com_score

        delta = action-self._previous_action
        action_rate_penalty = delta.square().mean(-1)*self._has_previous_action.to(dtype)
        self._previous_action.copy_(action)
        self._has_previous_action.fill_(True)

        q = physical.joint_pos
        limits = physical.soft_joint_pos_limits
        if limits.shape != (*q.shape, 2):
            raise ValueError("soft_joint_pos_limits形状必须是[B,N,2]")
        span = (limits[..., 1]-limits[..., 0]).clamp_min(1e-6)
        violation = (torch.relu(limits[..., 0]-q)+torch.relu(q-limits[..., 1]))/span
        joint_limit_penalty = violation.square().mean(-1)
        root_ang_vel_penalty = physical.root_ang_vel_b.square().mean(-1)
        slide_mask = support_mask & r.found
        slide_count = slide_mask.sum(-1).clamp_min(1)
        foot_slide_penalty = (r.relative_velocity[..., :2].square().sum(-1)*slide_mask).sum(-1)/slide_count
        undesired_contact_penalty = physical.undesired_contact.to(dtype)
        self.age += 1
        self.hold = torch.where(reached, self.hold+1, 0)
        completed = torch.zeros_like(reached); failed = torch.zeros_like(reached)
        for i, stage in enumerate(self.stages):
            row = self.phase == i
            completed |= row & (self.hold >= stage.hold_steps)
            failed |= row & (self.age >= stage.max_steps) & ~completed
        completed &= ~(result.terminated | result.truncated)
        success = completed & (self.phase == len(self.stages)-1)
        advance = completed & ~success
        self.phase[advance] += 1; self.age[advance] = 0; self.hold[advance] = 0
        terminated = result.terminated | failed | success
        truncated = result.truncated & ~terminated
        self.pending = terminated | truncated
        self._starts.zero_(); self.last_success = success
        self.last_metrics = {"success": success, "within_tolerance": reached,
                             "position_error_m": (pe*mask).sum(-1)/n,
                             "relative_speed_error_m_s": (ve*mask).sum(-1)/n,
                             "stage": self.phase.clone(), "task_failure": failed | result.terminated,
                             "support": support_reward, "upright": upright_reward,
                             "com_transfer": com_reward, "action_rate": action_rate_penalty,
                             "joint_limit": joint_limit_penalty,
                             "undesired_contact": undesired_contact_penalty,
                             "root_ang_vel": root_ang_vel_penalty,
                             "foot_slide": foot_slide_penalty}
        penalties = (cfg.action_rate_weight*action_rate_penalty+
                     cfg.joint_limit_weight*joint_limit_penalty+
                     cfg.undesired_contact_weight*undesired_contact_penalty+
                     cfg.root_ang_vel_weight*root_ang_vel_penalty+
                     cfg.foot_slide_weight*foot_slide_penalty)
        reward = (score + unload + support_reward + upright_reward + com_reward - penalties +
                  completed.to(dtype) - (failed | result.terminated).to(dtype))
        return TaskTransition(reward, terminated, truncated,
                              {"tracking": score, "unloading": unload,
                               "support": support_reward, "upright": upright_reward,
                               "com_transfer": com_reward,
                               "action_rate_penalty": cfg.action_rate_weight*action_rate_penalty,
                               "joint_limit_penalty": cfg.joint_limit_weight*joint_limit_penalty,
                               "undesired_contact_penalty": cfg.undesired_contact_weight*undesired_contact_penalty,
                               "root_ang_vel_penalty": cfg.root_ang_vel_weight*root_ang_vel_penalty,
                               "foot_slide_penalty": cfg.foot_slide_weight*foot_slide_penalty,
                               "stage_completed": completed.to(dtype)})


@dataclass(frozen=True)
class Range:
    """一个物理参数的闭区间，来自YAML；固定参数写成[值,值]。"""
    low: float
    high: float

    def __post_init__(self):
        if not all(isinstance(x, (float, int)) and not isinstance(x, bool)
                   and math.isfinite(x) for x in (self.low, self.high)) or self.low > self.high:
            raise ValueError("物理区间必须是递增有限实数")


class DomainScheduler:
    """只从预先划分的域抽物理条件；测试时难度固定为1，不继续训练课程。

    domains={域名:{参数名:[low,high]}}，train/evaluation是不同域名列表。
    难度缩小随机区间的宽度但不改区间中心，因此不会自动替你选安全的中心值。
    update只用完成回合的成功率；不能用测试结果调整训练域或课程。
    """
    def __init__(self, domains, train, evaluation, *, seed=42, level=.2,
                 promote_threshold=.8, window=20):
        self.domains = {name: {k: Range(*v) for k, v in domain.items()}
                        for name, domain in domains.items()}
        self.train_names, self.eval_names = tuple(train), tuple(evaluation)
        if not self.train_names or not self.eval_names or set(train) & set(evaluation):
            raise ValueError("训练域/评测域必须非空且名字不能重叠")
        if (set(train)|set(evaluation)) - set(domains):
            raise ValueError("域名不存在")
        key_sets = [set(d) for d in self.domains.values()]
        if not key_sets or any(s != key_sets[0] for s in key_sets):
            raise ValueError("所有域必须显式设置同一组参数")
        if not 0 <= level <= 1 or not 0 <= promote_threshold <= 1 or type(window) is not int or window < 1:
            raise ValueError("课程参数非法")
        for d in self.domains.values():
            for k in ("friction", "mass_scale"):
                if k in d and d[k].low <= 0:
                    raise ValueError(f"{k}必须正数")
            if "delay_steps" in d and (d["delay_steps"].low < 0 or
               any(v != int(v) for v in (d["delay_steps"].low, d["delay_steps"].high))):
                raise ValueError("delay_steps必须为非负整数区间")
        self.level = float(level); self.threshold = promote_threshold; self.window = window
        self.history = []; self.generator = torch.Generator().manual_seed(seed)

    def sample(self, n: int, *, evaluation=False):
        """输入环境数，输出各参数的CPU张量[B]和所抽域名；后端负责写入并读回。"""
        names = self.eval_names if evaluation else self.train_names
        selected = torch.randint(len(names), (n,), generator=self.generator)
        values = {k: torch.empty(n) for k in next(iter(self.domains.values()))}
        level = 1. if evaluation else self.level
        for i, name in enumerate(names):
            mask = selected == i; count = int(mask.sum())
            for k, r in self.domains[name].items():
                mid = (r.high+r.low)/2; half = (r.high-r.low)*level/2
                if k == "delay_steps":
                    lo, hi = math.ceil(mid-half), math.floor(mid+half)
                    if lo > hi: lo = hi = round(mid)
                    draw = torch.randint(lo, hi+1, (count,), generator=self.generator).float()
                else:
                    draw = mid-half + torch.rand(count, generator=self.generator)*2*half
                values[k][mask] = draw
        return values, tuple(names[i] for i in selected.tolist())

    def record(self, successes: torch.Tensor):
        """输入已完成的训练回合结果[K]；攒够window个才升级/降级课程，不在评测调用。"""
        self.history.extend(bool(v) for v in successes.detach().cpu().flatten().tolist())
        while len(self.history) >= self.window:
            batch, self.history = self.history[:self.window], self.history[self.window:]
            rate = sum(batch)/len(batch)
            if rate >= self.threshold: self.level = min(1., self.level+.1)
            elif rate < .2: self.level = max(0., self.level-.1)

    def split_report(self):
        """明确报告域名分离和参数区间是否真分离；仅换名字不会被标成参数零样本。"""
        pairs = []
        for ev in self.eval_names:
            for tr in self.train_names:
                a, b = self.domains[ev], self.domains[tr]
                disjoint = [k for k in a if a[k].high < b[k].low or b[k].high < a[k].low]
                pairs.append({"train": tr, "evaluation": ev, "disjoint_parameters": disjoint})
        return {"domain_names_disjoint": True,
                "all_parameter_supports_disjoint": all(p["disjoint_parameters"] for p in pairs),
                "pairs": pairs}


class RandomizedTask:
    """给任意PhysicsTask加真实参数采样、动作延迟和定时外力，不修改PPO原始动作。

    task必须暴露backend.set_parameters/readback/apply_force。
    物理参数由后端明确支持；不支持就报错，不能仅在字典里记下数值假装生效。
    动作延迟由真实执行队列实现，body_obs中上一动作仍记录策略原始命令。
    """
    auto_reset = False

    def __init__(self, task, scheduler: DomainScheduler, *, evaluation=False,
                 push_interval_steps=0, privileged_learning=False,
                 actor_observation="full", actor_dynamics="online"):
        self.task, self.backend, self.scheduler = task, task.backend, scheduler
        for k in ("robot_id", "robot", "num_envs", "device", "env_ids", "backend_kind"):
            setattr(self, k, getattr(task, k))
        self.evaluation = evaluation
        if type(privileged_learning) is not bool:
            raise TypeError("privileged_learning必须是bool")
        if actor_observation not in ("full", "deployable"):
            raise ValueError("actor_observation必须是full或deployable")
        if actor_dynamics not in ("none", "online", "oracle"):
            raise ValueError("actor_dynamics必须是none、online或oracle")
        if actor_dynamics == "oracle" and not privileged_learning:
            raise ValueError("oracle Actor需要开启privileged_learning")
        self.privileged_learning = privileged_learning
        self.actor_observation = actor_observation
        self.actor_dynamics = actor_dynamics
        if type(push_interval_steps) is not int or push_interval_steps < 0:
            raise ValueError("push_interval_steps必须为非负整数")
        self.push_interval_steps = push_interval_steps
        self.capacity = int(max(d.get("delay_steps", Range(0,0)).high for d in scheduler.domains.values()))+1
        self.queue = torch.zeros(self.num_envs, self.capacity, self.robot.dof, device=self.device)
        self.last_raw = self.queue[:, 0].clone(); self.slot = 0
        self.last_executed_action = self.last_raw.clone()
        self.delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clock = self.delay.clone(); self.push = self.last_raw.new_zeros((self.num_envs, 3))
        self.applied_parameters = {}; self.domain_names = [None]*self.num_envs
        self.last_applied_force = self.queue.new_zeros((self.num_envs, 3))
        # push_force_n 是未来定时扰动的幅值，在实际施加前不可观测，不能把它当作
        # 在线辨识目标；Critic/学生只读取上一控制步真正施加的力向量。
        parameter_names = tuple(sorted(k for k in next(iter(scheduler.domains.values()))
                                       if k != "push_force_n"))
        self.privileged_names = parameter_names + (
            "last_force_x", "last_force_y", "last_force_z")
        self._privileged_bounds = {}
        for name in self.privileged_names[:-3]:
            ranges = [domain[name] for domain in scheduler.domains.values()]
            self._privileged_bounds[name] = (min(x.low for x in ranges),
                                              max(x.high for x in ranges))
        max_push = max((max(abs(x.low), abs(x.high))
                       for domain in scheduler.domains.values()
                       for name, x in domain.items() if name == "push_force_n"), default=1.0)
        self._force_scale = max(float(max_push), 1.0)

    @property
    def privileged_dim(self):
        """Critic/系统辨识目标宽度；未开启时为0。"""
        return len(self.privileged_names) if self.privileged_learning else 0

    def privileged(self):
        """返回归一化仿真真值；只允许进入Critic和监督损失，不能进入Actor编码器。"""
        if not self.privileged_learning:
            return None
        values = []
        for name in self.privileged_names[:-3]:
            value = self.applied_parameters[name]
            low, high = self._privileged_bounds[name]
            if high == low:
                values.append(torch.zeros_like(value))
            else:
                values.append(2 * (value-low)/(high-low) - 1)
        force = (self.last_applied_force/self._force_scale).unbind(-1)
        result = torch.stack((*values, *force), -1)
        if not torch.isfinite(result).all().item():
            raise ValueError("privileged物理真值包含非有限值")
        return result

    def policy_dynamics(self, encoding, privileged=None):
        """选择 Actor 实际读取的动力学表示。

        online 是可部署的历史编码；none 消融历史 latent；oracle 将仿真真值
        填入 latent，仅作为不可部署性能上限。context 始终来自当前可观测帧。
        """
        if self.actor_dynamics == "online":
            return encoding
        from ..models.physics import DynamicsEncoding
        latent = torch.zeros_like(encoding.latent)
        if self.actor_dynamics == "oracle":
            if privileged is None:
                raise ValueError("oracle Actor缺少privileged真值")
            if privileged.shape[:-1] != latent.shape[:-1]:
                raise ValueError("oracle privileged批次形状与latent不一致")
            if privileged.shape[-1] > latent.shape[-1]:
                raise ValueError("oracle privileged维度大于latent宽度")
            padding = latent[..., privileged.shape[-1]:]
            latent = torch.cat((privileged, padding), dim=-1)
        return DynamicsEncoding(latent, encoding.context, encoding.hidden,
                                torch.zeros_like(encoding.prediction_error))

    @property
    def episode_start(self): return self.task.episode_start

    @property
    def last_metrics(self): return self.task.last_metrics

    def reset(self, env_ids=None):
        """重置后应用参数，读回核对；同环境队列清零，其他环境参数保持不变。"""
        self.task.reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        values, names = self.scheduler.sample(len(ids), evaluation=self.evaluation)
        values = {k: v.to(self.device) for k, v in values.items()}
        physics = {k: v for k,v in values.items() if k not in ("delay_steps", "push_force_n")}
        readback = self.backend.set_parameters(ids, physics)
        if set(readback) != set(physics): raise ValueError("后端没有完整读回物理参数")
        for k,v in physics.items():
            torch.testing.assert_close(readback[k], v, rtol=1e-5, atol=1e-6,
                                       msg=f"{k}实际读回值与要求不一致")
        self.queue[ids] = 0; self.last_raw[ids] = 0;self.last_executed_action[ids]=0
        self.clock[ids] = 0
        self.last_applied_force[ids] = 0
        self.delay[ids] = values.get("delay_steps", torch.zeros(len(ids), device=self.device)).long()
        self.push[ids] = 0
        self.push[ids,0] = values.get("push_force_n", torch.zeros(len(ids), device=self.device))
        for k,v in values.items():
            if k not in self.applied_parameters:
                self.applied_parameters[k] = torch.zeros(self.num_envs, device=self.device)
            self.applied_parameters[k][ids] = v
        for i,n in zip(ids.cpu().tolist(), names): self.domain_names[i] = n

    def read(self):
        """读取执行后的身体状态，单独保留原始上一步动作，不把延迟动作冒充采样动作。"""
        from ..algorithms.sequence_training import clone_tree
        frame = clone_tree(self.task.read())
        frame.body_obs[..., -self.robot.dof:] = self.last_raw
        if self.actor_observation == "deployable":
            # K1策略不读取仿真器精确根线速度、接触法线或法向力。足端相对
            # 位置可由关节编码器做FK估计，接触开关可由实机接触估计器提供。
            # 相对地面速度仍含仿真器精确根平移速度，因此也必须屏蔽。
            frame.base_lin_vel_b.zero_()
            frame.core_valid[..., :3] = False
            frame.contact_features[..., 3:10] = 0
            frame.contact_features[..., 11] = 0
            # 当前后端的物体状态直接来自仿真真值；接入真实视觉/状态估计器前
            # 不允许通过DynamicsModel旁路进入Actor。
            frame.object_features.zero_()
            frame.object_mask.zero_()
        return frame

    def goal(self): return self.task.goal()

    def step(self, raw_action):
        """原始action进队列→延迟后执行→返回同一步真实奖励；PPO仍保存原始action。"""
        if raw_action.shape != self.last_raw.shape or not torch.isfinite(raw_action).all().item():
            raise ValueError("原始动作形状错误或非有限")
        self.queue[:,self.slot] = raw_action
        index = (self.slot-self.delay) % self.capacity
        executed = self.queue[torch.arange(self.num_envs,device=self.device),index].clone()
        self.slot = (self.slot+1) % self.capacity
        force = torch.zeros_like(self.push)
        if self.push_interval_steps:
            active = ((self.clock+1) % self.push_interval_steps) == 0
            force[active] = self.push[active]
        self.backend.apply_force(force)
        result = self.task.step(executed)
        self.last_executed_action.copy_(executed)
        self.last_applied_force.copy_(force)
        self.last_raw.copy_(raw_action); self.clock += 1
        done = result.terminated | result.truncated
        if not self.evaluation and done.any().item():
            success = self.last_metrics.get("success", torch.zeros_like(done))
            self.scheduler.record(success[done])
        return result
