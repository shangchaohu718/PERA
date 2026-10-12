"""【历史对照，不是新项目默认入口】
新物理训练使用 api.py / algorithms.sequence_training。
第 10 个文件：与现有多机器人 Actor/Critic 兼容的序列 PPO。

本文件完整实现：按机器人存储 rollout、时间限制修正的 GAE、PPO clipped loss、
Value loss、Entropy、可选 MoE 辅助项、按环境分组的序列 minibatch、梯度裁剪、
旧策略概率对齐 Gate，以及模型/优化器 checkpoint。

重要约定（不能省略）：
1. 每份 buffer 只装一种机器人，采样用 act_step()，记录**未裁剪的原始 action**。
2. 每个 transition 提供 next_value，且必须用真正的下一状态计算；如果 time-limit
   截断并自动 reset，next_value 必须来自 reset 前的 final observation，不能用 reset 后。
3. episode_start 指本次动作发生前的观测是否是新回合第一帧；与 done 不是同一时刻。
4. 不在时间轴切 PPO minibatch，而是按环境轴切，并保留完整 rollout 与可选前缀。
5. 如果 rollout 在回合中途开始，必须提供足够历史前缀。前缀只供时序上下文，
   不计算 PPO 损失；长度至少 n_layers*(context_len-1)，除非前缀含实际 reset。
6. 无前缀时，要求采样的第一个时间点确实为各环境的新回合第一帧。
7. 更新前重新计算采样时 action 的 log_prob，超过容差直接拒绝训练。
   这能及时发现采样/训练历史不一致、错用了裁剪后的 action 或策略已变化。
8. 每次更新使旧 KV cache 作废。历史前缀允许梯度经过，但不计入 PPO 奖励损失。
9. Critic 可使用特权状态，Actor 不读取；不同机器人共享策略主干，但动作头不同。
10. 这只是训练算法，不含 Isaac Lab/mjlab 环境驱动、接触奖励或数据文件读取。
    没有真实 rollout 验证前，不应宣称已经完成端到端机器人训练。

运行：python -m unified_motion.algorithms.ppo
依赖：Python >=3.10，PyTorch，以及前 09 个模块。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Sequence

import torch
from torch import nn

from .reference_actor_critic import ActionStep, MultiRobotActorCritic
from ..reference_observations import ReferenceObservation
from ..robot_contract import RobotContract
from ..algorithms.common import PPOConfig, compute_gae, _require_number






@dataclass
class HistoryPrefix:
    """采样批次之前、真实保存下来的历史；不能用随机张量冒充。

    batch 维在前：[B,P,*]。每一行与 PPO rollout 的同一个环境 ID 对应。
    若带 privileged，Critic 也会读取其历史位置，但这些位置不会计算 PPO loss。
    """

    observation: torch.Tensor
    reference: ReferenceObservation
    episode_start: torch.Tensor
    env_ids: tuple[int, ...]
    privileged: torch.Tensor | None = None


@dataclass
class PPOBatch:
    """训练张量统一为 [B,L,...]。L=历史前缀 P + rollout 的 T。

    loss_mask=True 的位置才计算 PPO；前缀全部 False，不充当训练样本。
    valid_mask 表示真实时间点；在这里前缀和 rollout 都应是真实数据。
    """

    robot_id: str
    observation: torch.Tensor
    reference: ReferenceObservation
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_start: torch.Tensor
    valid_mask: torch.Tensor
    loss_mask: torch.Tensor
    privileged: torch.Tensor | None
    prefix_len: int

    @property
    def num_envs(self) -> int:
        return self.observation.shape[0]

    def select_envs(self, index: torch.Tensor) -> PPOBatch:
        """只切环境维，时间轴绝对不能在这里截断。"""
        if not isinstance(index, torch.Tensor) or index.ndim != 1 or index.dtype != torch.long:
            raise TypeError("minibatch index 必须是一维 long Tensor")
        if index.device != self.observation.device:
            raise ValueError("minibatch index 与数据设备不同")

        def take(x: torch.Tensor) -> torch.Tensor:
            return x.index_select(0, index)

        return PPOBatch(
            robot_id=self.robot_id,
            observation=take(self.observation),
            reference=ReferenceObservation(take(self.reference.features), take(self.reference.valid_mask)),
            actions=take(self.actions), old_log_prob=take(self.old_log_prob),
            old_value=take(self.old_value), old_mean=take(self.old_mean),
            old_std=take(self.old_std), returns=take(self.returns),
            advantages=take(self.advantages), episode_start=take(self.episode_start),
            valid_mask=take(self.valid_mask), loss_mask=take(self.loss_mask),
            privileged=None if self.privileged is None else take(self.privileged),
            prefix_len=self.prefix_len,
        )




class RolloutBuffer:
    """为一个 robot_id 存储一次 on-policy rollout，时间轴先于环境轴。

    由外部环境循环调用 add()，不能把一个机器人组中的环境行悄悄交换顺序。
    每次 add 都必须传入真实 next_value：由 model.value(next_obs, next_ref) 获得。
    如果 auto-reset，truncated 的 next_obs 一定要取 reset 前 final_observation。
    """

    def __init__(
        self,
        robot_id: str,
        robot: RobotContract,
        *,
        horizon: int,
        num_envs: int,
        reference_steps: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        privileged_dim: int = 0,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(robot, RobotContract) or not isinstance(robot_id, str) or not robot_id.strip():
            raise TypeError("robot 与 robot_id 非法")
        for name, v in (("horizon", horizon), ("num_envs", num_envs),
                        ("reference_steps", reference_steps)):
            if type(v) is not int or v <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if type(privileged_dim) is not int or privileged_dim < 0:
            raise ValueError("privileged_dim 必须是非负整数")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("rollout dtype 必须是 float32 或 float64")
        self.robot_id, self.robot = robot_id, robot
        self.horizon, self.num_envs, self.reference_steps = horizon, num_envs, reference_steps
        self.device, self.dtype, self.privileged_dim = torch.device(device), dtype, privileged_dim
        self.env_ids = tuple(range(num_envs)) if env_ids is None else tuple(env_ids)
        if (len(self.env_ids) != num_envs or any(type(i) is not int for i in self.env_ids)
                or len(set(self.env_ids)) != num_envs):
            raise ValueError("env_ids 必须与批次一一对应且无重复")
        self._allocate()
        self.clear()

    def _allocate(self) -> None:
        t, b, n, h = self.horizon, self.num_envs, self.robot.dof, self.reference_steps
        dev, dt = self.device, self.dtype

        def tensor(*shape):
            return torch.empty(*shape, dtype=dt, device=dev)

        def bools(*shape):
            return torch.empty(*shape, dtype=torch.bool, device=dev)

        self.observations = tensor(t, b, 6+3*n)
        self.references = tensor(t, b, h, 16+2*n)
        self.reference_masks = bools(t, b, h)
        self.actions = tensor(t, b, n)
        self.old_log_probs = tensor(t, b)
        self.old_values = tensor(t, b)
        self.old_means = tensor(t, b, n)
        self.old_stds = tensor(t, b, n)
        self.rewards = tensor(t, b)
        self.next_values = tensor(t, b)
        self.terminated = bools(t, b)
        self.truncated = bools(t, b)
        self.episode_starts = bools(t, b)
        self.privileged = tensor(t, b, self.privileged_dim) if self.privileged_dim else None

    def clear(self) -> None:
        """准备新的 on-policy rollout；旧数据在此之后不可再用于更新。"""
        self._pos = 0
        self._consumed = False
        self._advantages = None
        self._returns = None
        self._return_config = None

    @property
    def full(self) -> bool:
        return self._pos == self.horizon

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def size(self) -> int:
        return self._pos

    def _copy(self, dst: torch.Tensor, src: torch.Tensor, label: str) -> None:
        if not isinstance(src, torch.Tensor) or src.shape != dst.shape:
            raise ValueError(f"{label} 应为 {tuple(dst.shape)}")
        if src.device != self.device or src.dtype != dst.dtype:
            raise ValueError(f"{label} dtype / device 不匹配")
        if src.is_floating_point() and not torch.isfinite(src).all().item():
            raise ValueError(f"{label} 含 NaN / Inf")
        dst.copy_(src.detach())  # rollout 必须与采样时的计算图分离。

    @torch.no_grad()
    def add(
        self,
        *,
        observation: torch.Tensor,
        reference: ReferenceObservation,
        step: ActionStep,
        reward: torch.Tensor,
        next_value: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        episode_start: torch.Tensor,
        env_ids: Sequence[int],
        privileged: torch.Tensor | None = None,
    ) -> None:
        """保存一次环境 step 的数据，原始 action 不可在这一步裁剪。"""
        if self._consumed:
            raise RuntimeError("rollout 已用于 PPO 更新；请先 clear() 收集新数据")
        if self.full:
            raise RuntimeError("rollout 已满")
        if tuple(env_ids) != self.env_ids:
            raise ValueError("env_ids 与采样时的环境行顺序不一致")
        if not isinstance(step, ActionStep) or step.robot_id != self.robot_id:
            raise ValueError("step 必须是本 robot_id 的 ActionStep")
        if step.joint_names != self.robot.joint_names:
            raise ValueError("采样动作的关节顺序与 RobotContract 不符")
        if not isinstance(reference, ReferenceObservation):
            raise TypeError("reference 必须是 ReferenceObservation")
        p = self._pos
        if self.privileged_dim:
            if privileged is None:
                raise ValueError("Critic 需要 privileged 数据")
            self._copy(self.privileged[p], privileged, "privileged")
        elif privileged is not None:
            raise ValueError("当前机器人不需要 privileged 数据")
        data = (
            (self.observations[p], observation, "observation"),
            (self.references[p], reference.features, "reference.features"),
            (self.reference_masks[p], reference.valid_mask, "reference.valid_mask"),
            (self.actions[p], step.action, "raw_action"),
            (self.old_log_probs[p], step.log_prob, "log_prob"),
            (self.old_values[p], step.value, "value"),
            (self.old_means[p], step.mean, "mean"),
            (self.old_stds[p], step.std, "std"),
            (self.rewards[p], reward, "reward"),
            (self.next_values[p], next_value, "next_value"),
            (self.terminated[p], terminated, "terminated"),
            (self.truncated[p], truncated, "truncated"),
            (self.episode_starts[p], episode_start, "episode_start"),
        )
        for dst, src, label in data:
            self._copy(dst, src, label)
        if not self.reference_masks[p].any(dim=-1).all().item():
            raise ValueError("每次采样必须有至少一个有效参考目标")
        if (self.reference_masks[p, :, 1:] & ~self.reference_masks[p, :, :-1]).any().item():
            raise ValueError("参考有效位必须在前，补位在后")
        if (self.terminated[p] & self.truncated[p]).any().item():
            raise ValueError("terminated/truncated 不能同时为 True")
        if (self.old_stds[p] <= 0).any().item():
            raise ValueError("std 必须大于0")
        if p > 0:
            expected = self.terminated[p-1] | self.truncated[p-1]
            if not torch.equal(self.episode_starts[p], expected):
                raise ValueError("回合开始标记与前一帧 done 不一致")
        self._pos += 1
        self._advantages = self._returns = None

    def compute_returns(self, config: PPOConfig) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.full or self._consumed:
            raise RuntimeError("GAE 只能对尚未消费的完整 rollout 计算")
        self._advantages, self._returns = compute_gae(
            self.rewards, self.old_values, self.next_values,
            self.terminated, self.truncated,
            gamma=config.gamma, gae_lambda=config.gae_lambda,
        )
        self._return_config = (config.gamma, config.gae_lambda)
        return self._advantages, self._returns

    def _validate_prefix(self, prefix: HistoryPrefix | None, *, min_history: int) -> int:
        if prefix is None:
            if not self.episode_starts[0].all().item():
                raise ValueError("rollout 在回合中途开始，必须提供真实历史前缀")
            return 0
        if not isinstance(prefix, HistoryPrefix):
            raise TypeError("prefix 必须是 HistoryPrefix")
        if type(min_history) is not int or min_history < 0:
            raise ValueError("min_history 必须是非负整数")
        if not isinstance(prefix.observation, torch.Tensor) or prefix.observation.ndim != 3:
            raise ValueError("prefix.observation 必须是 [B,P,O]")
        if prefix.env_ids != self.env_ids:
            raise ValueError("历史前缀的环境 ID/顺序与当前 rollout 不一致")
        b, p, o = prefix.observation.shape
        if b != self.num_envs or p <= 0 or o != self.observations.shape[-1]:
            raise ValueError("历史前缀的环境数/长度/观测宽度不匹配")
        if not isinstance(prefix.reference, ReferenceObservation):
            raise TypeError("prefix.reference 类型错误")
        for name, value, shape, dtype in (
            ("observation", prefix.observation, (b,p,o), self.dtype),
            ("reference.features", prefix.reference.features, (b,p,self.reference_steps,self.references.shape[-1]), self.dtype),
            ("reference.valid_mask", prefix.reference.valid_mask, (b,p,self.reference_steps), torch.bool),
            ("episode_start", prefix.episode_start, (b,p), torch.bool),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.device != self.device or value.dtype != dtype:
                raise ValueError(f"prefix 的 {name} 形状/类型/设备错误")
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise ValueError(f"prefix {name} 包含 NaN/Inf")
        if not prefix.reference.valid_mask.any(dim=-1).all().item():
            raise ValueError("历史前缀的每个真实时刻都必须有有效参考")
        if (prefix.reference.valid_mask[...,1:] & ~prefix.reference.valid_mask[...,:-1]).any().item():
            raise ValueError("历史前缀参考掩码必须连续")
        if self.privileged_dim:
            if (not isinstance(prefix.privileged, torch.Tensor)
                    or prefix.privileged.shape != (b,p,self.privileged_dim)
                    or prefix.privileged.device != self.device or prefix.privileged.dtype != self.dtype):
                raise ValueError("prefix 缺少正确的 privileged 数据")
            if not torch.isfinite(prefix.privileged).all().item():
                raise ValueError("prefix privileged 不是有限值")
        elif prefix.privileged is not None:
            raise ValueError("当前机器人不需要 prefix privileged")
        # 若第一个 rollout 时刻不是新回合、前缀内也没有对应 reset，就必须保留
        # 至少有效感受野长度的真实前文，才能等价重算多层注意力的旧 log_prob。
        if min_history > p:
            for i in range(b):
                if not self.episode_starts[0, i].item() and not prefix.episode_start[i].any().item():
                    raise ValueError(f"环境 {self.env_ids[i]} 历史不足：至少 {min_history} 帧")
        return p

    def make_batch(
        self, config: PPOConfig, *, prefix: HistoryPrefix | None = None,
        min_history: int = 0,
    ) -> PPOBatch:
        """生成 [B,P+T,...]，P 只作为上下文，绝不算入 PPO 损失。"""
        if not self.full or self._consumed:
            raise RuntimeError("只能从未消费的完整 rollout 构造 PPOBatch")
        if (self._advantages is None or self._returns is None
                or self._return_config != (config.gamma, config.gae_lambda)):
            self.compute_returns(config)
        p = self._validate_prefix(prefix, min_history=min_history)
        t, b = self.horizon, self.num_envs

        def bt(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(0, 1).contiguous()

        obs = bt(self.observations)
        ref = bt(self.references)
        ref_mask = bt(self.reference_masks)
        starts = bt(self.episode_starts)
        privileged = None if self.privileged is None else bt(self.privileged)
        actions, lp = bt(self.actions), bt(self.old_log_probs)
        values, means, stds = bt(self.old_values), bt(self.old_means), bt(self.old_stds)
        ret, adv = bt(self._returns), bt(self._advantages)
        if config.normalize_advantage:
            # 全体真实训练样本统一归一化，不根据每个 minibatch 分别算方差。
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        loss_mask = torch.ones(b, t, dtype=torch.bool, device=self.device)
        valid_mask = torch.ones_like(loss_mask)
        if p:
            obs = torch.cat((prefix.observation.detach(), obs), dim=1)
            ref = torch.cat((prefix.reference.features.detach(), ref), dim=1)
            ref_mask = torch.cat((prefix.reference.valid_mask.detach(), ref_mask), dim=1)
            starts = torch.cat((prefix.episode_start.detach(), starts), dim=1)
            if privileged is not None:
                privileged = torch.cat((prefix.privileged.detach(), privileged), dim=1)
            def prepend_zero(x: torch.Tensor, *, fill: float = 0) -> torch.Tensor:
                return torch.cat((x.new_full((b,p,*x.shape[2:]), fill), x), dim=1)
            actions = prepend_zero(actions)
            lp = prepend_zero(lp)
            values = prepend_zero(values)
            means = prepend_zero(means)
            stds = prepend_zero(stds, fill=1)
            ret = prepend_zero(ret)
            adv = prepend_zero(adv)
            loss_mask = torch.cat((torch.zeros(b,p,dtype=torch.bool,device=self.device),loss_mask),dim=1)
            valid_mask = torch.ones_like(loss_mask)
        return PPOBatch(
            self.robot_id, obs, ReferenceObservation(ref, ref_mask), actions, lp,
            values, means, stds, ret, adv, starts, valid_mask, loss_mask,
            privileged, p,
        )


@dataclass
class PPOStats:
    """本次 PPO 更新统计，供上层日志、早停 Gate、跨实验比较使用。"""

    updates: int
    early_stopped: bool
    policy_loss: float
    value_loss: float
    entropy: float
    balance_loss: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    max_preupdate_logprob_error: float
    max_preupdate_value_error: float
    max_preupdate_mean_error: float
    max_preupdate_std_error: float


class PPOTrainer:
    """训练器与物理仿真器解耦，模型权重和优化器都归这个对象管理。"""

    def __init__(self, model: MultiRobotActorCritic, config: PPOConfig | None = None) -> None:
        if not isinstance(model, MultiRobotActorCritic):
            raise TypeError("model 必须是 MultiRobotActorCritic")
        self.model = model
        self.config = PPOConfig() if config is None else config
        if not isinstance(self.config, PPOConfig):
            raise TypeError("config 必须是 PPOConfig")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                           eps=self.config.adam_eps)
        self.total_updates = 0

    def _verify_policy(self, batch: PPOBatch) -> tuple[float, float, float, float]:
        """在进行任何优化器更新前核查采样与重算的概率/价值是否对应。"""
        if not self.config.verify_old_policy:
            return 0.0, 0.0, 0.0, 0.0
        self.model.eval()
        max_lp, max_v, max_mean, max_std = 0.0, 0.0, 0.0, 0.0
        for ids in torch.arange(batch.num_envs, device=batch.observation.device).split(self.config.minibatch_envs):
            sub = batch.select_envs(ids)
            with torch.no_grad():
                ev = self.model.evaluate_actions(
                    sub.robot_id, sub.observation, sub.reference, sub.actions,
                    privileged=sub.privileged, valid_mask=sub.valid_mask,
                    episode_start=sub.episode_start,
                )
                lp_err = (ev.log_prob[sub.loss_mask] - sub.old_log_prob[sub.loss_mask]).abs().max().item()
                v_err = (ev.policy.values[sub.loss_mask] - sub.old_value[sub.loss_mask]).abs().max().item()
                mean_err = (ev.policy.mean[sub.loss_mask] - sub.old_mean[sub.loss_mask]).abs().max().item()
                std_err = (ev.policy.std[sub.loss_mask] - sub.old_std[sub.loss_mask]).abs().max().item()
            max_lp = max(max_lp, lp_err)
            max_v = max(max_v, v_err)
            max_mean = max(max_mean, mean_err)
            max_std = max(max_std, std_err)
        if (max_lp > self.config.logprob_atol or max_v > self.config.value_atol
                or max_mean > self.config.distribution_atol or max_std > self.config.distribution_atol):
            raise RuntimeError(
                "PPO 旧策略复算 Gate 不通过：logprob 最大误差 "
                f"{max_lp:.6g}，value {max_v:.6g}，mean {max_mean:.6g}，std {max_std:.6g}。"
                "检查历史前缀、动作是否被裁剪、旧策略是否改变、回合/reset 标记与环境顺序。"
            )
        return max_lp, max_v, max_mean, max_std

    def update(self, buffer: RolloutBuffer, *, prefix: HistoryPrefix | None = None) -> PPOStats:
        """一次 PPO 更新：同一份 rollout 多轮迭代，时间序列不被切断。"""
        if not isinstance(buffer, RolloutBuffer):
            raise TypeError("buffer 必须是 RolloutBuffer")
        if buffer.robot_id not in self.model.robot_ids:
            raise ValueError("Rollout 的机器人没有注册到当前模型")
        if buffer.robot.joint_names != self.model._robots[buffer.robot_id].joint_names:
            raise ValueError("Rollout 与模型的关节名单不一致")
        p = next(self.model.parameters())
        if buffer.device != p.device or buffer.dtype != p.dtype:
            raise ValueError("Rollout 的设备/精度必须与 Actor 相同")
        expected_pdim = self.model.critic.privileged_dims[buffer.robot_id]
        if buffer.privileged_dim != expected_pdim:
            raise ValueError("Rollout 的 privileged_dim 与 Critic 不一致")
        if buffer.reference_steps < 1 or buffer.horizon < 1:
            raise ValueError("rollout 长度必须是正整数")
        if buffer.consumed:
            raise RuntimeError("这份 rollout 已消费，不能再次用于 PPO")
        min_history = self.model.config.n_layers * (self.model.config.context_len - 1)
        batch = buffer.make_batch(self.config, prefix=prefix, min_history=min_history)
        if batch.prefix_len and self.config.balance_coef:
            # 旧 MoE 模块返回的是整段平均 balance，不带逐 token 权重；
            # 使用前缀时不能准确地只对训练位置计算平衡项，因此禁止静默偏置。
            raise ValueError("有 burn-in 前缀时，必须设置 balance_coef=0")
        max_lp, max_v, max_mean, max_std = self._verify_policy(batch)
        self.model.train()
        buffer._consumed = True  # 从现在开始不能用同一旧策略数据更新第二次。
        totals = [0.0] * 7
        num_updates, early = 0, False
        for epoch in range(self.config.epochs):
            permutation = torch.randperm(batch.num_envs, device=batch.observation.device)
            for ids in permutation.split(self.config.minibatch_envs):
                sub = batch.select_envs(ids)
                ev = self.model.evaluate_actions(
                    sub.robot_id, sub.observation, sub.reference, sub.actions,
                    privileged=sub.privileged, valid_mask=sub.valid_mask,
                    episode_start=sub.episode_start,
                )
                valid = sub.loss_mask
                # 原始 action 的概率比；环境对 q_target 做过限位也不影响这个定义。
                log_ratio = ev.log_prob[valid] - sub.old_log_prob[valid]
                ratio = log_ratio.exp()
                if not torch.isfinite(ratio).all().item():
                    raise FloatingPointError("PPO 概率比出现 NaN/Inf，拒绝继续更新")
                # 正规 PPO clipped surrogate：最大化取 min，因此损失取负号。
                advantage = sub.advantages[valid]
                unclipped = ratio * advantage
                clipped = ratio.clamp(1-self.config.clip_ratio, 1+self.config.clip_ratio) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                values = ev.policy.values[valid]
                returns = sub.returns[valid]
                if self.config.value_clip is None:
                    value_loss = 0.5*(values-returns).square().mean()
                else:
                    old_values = sub.old_value[valid]
                    v_clipped = old_values + (values-old_values).clamp(
                        -self.config.value_clip, self.config.value_clip,
                    )
                    value_loss = 0.5*torch.maximum(
                        (values-returns).square(), (v_clipped-returns).square()
                    ).mean()
                entropy = ev.entropy[valid].mean()
                balance = ev.policy.balance_loss if self.config.balance_coef else policy_loss.new_zeros(())
                # Analytic KL(old || new)，按动作维度汇总。比抽样 log_ratio 更稳定。
                old_std = sub.old_std[valid]
                old_mean = sub.old_mean[valid]
                new_std = ev.policy.std[valid]
                new_mean = ev.policy.mean[valid]
                kl = (
                    (new_std.log()-old_std.log())
                    + (old_std.square()+(old_mean-new_mean).square())/(2*new_std.square()) - 0.5
                ).sum(dim=-1).mean().clamp_min(0)
                clip_fraction = ((ratio-1).abs() > self.config.clip_ratio).float().mean()
                # KL 早停在反向前检查；不在此自动更改 LR，便于做单变量实验。
                if self.config.target_kl is not None and kl.item() > self.config.target_kl:
                    early = True
                    break
                loss = (policy_loss + self.config.value_coef*value_loss
                        - self.config.entropy_coef*entropy
                        + self.config.balance_coef*balance)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("PPO 总损失非有限，已拒绝梯度更新")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True,
                )
                if not torch.isfinite(grad_norm).item():
                    raise FloatingPointError("PPO 发现非有限梯度")
                self.optimizer.step()
                # 注意：模型权重改变后，之前采样的缓存立即失效。
                self.model.invalidate_caches()
                self.total_updates += 1
                num_updates += 1
                numbers = (policy_loss, value_loss, entropy, balance, kl, clip_fraction, grad_norm)
                totals = [old + float(x.detach().item()) for old, x in zip(totals, numbers)]
            if early:
                break
        avg = [x / max(num_updates, 1) for x in totals]
        return PPOStats(num_updates, early, *avg, max_lp, max_v, max_mean, max_std)

    def state_dict(self) -> dict:
        """保存模型、优化器和训练计数；下次恢复仍需自行重建 rollout 环境。"""
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "robots": self.model.robot_ids,
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, checkpoint: dict) -> None:
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
            raise ValueError("PPO checkpoint 版本不匹配")
        if checkpoint.get("config") != asdict(self.config) or checkpoint.get("robots") != self.model.robot_ids:
            raise ValueError("PPO checkpoint 配置/机器人注册信息不匹配")
        if type(checkpoint.get("total_updates")) is not int or checkpoint["total_updates"] < 0:
            raise ValueError("PPO checkpoint 更新计数错误")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_updates = int(checkpoint["total_updates"])
        self.model.invalidate_caches()
        self.optimizer.zero_grad(set_to_none=True)




