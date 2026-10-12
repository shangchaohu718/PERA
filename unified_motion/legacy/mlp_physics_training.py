"""【历史对照，不是新项目默认入口】
新物理训练使用 api.py / algorithms.sequence_training。
第15B个文件：物理反馈版 mjlab 采样 + MLP PPO + 独立动力学监督学习。

本文件是第14模块的真正训练接线，不是旧 Transformer PPO 的包装假象。

整个采样顺序：
    mjlab 实际状态 o_t → DynamicsModel.step() 得到 z_t
    → PhysicsMLPActorCritic 得到原始 a_t / log_prob / value
    → DynamicsModel.predict_next() 预测 o_(t+1)
    → mjlab.step(a_t) 真正执行
    → 读取 reset 之前的真实 o_(t+1)
    → MotionTrackingReward 计算本步奖励
    → 保存预测误差、下一状态、奖励、动作和旧策略分布
    → 结束的环境单独 reset，并推进 MotionSampler。

训练顺序：
    1. 使用已经采集的原始 action/log_prob 做 PPO 更新，只改 MLP Actor/Critic。
    2. 使用采样得到的真实状态转移监督 DynamicsModel，独立优化它。
    3. 动力学权重变化后清空旧运行时记忆，重新开始新的物理 rollout。

本版明确边界：
    - 先训练“有参考轨迹”的 K1。独立接触/物体 Goal、G1、MoE 全部留给下一版。
    - 只能在第11模块的 auto_reset=False 环境中运行；绝不使用 reset 后状态当 final state。
    - 这个文件不会伪造地面接触、示范数据、物体状态或奖励。
    - 不允许把执行器裁剪后的关节角当作 PPO 原始 action。
    - 不含自测函数和测试脚本，功能验收留给用户统一进行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ..algorithms.common import PPOConfig, compute_gae
from ..envs.mjlab_physics import K1PhysicsBridge
from ..data.motion_sampler import MotionSampler
from ..models.physics import (
    PhysicsObservation, ControlGoal, DynamicsEncoding, DynamicsMemory,
    DynamicsModel, PhysicsMLPActorCritic,
)
from ..reference_observations import ReferenceObservation
from ..rewards.motion_tracking import MotionTrackingReward


@dataclass
class PhysicsRollout:
    """一次采样保存的所有真实经验；时间维均放在 B 后面便于回放。

    输入来源：只允许由 PhysicsRolloutCollector.collect() 根据真实 step 生成。
    字段：
        observation / next_observation：[B,T,...]，实际状态和执行动作后的状态。
        reference：ReferenceObservation，[B,T,H,R]，每一帧自己的动作目标。
        dynamics_context / dynamics_latent：[B,T,P/Z]，采样时冻结的物理特征。
        action、old_mean、old_std：[B,T,N]，未经执行器裁剪的动作信息。
        old_log_prob / old_value：[B,T]，旧策略计算的概率/价值。
        reward / next_value / terminated / truncated：[T,B]，用于 GAE。
        episode_start：[B,T]，当前帧是否属于新回合第一帧。
        initial_hidden：[B,R]，首帧之前的动力学记忆；不使用未来信息。
        previous_predicted_core：[B,T,9+2N]，t-1 的预测，用于重算反馈误差。
        previous_prediction_valid：[B,T]，对应位置能否使用 t-1 的预测。
    输出：由 PPOUpdater 更新 Actor；由 train_dynamics() 训练下一帧预测。
    """

    observation: PhysicsObservation
    next_observation: PhysicsObservation
    reference: ReferenceObservation
    dynamics_context: torch.Tensor
    dynamics_latent: torch.Tensor
    action: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    next_value: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_start: torch.Tensor
    initial_hidden: torch.Tensor
    previous_predicted_core: torch.Tensor
    previous_prediction_valid: torch.Tensor
    valid_transition: torch.Tensor
    reward_terms: dict[str, torch.Tensor]

    @property
    def horizon(self) -> int:
        """功能：返回此次采样包含多少个真实控制步。输入：无。输出：整数 T。"""
        return self.action.shape[1]

    @property
    def num_envs(self) -> int:
        """功能：返回并行环境数。输入：无。输出：整数 B。"""
        return self.action.shape[0]


def _stack_frames(frames: list[PhysicsObservation]) -> PhysicsObservation:
    """功能：把连续采集的 T 帧拼成能进行训练的 [B,T,...]。

    输入：frames 是非空列表，每项 PhysicsObservation 都含 B 个环境的一帧数据。
    输出：PhysicsObservation，所有张量都多一个 T 时间维，顺序不变。
    原因：动力学模型的 GRU 必须按正确顺序学习，而不能将时间随机打乱。
    """
    if not frames:
        raise ValueError("没有采集任何物理观测")
    return PhysicsObservation(*(
        torch.stack([getattr(x, name) for x in frames], dim=1)
        for name in (
            "body_obs", "base_lin_vel_b", "contact_features", "contact_mask",
            "object_features", "object_mask", "core_valid",
        )
    ))


def _choose_envs(frame: PhysicsObservation, ids: torch.Tensor) -> PhysicsObservation:
    """功能：只取某些并行机器人环境，不改变这些环境内部的时间顺序。

    输入：frame 各字段为 [B,...]；ids 为待选择环境的一维整数编号 [K]。
    输出：新的 PhysicsObservation，字段变成 [K,...]。
    用途：PPO 按环境切 minibatch，避免切断动力学历史。
    """
    return PhysicsObservation(*(
        getattr(frame, name).index_select(0, ids)
        for name in (
            "body_obs", "base_lin_vel_b", "contact_features", "contact_mask",
            "object_features", "object_mask", "core_valid",
        )
    ))


class PhysicsRolloutCollector:
    """在真实 mjlab 物理环境里逐步采样，给 PPO 和动力学训练准备数据。

    输入：
        physics_bridge：第15A模块的 K1PhysicsBridge，环境需要 auto_reset=False。
        sampler：第12模块 MotionSampler，已装载 K1 重定向动作。
        dynamics：第14模块 DynamicsModel，输出物理估计和下一状态预测。
        policy：第14模块 PhysicsMLPActorCritic，输出 22 个原始动作。
        reward_fn：第13模块 MotionTrackingReward，真实对照当前参考动作打分。
    输出：collect(T) → PhysicsRollout；不会更新网络权重。

    当前只运行“动作参考追踪”任务，没有接触阶段标签的专用滑步奖励不能开启。
    """

    def __init__(
        self, physics_bridge: K1PhysicsBridge, sampler: MotionSampler,
        dynamics: DynamicsModel, policy: PhysicsMLPActorCritic,
        reward_fn: MotionTrackingReward,
    ) -> None:
        """功能：检查五个模块用的是同一款 K1、同一批环境和同样的采样频率。

        输入：参见类说明。
        输出：无；建立采样器运行时状态，但不自动启动 mjlab。
        """
        if not isinstance(physics_bridge, K1PhysicsBridge):
            raise TypeError("physics_bridge 必须为 K1PhysicsBridge")
        if not isinstance(sampler, MotionSampler):
            raise TypeError("sampler 必须为 MotionSampler")
        if not isinstance(dynamics, DynamicsModel) or not isinstance(policy, PhysicsMLPActorCritic):
            raise TypeError("必须使用第14模块的 DynamicsModel 和 PhysicsMLPActorCritic")
        if not isinstance(reward_fn, MotionTrackingReward):
            raise TypeError("reward_fn 必须是 MotionTrackingReward")
        self.physics_bridge = physics_bridge
        self.bridge = physics_bridge.bridge
        self.sampler = sampler
        self.dynamics = dynamics
        self.policy = policy
        self.reward_fn = reward_fn
        self.robot = self.bridge.robot
        self.num_envs = self.bridge.num_envs
        self.device = self.bridge.device
        if (dynamics.robot != self.robot or policy.robot != self.robot
                or reward_fn.robot != self.robot or sampler.library.robot != self.robot):
            raise ValueError("动作库、物理模型、Actor 和奖励函数必须属于同一个机器人")
        if sampler.num_envs != self.num_envs or abs(sampler.step_dt - float(self.bridge.env.step_dt)) > 1e-8:
            raise ValueError("动作采样器的并行环境数或控制周期与 mjlab 不一致")
        if sampler.library.device != self.device:
            raise ValueError("动作数据和 mjlab 必须在同一张设备上")
        if dynamics.physics_width != policy.physics_width or dynamics.latent_width != policy.latent_width:
            raise ValueError("动力学物理特征宽度和 MLP Actor 的输入宽度不一致")
        if policy.privileged_dim:
            raise ValueError("当前 Collector 未配置 Critic 特权状态，请使用 privileged_dim=0")
        if not sampler.offsets_s[0].abs().item() < 1e-7:
            raise ValueError("参考窗口第一个时间偏移必须是0，奖励才与当前状态对齐")
        if any(getattr(reward_fn.cfg, name) > 0 for name in (
            "foot_xy_weight", "foot_height_weight", "foot_rot_weight",
            "foot_slide_vel_weight", "foot_anchor_still_weight", "foot_contact_weight",
        )):
            raise ValueError("尚无匹配的鞋底 site 目标/相位标注，不能打开脚部专项奖励")
        if any(getattr(reward_fn.cfg, name) > 0 for name in (
            "body_pos_weight", "body_rot_weight", "body_lin_vel_weight",
        )) and not sampler.library.body_names:
            raise ValueError("使用身体部件奖励之前，需要动作库包含 body_reference 数据")
        self._ready = False
        self._current: PhysicsObservation | None = None
        self._dynamics_now: DynamicsEncoding | None = None
        self._memory: DynamicsMemory | None = None
        self._pre_hidden: torch.Tensor | None = None
        self._pre_prediction: torch.Tensor | None = None
        self._pre_prediction_valid: torch.Tensor | None = None

    @torch.no_grad()
    def reset_all(self) -> None:
        """功能：真正重置全部 mjlab 环境、参考动作和动力学记忆。

        输入：无。
        输出：无；保存刚重置后第0帧的物理观测和动力学隐藏状态。
        调用时机：第一次采样前；或 DynamicsModel 更新权重、缓存失效后。
        """
        self.policy.eval()
        self.dynamics.eval()
        self.bridge.reset()
        self.sampler.reset(
            current_root_pos_w=self.bridge.entity.data.root_link_pos_w,
            current_root_quat_wxyz=self.bridge.entity.data.root_link_quat_w,
        )
        frame = self.physics_bridge.read()
        begins = self.bridge.episode_start
        self._pre_hidden = torch.zeros(
            self.num_envs, self.dynamics.memory_width,
            device=self.device, dtype=frame.body_obs.dtype,
        )
        self._pre_prediction = torch.zeros(
            self.num_envs, 9 + 2 * self.robot.dof,
            device=self.device, dtype=frame.body_obs.dtype,
        )
        self._pre_prediction_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dyn, memory = self.dynamics.step(
            frame, episode_start=begins,
            env_ids=self.bridge.env_ids, memory=None,
        )
        self._current = frame
        self._dynamics_now = dyn
        self._memory = memory
        self._ready = True

    def _reference_goal(self) -> ReferenceObservation:
        """功能：把当前动作参考整理成 MLP Actor 的 Goal 输入。

        输入：无；读取 sampler 当前时刻和 mjlab 当前真实根部件位姿。
        输出：ReferenceObservation，形状 [B,H,16+2N]，每个环境各自一条参考。
        注意：这里只使用示范轨迹，未把滑板等新任务的接触规划冒充已有数据。
        """
        ref = self.sampler.reference_window()
        data = self.bridge.entity.data
        return self.bridge.reference_builder(
            ref,
            current_root_pos_w=data.root_link_pos_w,
            current_root_quat_wxyz=data.root_link_quat_w,
        )

    @torch.no_grad()
    def collect(self, horizon: int) -> PhysicsRollout:
        """功能：采集 T 个真实物理步；不会训练网络，也不会添加假奖励。

        输入：horizon 是需要采样多少个控制步，例如 32 或 128。
        输出：PhysicsRollout，B 个并行环境的 T 步物理反馈、动作与奖励。

        时序要求：
          每一步只在真实 o_(t+1) 出现后计算预测误差。
          done 时先读 final observation、奖励和 bootstrap，再 reset。
          动作参考从 t 推到 t+1；片段结束是任务终止，不是继续下一片段。
        """
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon 必须是正整数")
        if not self._ready:
            self.reset_all()
        if self.dynamics.training or self.policy.training:
            raise RuntimeError("采样时模型必须在 eval 模式；训练权重更新后先退出采样")
        recorded: dict[str, list] = {key: [] for key in (
            "frames", "next_frames", "references", "ref_masks", "context", "latent",
            "actions", "means", "stds", "logprobs", "values", "rewards", "next_values",
            "terminated", "truncated", "starts", "preds", "pred_valid",
        )}
        reward_terms: dict[str, list[torch.Tensor]] = {}
        initial_hidden = self._pre_hidden.detach().clone()
        for _ in range(horizon):
            frame = self._current
            dyn = self._dynamics_now
            memory = self._memory
            begins = self.bridge.episode_start
            if frame is None or dyn is None or memory is None:
                raise RuntimeError("尚未初始化物理观测和动力学记忆")
            if not torch.equal(begins, begins.to(torch.bool)):
                raise RuntimeError("episode_start 应为 bool")
            reference = self._reference_goal()
            policy_result = self.policy(frame, dyn, ControlGoal(motion=reference))
            action, log_prob = policy_result.sample()
            prediction = self.dynamics.predict_next(frame, action, dyn.latent)
            after_action = self.dynamics.remember_prediction(memory, prediction)

            # 物理动作真正执行；第11模块会把策略关节顺序转换为 mjlab 执行顺序。
            transition = self.bridge.step(action)
            self.sampler.advance()
            # 此刻有可能已终止：必须在任何 reset 之前读取最后真实物理状态。
            next_frame = self.physics_bridge.read()
            raw_reference = self.sampler.reference_window()
            body_reference = (self.sampler.body_reference_window()
                              if self.sampler.library.body_names else None)
            reward_output = self.reward_fn(
                self.physics_bridge.tracking_state(), raw_reference,
                bodies=self.physics_bridge.body_tracking_state() if body_reference is not None else None,
                body_reference=body_reference,
            )
            terminated = transition.terminated | self.sampler.finished()
            truncated = transition.truncated & ~terminated
            ended = terminated | truncated

            # 这次用于计算 V(o_(t+1))：真实 final state，不是 reset 后的初始状态。
            next_dyn, _ = self.dynamics.step(
                next_frame, episode_start=torch.zeros_like(ended),
                env_ids=self.bridge.env_ids, memory=after_action,
            )
            # sampler 在片段终点会把当前帧钳到末尾；这里仍然是有效参考。
            next_reference = self._reference_goal()
            next_value = self.policy(next_frame, next_dyn,
                                     ControlGoal(motion=next_reference)).value
            # terminated 后没有未来价值；时间截断才允许通过 final state bootstrap。
            next_value = torch.where(terminated, 0.0, next_value)

            recorded["frames"].append(frame)
            recorded["next_frames"].append(next_frame)
            recorded["references"].append(reference.features.detach().clone())
            recorded["ref_masks"].append(reference.valid_mask.detach().clone())
            recorded["context"].append(dyn.context.detach().clone())
            recorded["latent"].append(dyn.latent.detach().clone())
            recorded["actions"].append(action.detach().clone())
            recorded["means"].append(policy_result.mean.detach().clone())
            recorded["stds"].append(policy_result.std.detach().clone())
            recorded["logprobs"].append(log_prob.detach().clone())
            recorded["values"].append(policy_result.value.detach().clone())
            recorded["rewards"].append(reward_output.total.detach().clone())
            recorded["next_values"].append(next_value.detach().clone())
            recorded["terminated"].append(terminated.detach().clone())
            recorded["truncated"].append(truncated.detach().clone())
            recorded["starts"].append(begins.detach().clone())
            recorded["preds"].append(self._pre_prediction.detach().clone())
            recorded["pred_valid"].append(self._pre_prediction_valid.detach().clone())
            for name, term in reward_output.terms.items():
                reward_terms.setdefault(name, []).append(term.detach().clone())

            if ended.any().item():
                ids = ended.nonzero(as_tuple=False).flatten()
                self.bridge.reset(ids)
                data = self.bridge.entity.data
                self.sampler.reset(
                    env_ids=ids,
                    current_root_pos_w=data.root_link_pos_w[ids],
                    current_root_quat_wxyz=data.root_link_quat_w[ids],
                )
            # 接下来重新读取：结束的行是 reset 后状态，其他行仍是正常连续状态。
            next_current = self.physics_bridge.read()
            self._pre_hidden = after_action.hidden.detach().clone()
            self._pre_prediction = prediction.next_core.detach().clone()
            self._pre_prediction_valid = torch.ones_like(ended)
            dyn_now, new_memory = self.dynamics.step(
                next_current, episode_start=self.bridge.episode_start,
                env_ids=self.bridge.env_ids, memory=after_action,
            )
            self._current, self._dynamics_now, self._memory = next_current, dyn_now, new_memory

        def stack(key: str) -> torch.Tensor:
            """功能：把 T 次记录合并为 [B,T,...]；输入为字段名；输出为张量。"""
            return torch.stack(recorded[key], dim=1)

        return PhysicsRollout(
            observation=_stack_frames(recorded["frames"]),
            next_observation=_stack_frames(recorded["next_frames"]),
            reference=ReferenceObservation(stack("references"), stack("ref_masks")),
            dynamics_context=stack("context"),
            dynamics_latent=stack("latent"),
            action=stack("actions"), old_mean=stack("means"), old_std=stack("stds"),
            old_log_prob=stack("logprobs"), old_value=stack("values"),
            reward=torch.stack(recorded["rewards"], dim=0),
            next_value=torch.stack(recorded["next_values"], dim=0),
            terminated=torch.stack(recorded["terminated"], dim=0),
            truncated=torch.stack(recorded["truncated"], dim=0),
            episode_start=stack("starts"), initial_hidden=initial_hidden,
            previous_predicted_core=stack("preds"),
            previous_prediction_valid=stack("pred_valid"),
            valid_transition=torch.ones(self.num_envs, horizon, device=self.device, dtype=torch.bool),
            reward_terms={name: torch.stack(values, dim=0) for name, values in reward_terms.items()},
        )

    def on_dynamics_updated(self) -> None:
        """功能：通知 Collector 动力学模型已经改权重，下次采样必须重新建立历史。

        输入：无。输出：无。
        用途：防止拿旧 GRU hidden 配合新 DynamicsModel 推理。
        """
        self.dynamics.invalidate_memories()
        self._ready = False
        self._memory = None
        self._dynamics_now = None
        self._current = None


@dataclass(frozen=True)
class PhysicsPPOStats:
    """一次 MLP PPO 更新的训练信息，给日志和优化稳定性监控使用。"""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    optim_steps: int
    old_policy_max_error: float


class PhysicsPPOUpdater:
    """直接训练第14模块 PhysicsMLPActorCritic，不重写旧 Transformer PPO。

    输入：policy 与 PPOConfig（复用第10模块参数定义）。
    输出：update(rollout) 更新策略/价值网络，返回 PhysicsPPOStats。
    注意：DynamicsModel 的参数不在这里更新；动力学模型另用真实下一帧监督。
    """

    def __init__(self, policy: PhysicsMLPActorCritic,
                 config: PPOConfig | None = None) -> None:
        """功能：建立 PPO 的 Adam 优化器。

        输入：policy 是第14模块的轻量 MLP，config 是 PPO 参数。
        输出：无；得到可以反复调用 update() 的训练器。
        """
        if not isinstance(policy, PhysicsMLPActorCritic):
            raise TypeError("策略必须是 PhysicsMLPActorCritic")
        self.policy = policy
        self.config = config or PPOConfig()
        if not isinstance(self.config, PPOConfig):
            raise TypeError("config 必须是 PPOConfig")
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate,
            eps=self.config.adam_eps,
        )
        self.total_optim_steps = 0

    def _evaluate(self, rollout: PhysicsRollout, ids: torch.Tensor):
        """功能：按原时间顺序，重新计算保存动作的概率和 Critic value。

        输入：rollout 为旧策略采样的数据；ids 为当前 minibatch 的环境编号 [K]。
        输出：new_log_prob、new_value、entropy、mean、std，形状 [K,T] 或 [K,T,N]。
        说明：动力学特征直接使用采样时保存的 z/context，不让 PPO 改 DynamicsModel。
        """
        log_probs, values, entropies, means, stds = [], [], [], [], []
        for t in range(rollout.horizon):
            frame = _choose_envs(rollout.observation.at(t), ids)
            ref = ReferenceObservation(
                rollout.reference.features[:, t].index_select(0, ids),
                rollout.reference.valid_mask[:, t].index_select(0, ids),
            )
            phys = DynamicsEncoding(
                rollout.dynamics_latent[:, t].index_select(0, ids),
                rollout.dynamics_context[:, t].index_select(0, ids),
                hidden=rollout.initial_hidden.index_select(0, ids),
                prediction_error=rollout.previous_predicted_core[:, t].index_select(0, ids).new_zeros(
                    len(ids), rollout.previous_predicted_core.shape[-1]),
            )
            result = self.policy(frame, phys, ControlGoal(motion=ref))
            distribution = result.distribution()
            actions = rollout.action[:, t].index_select(0, ids)
            log_probs.append(distribution.log_prob(actions))
            values.append(result.value)
            entropies.append(distribution.entropy())
            means.append(result.mean)
            stds.append(result.std)
        return (torch.stack(log_probs, dim=1), torch.stack(values, dim=1),
                torch.stack(entropies, dim=1), torch.stack(means, dim=1),
                torch.stack(stds, dim=1))

    def _verify_old_policy(self, rollout: PhysicsRollout) -> float:
        """功能：更新任何参数前，比对旧策略记录与当前策略重算结果。

        输入：刚由 Collector 采集的 rollout。
        输出：最大的动作概率/均值/方差/价值误差（float）。
        说明：过大通常是采样时数据顺序错乱、参数更新或目标错位，不继续训练。
        """
        with torch.no_grad():
            ids = torch.arange(rollout.num_envs, device=rollout.action.device)
            lp, v, _, mean, std = self._evaluate(rollout, ids)
            errors = (
                (lp - rollout.old_log_prob).abs().max(),
                (v - rollout.old_value).abs().max(),
                (mean - rollout.old_mean).abs().max(),
                (std - rollout.old_std).abs().max(),
            )
            max_error = max(float(x.item()) for x in errors)
            if ((lp - rollout.old_log_prob).abs().max() > self.config.logprob_atol
                    or (v - rollout.old_value).abs().max() > self.config.value_atol
                    or (mean - rollout.old_mean).abs().max() > self.config.distribution_atol
                    or (std - rollout.old_std).abs().max() > self.config.distribution_atol):
                raise RuntimeError("旧策略复算失败：动作概率/均值/标准差/价值不一致，停止 PPO")
            return max_error

    def update(self, rollout: PhysicsRollout) -> PhysicsPPOStats:
        """功能：使用真实采样轨迹执行 PPO，修改 MLP Actor/Critic 的权重。

        输入：collect(T) 返回的 PhysicsRollout。
        输出：PhysicsPPOStats，包括 KL、clip_fraction、entropy、梯度范数。
        注意：原始 action 保持不变，PPO 不会反向更新 DynamicsModel 的 GRU。
        """
        if not isinstance(rollout, PhysicsRollout):
            raise TypeError("rollout 必须是 PhysicsRollout")
        self.policy.train()
        old_error = self._verify_old_policy(rollout) if self.config.verify_old_policy else 0.0
        advantages, returns = compute_gae(
            rollout.reward, rollout.old_value.transpose(0, 1), rollout.next_value,
            rollout.terminated, rollout.truncated,
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
        )
        advantages, returns = advantages.transpose(0, 1), returns.transpose(0, 1)
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        latest = (0.0,) * 6
        steps = 0
        early_stop = False
        for epoch in range(self.config.epochs):
            perm = torch.randperm(rollout.num_envs, device=rollout.action.device)
            for start in range(0, rollout.num_envs, self.config.minibatch_envs):
                ids = perm[start:start + self.config.minibatch_envs]
                new_log_prob, value, entropy, _, _ = self._evaluate(rollout, ids)
                old_lp = rollout.old_log_prob.index_select(0, ids)
                advantage = advantages.index_select(0, ids)
                ret = returns.index_select(0, ids)
                old_value = rollout.old_value.index_select(0, ids)
                ratio = torch.exp(new_log_prob - old_lp)
                clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                if self.config.value_clip is not None:
                    clipped_value = old_value + (value - old_value).clamp(
                        -self.config.value_clip, self.config.value_clip,
                    )
                    vloss = torch.maximum((value - ret).square(),
                                          (clipped_value - ret).square()).mean()
                else:
                    vloss = (value - ret).square().mean()
                ent = entropy.mean()
                loss = policy_loss + self.config.value_coef * vloss - self.config.entropy_coef * ent
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("PPO loss 非有限，停止更新")
                approx_kl = (old_lp - new_log_prob).mean()
                if self.config.target_kl is not None and approx_kl.item() > self.config.target_kl:
                    early_stop = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                if not torch.isfinite(grad).item():
                    raise FloatingPointError("PPO 梯度异常，停止更新")
                self.optimizer.step()
                self.total_optim_steps += 1
                steps += 1
                latest = (
                    float(policy_loss.item()), float(vloss.item()), float(ent.item()),
                    float(approx_kl.item()),
                    float(((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()),
                    float(grad.item()),
                )
            if early_stop:
                break
        self.policy.eval()
        return PhysicsPPOStats(*latest, steps, old_error)


@dataclass(frozen=True)
class DynamicsTrainStats:
    """动力学监督更新统计。"""

    loss: float
    terms: dict[str, float]
    optimizer_steps: int


class DynamicsSupervisedUpdater:
    """用真实 (当前状态,动作,下一状态) 训练独立动力学模型。

    不使用假示范或环境真值摩擦做输入。这个优化器与 PPO 优化器分开。
    """

    def __init__(self, dynamics: DynamicsModel, *, learning_rate: float = 1e-4,
                 max_grad_norm: float = 1.0) -> None:
        """功能：创建动力学模型的监督学习优化器。

        输入：dynamics 第14模块；learning_rate 学习率；max_grad_norm 梯度裁剪阈值。
        输出：无。
        """
        if not isinstance(dynamics, DynamicsModel):
            raise TypeError("dynamics 必须是 DynamicsModel")
        if learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("学习率和梯度上限必须大于零")
        self.dynamics = dynamics
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=learning_rate)
        self.steps = 0

    def update(self, rollout: PhysicsRollout) -> DynamicsTrainStats:
        """功能：基于执行后真实状态监督预测误差，更新 Encoder、GRU 与 Predictor。

        输入：真实 PhysicsRollout，包含状态序列、上一步预测和每步动作。
        输出：DynamicsTrainStats，返回损失、分项指标和一次优化器更新次数。

        时序：只根据过去和当前计算 z_t；预测 s_(t+1) 后再和真实下一帧比较。
        """
        self.dynamics.train()
        model = self.dynamics
        encoding = model.forward_sequence(
            rollout.observation,
            episode_start=rollout.episode_start,
            initial_hidden=rollout.initial_hidden.detach(),
            previous_predicted_core=rollout.previous_predicted_core,
            previous_prediction_valid=rollout.previous_prediction_valid,
        )
        prediction = model.predict_next(rollout.observation, rollout.action, encoding.latent)
        loss = model.prediction_loss(
            prediction, rollout.next_observation,
            valid_transition=rollout.valid_transition,
        )
        if not torch.isfinite(loss.total).item():
            raise FloatingPointError("动力学预测损失含 NaN/Inf")
        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        if not torch.isfinite(grad).item():
            raise FloatingPointError("动力学梯度非有限")
        self.optimizer.step()
        self.steps += 1
        model.invalidate_memories()
        model.eval()
        return DynamicsTrainStats(
            float(loss.total.item()),
            {name: float(term.item()) for name, term in loss.terms.items()},
            self.steps,
        )
