"""【历史对照，不是新项目默认入口】
新物理训练使用 api.py / algorithms.sequence_training。
第 09 个文件：多机器人 PPO 高斯 Actor + 独立 Value Critic。

运行：python -m unified_motion.models.actor_critic
依赖已有 MultiRobotEncoder / ReferenceConditionedTransformer，不复制旧模块。

核心合同：
1. 一个共享 Transformer 主干；每种机器人各有动作输出头和动作 log_std。
2. Actor 输出的是**未限幅的原始 action**。训练时按原始 action 记录/重算 log_prob；
   环境负责 action_scale、PD 目标限位和真实力矩限制。本模块绝不悄悄裁剪 action。
3. 每个机器人使用独立的 Critic 输入分支 + 共享 Value Head。Critic 不复用
   Actor 的 Transformer 特征，避免价值损失及可选特权信息污染 Actor 输入。
4. Critic 可选接收仿真中独有的 privileged 信息；Actor 永远无法读取它。
5. forward/evaluate_actions 用 [B,T,...] 进行 PPO 序列训练；act_step 使用 [B,...]
   和逐环境 KV Cache 进行采样/推理。PPO 重算 log_prob 时必须提供正确历史。
6. valid_mask=True 表示有效时刻。序列尾部补位的动作/概率/价值输出严格归零。
7. 这是 PyTorch 策略网络，不实现 PPO 优化、执行器、奖励、仿真或实机部署。
8. 不自动添加 MoE balance_loss；由后续 PPO 训练器显式选择其权重。

与 HoloMotion 是架构思路关联，并非官方原文件/官方权重兼容实现。
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn
from ..models.distributions import GaussianActionHead
from torch.distributions import Independent, Normal

from ..robot_contract import RobotContract, get_robot_contract
from ..reference_observations import ReferenceObservation
from ..models.encoder import EncodedMotion, MultiRobotEncoder
from ..models.transformer import (
    ReferenceConditionedTransformer, TransformerCache, TransformerConfig,
    TransformerOutput,
)


@dataclass
class PolicyValueOutput:
    """整段策略前向结果，动作维数 N 由 robot_id 对应的机器人决定。"""

    robot_id: str
    joint_names: tuple[str, ...]       # 输出动作列的固定策略关节顺序
    mean: torch.Tensor               # [B,T,N]，原始（未裁剪）动作均值
    std: torch.Tensor                # [B,T,N]，各关节随机探索标准差
    values: torch.Tensor             # [B,T]，Critic 预测未来累计回报
    valid_mask: torch.Tensor         # [B,T]，True=有效
    balance_loss: torch.Tensor       # MoE 负载均衡辅助项，不自动参与 PPO
    router_entropy: torch.Tensor     # [L]，Router 熵，不是动作 entropy
    expert_load: torch.Tensor        # [L,E]，专家被选择的比例
    expert_counts: torch.Tensor      # [L,E]，专家被选择次数

    def distribution(self) -> Independent:
        """对 N 个关节的高斯密度相乘；log_prob 自动在关节维求和。"""
        return Independent(Normal(self.mean, self.std), 1)


@dataclass
class ActionEvaluation:
    """PPO 重新评估 rollout 中保存的原始 action，用于计算概率比。"""

    policy: PolicyValueOutput
    log_prob: torch.Tensor           # [B,T]，一个时间步所有关节的联合 log_prob
    entropy: torch.Tensor            # [B,T]，动作分布的联合熵


@dataclass
class ActionStep:
    """环境逐帧采样返回：action 是策略原始输出，不是 PD 目标或力矩。"""

    robot_id: str
    joint_names: tuple[str, ...]
    action: torch.Tensor             # [B,N]
    mean: torch.Tensor               # [B,N]
    std: torch.Tensor                # [B,N]
    log_prob: torch.Tensor           # [B]
    entropy: torch.Tensor            # [B]
    value: torch.Tensor              # [B]
    balance_loss: torch.Tensor       # 标量；采样阶段只用于诊断
    router_entropy: torch.Tensor     # [L]
    expert_load: torch.Tensor        # [L,E]




class MultiRobotValueCritic(nn.Module):
    """独立于 Actor 的 Critic：各机器人先映射到同一隐层，再用共享价值头。

    输入包括当前身体观测和参考窗口的 masked mean；这是轻量基线，不对
    参考 H 时间轴再做复杂 Attention。privileged 只到 Critic，不能进入 Actor。
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        robots: Mapping[str, RobotContract],
        *,
        hidden_dim: int = 256,
        privileged_dims: Mapping[str, int] | None = None,
        check_values: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(robots, Mapping) or not robots:
            raise ValueError("robots 必须是非空映射")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim 必须是正整数")
        self._robots = {k: robots[k] for k in sorted(robots)}
        for k, robot in self._robots.items():
            if not isinstance(k, str) or not k.strip() or not isinstance(robot, RobotContract):
                raise TypeError("机器人 ID 或合同格式错误")
        self._keys = {k: f"robot_{i}" for i, k in enumerate(self._robots)}
        dims = {} if privileged_dims is None else dict(privileged_dims)
        if set(dims) - set(self._robots):
            raise ValueError("privileged_dims 包含没有注册的机器人")
        for k, d in dims.items():
            if type(d) is not int or d < 0:
                raise ValueError(f"机器人 {k} 的 privileged_dim 必须是非负整数")
        self.privileged_dims = {k: dims.get(k, 0) for k in self._robots}
        self.hidden_dim = hidden_dim
        self.check_values = check_values
        self.input_encoders = nn.ModuleDict()
        for k, robot in self._robots.items():
            n = robot.dof
            # 6+3N 来自 observations.py；16+2N 来自 reference_observations.py。
            input_dim = (6 + 3*n) + (16 + 2*n) + self.privileged_dims[k]
            self.input_encoders[self._keys[k]] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
        self.value_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def get_extra_state(self) -> dict:
        return dict(schema=self.SCHEMA_VERSION, hidden_dim=self.hidden_dim,
                    privileged_dims=self.privileged_dims,
                    robots={k: (r.root_name, r.joint_names) for k, r in self._robots.items()},
                    pooling="masked_mean_reference_v1")

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("Critic checkpoint 与机器人、特权信息或池化定义不匹配")

    def forward(
        self, robot_id: str, observation: torch.Tensor,
        reference: ReferenceObservation, *,
        privileged: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B,O]/[B,H,R]→[B] 或 [B,T,O]/[B,T,H,R]→[B,T]。"""
        if robot_id not in self._robots:
            raise KeyError(f"未知机器人 {robot_id!r}")
        if not isinstance(observation, torch.Tensor) or observation.ndim not in (2, 3):
            raise TypeError("observation 必须是 [B,O] 或 [B,T,O]")
        if not isinstance(reference, ReferenceObservation):
            raise TypeError("reference 必须是 ReferenceObservation")
        n = self._robots[robot_id].dof
        lead = observation.shape[:-1]
        if any(s == 0 for s in lead):
            raise ValueError("批次与时间轴不能为空")
        if observation.shape[-1] != 6 + 3*n:
            raise ValueError("Critic 身体观测维数不匹配")
        ref, ref_mask = reference.features, reference.valid_mask
        h = ref.shape[-2] if isinstance(ref, torch.Tensor) and ref.ndim == observation.ndim + 1 else 0
        if not h or ref.shape != (*lead, h, 16 + 2*n):
            raise ValueError("Critic 参考特征的 batch/time/宽度不匹配")
        if not isinstance(ref_mask, torch.Tensor) or ref_mask.shape != (*lead, h):
            raise ValueError("Critic 参考 Mask 形状不匹配")
        p = self.input_encoders[self._keys[robot_id]][0].weight
        mask = torch.ones(lead, dtype=torch.bool, device=p.device) if valid_mask is None else valid_mask
        if not isinstance(mask, torch.Tensor) or mask.shape != lead or mask.dtype != torch.bool or mask.device != p.device:
            raise ValueError("Critic 有效时间 mask 的形状/类型/设备错误")
        if ref_mask.dtype != torch.bool or ref_mask.device != p.device:
            raise ValueError("Critic 参考 Mask 必须是同设备 bool")
        for name, value in (("observation", observation), ("reference", ref)):
            if value.device != p.device or value.dtype != p.dtype:
                raise ValueError(f"Critic {name} 的设备或 dtype 与网络不匹配")
            if self.check_values and not torch.isfinite(value).all().item():
                raise ValueError(f"Critic {name} 包含 NaN/Inf")
        if self.check_values:
            if (mask & ~ref_mask.any(dim=-1)).any().item():
                raise ValueError("有效状态必须至少有一个参考点")
            if observation.ndim == 3 and (mask[:, 1:] & ~mask[:, :-1]).any().item():
                raise ValueError("Critic 的时间补位必须位于末尾")
            if ((ref_mask[..., 1:] & ~ref_mask[..., :-1])
                    & mask.unsqueeze(-1)).any().item():
                raise ValueError("Critic 的参考补位必须位于 H 轴末尾")
        pdim = self.privileged_dims[robot_id]
        if pdim:
            if not isinstance(privileged, torch.Tensor) or privileged.shape != (*lead, pdim):
                raise ValueError(f"Critic 需要 privileged: {(*lead, pdim)}")
            if privileged.device != p.device or privileged.dtype != p.dtype:
                raise ValueError("privileged 的设备或 dtype 错误")
            if self.check_values and not torch.isfinite(privileged).all().item():
                raise ValueError("privileged 包含 NaN/Inf")
        elif privileged is not None:
            raise ValueError("此机器人 privileged_dim=0，不允许额外传入 privileged")
        safe_valid_ref = ref_mask & mask.unsqueeze(-1)
        # 只让真实参考帧进入平均；补位元素的梯度严格为零。
        safe_ref = torch.where(safe_valid_ref.unsqueeze(-1), ref, 0.0)
        counts = safe_valid_ref.sum(dim=-1, keepdim=True).clamp_min(1)
        avg_ref = safe_ref.sum(dim=-2) / counts
        parts = [torch.where(mask.unsqueeze(-1), observation, 0.0), avg_ref]
        if pdim:
            parts.append(torch.where(mask.unsqueeze(-1), privileged, 0.0))
        critic_input = torch.cat(parts, dim=-1)
        y = self.value_head(self.input_encoders[self._keys[robot_id]](critic_input)).squeeze(-1)
        y = torch.where(mask, y, 0.0).to(observation.dtype)
        if self.check_values and not torch.isfinite(y).all().item():
            raise ValueError("Critic 价值预测非有限")
        return y


class MultiRobotActorCritic(nn.Module):
    """完整的 Actor/Critic 网络接口；可作为未来 PPO 训练器的核心。

    Actor: 机器人专用输入 Encoder → 共享 Transformer/MoE → 专用 Gaussian Head。
    Critic: 独立输入分支 → 共享 value head；允许看训练时的 privileged 特征。
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        robots: Mapping[str, RobotContract],
        *,
        transformer_config: TransformerConfig | None = None,
        encoder_hidden_dim: int = 512,
        critic_hidden_dim: int = 256,
        init_action_std: float = 1.0,
        privileged_dims: Mapping[str, int] | None = None,
        check_values: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(robots, Mapping) or not robots:
            raise ValueError("robots 必须是非空映射")
        self._robots = {k: robots[k] for k in sorted(robots)}
        for k, robot in self._robots.items():
            if not isinstance(k, str) or not k.strip() or not isinstance(robot, RobotContract):
                raise TypeError("机器人 ID 或合同格式错误")
        if not isinstance(check_values, bool):
            raise TypeError("check_values 必须是 bool")
        config = TransformerConfig() if transformer_config is None else transformer_config
        if not isinstance(config, TransformerConfig):
            raise TypeError("transformer_config 必须是 TransformerConfig")
        if config.check_values != check_values:
            raise ValueError("transformer_config.check_values 必须与 check_values 一致")
        self.config = config
        self.check_values = check_values
        self.encoder = MultiRobotEncoder(
            self._robots, d_model=config.d_model,
            hidden_dim=encoder_hidden_dim, check_values=check_values,
        )
        self.backbone = ReferenceConditionedTransformer(config)
        self._keys = {k: f"robot_{i}" for i, k in enumerate(self._robots)}
        self.action_heads = nn.ModuleDict({
            self._keys[k]: GaussianActionHead(config.d_model, r.dof, init_action_std, check_values=check_values)
            for k, r in self._robots.items()
        })
        self.critic = MultiRobotValueCritic(
            self._robots, hidden_dim=critic_hidden_dim,
            privileged_dims=privileged_dims, check_values=check_values,
        )

    @property
    def robot_ids(self) -> tuple[str, ...]:
        return tuple(self._robots)

    def get_extra_state(self) -> dict:
        return dict(schema=self.SCHEMA_VERSION, robot_ids=self.robot_ids,
                    joints={k: (r.root_name, r.joint_names) for k, r in self._robots.items()},
                    actor="unsquashed_independent_diagonal_normal", critic="separate_mlp")

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("ActorCritic checkpoint 的机器人或动作语义不匹配")
        self.invalidate_caches()

    def invalidate_caches(self) -> None:
        """优化器更新任何 Actor 权重后必须调用，再用新网络重建历史。"""
        self.backbone.invalidate_caches()

    def train(self, mode: bool = True):
        """从评估模式切回训练模式时，由 backbone 废弃旧 KV Cache。"""
        return super().train(mode)

    def _check_robot(self, robot_id: str) -> RobotContract:
        try:
            return self._robots[robot_id]
        except (KeyError, TypeError):
            raise KeyError(f"未知机器人 {robot_id!r}，可选 {self.robot_ids}") from None

    @staticmethod
    def _check_mask(
        mask: torch.Tensor | None, lead: tuple[int, ...], device: torch.device,
        *, name: str, default: bool,
    ) -> torch.Tensor:
        if mask is None:
            return torch.full(lead, default, dtype=torch.bool, device=device)
        if (not isinstance(mask, torch.Tensor) or mask.shape != lead or mask.device != device
                or mask.dtype != torch.bool):
            raise ValueError(f"{name} 必须是形状 {lead} 的同设备 bool 张量")
        return mask

    def _sequence_encoding(
        self, robot_id: str, observation: torch.Tensor,
        reference: ReferenceObservation, valid: torch.Tensor,
    ) -> EncodedMotion:
        """让上游 Encoder 接受尾部时间补位，同时保持补位不会参与梯度。

        之前 Encoder 规定每个窗口至少有一帧有效参考；但 PPO 序列末尾可以
        只有补位。此处临时给补位行提供 1 个零参考，编码后恢复真实 mask。
        不改老文件，不把这个占位当作真实未来信息。
        """
        if not isinstance(reference, ReferenceObservation):
            raise TypeError("reference 必须是 ReferenceObservation")
        if not isinstance(observation, torch.Tensor) or observation.ndim != 3:
            raise ValueError("PPO forward 需要 [B,T,O]")
        if not isinstance(reference.features, torch.Tensor) or reference.features.ndim != 4:
            raise ValueError("PPO reference.features 必须是 [B,T,H,R]")
        if not isinstance(reference.valid_mask, torch.Tensor) or reference.valid_mask.ndim != 3:
            raise ValueError("PPO reference.valid_mask 必须是 [B,T,H]")
        if any(s == 0 for s in observation.shape):
            raise ValueError("PPO 序列不允许空批次、空时间轴")
        lead = observation.shape[:-1]
        mask = reference.valid_mask
        if mask.shape[:2] != lead or mask.shape[-1] != reference.features.shape[-2]:
            raise ValueError("参考与身体状态的 batch/time/H 不一致")
        if mask.dtype != torch.bool or mask.device != observation.device:
            raise ValueError("参考 mask 必须同设备 bool")
        if (valid[:, 1:] & ~valid[:, :-1]).any().item():
            raise ValueError("补位时刻只能出现在序列末尾")
        if (valid & ~mask.any(dim=-1)).any().item():
            raise ValueError("有效状态至少需要一条参考")
        if self.check_values and (
            not torch.isfinite(observation).all().item()
            or not torch.isfinite(reference.features).all().item()
        ):
            raise ValueError("输入包含 NaN 或 Inf")
        # 给补位时刻放一个临时有效的全零参考，以满足上游 Encoder 的合同。
        dummy = torch.zeros_like(mask)
        dummy[..., 0] = ~valid
        encoder_mask = mask & valid.unsqueeze(-1) | dummy
        safe_obs = torch.where(valid.unsqueeze(-1), observation, 0.0)
        safe_refs = torch.where(valid.unsqueeze(-1).unsqueeze(-1), reference.features, 0.0)
        encoded = self.encoder(robot_id, safe_obs, ReferenceObservation(safe_refs, encoder_mask))
        return EncodedMotion(
            robot_id,
            torch.where(valid.unsqueeze(-1), encoded.state_token, 0.0),
            torch.where(valid.unsqueeze(-1).unsqueeze(-1), encoded.reference_tokens, 0.0),
            mask & valid.unsqueeze(-1),
        )

    def _make_policy(
        self, robot_id: str, backbone: TransformerOutput,
        values: torch.Tensor,
    ) -> PolicyValueOutput:
        """动作分布用未限幅均值+可训练标准差，动作列与关节合同顺序相同。"""
        head = self.action_heads[self._keys[robot_id]]
        mean, std = head(backbone.features)
        valid = backbone.valid_mask
        # 严格清理时间补位：PPO 的样本权重也必须在 loss 阶段乘有效 mask。
        mean = torch.where(valid.unsqueeze(-1), mean, 0.0)
        std = torch.where(valid.unsqueeze(-1), std, 1.0)
        return PolicyValueOutput(
            robot_id, self._robots[robot_id].joint_names,
            mean, std, values, valid.clone(), backbone.balance_loss,
            backbone.router_entropy, backbone.expert_load, backbone.expert_counts,
        )

    def forward(
        self,
        robot_id: str, observation: torch.Tensor,
        reference: ReferenceObservation, *,
        privileged: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        episode_start: torch.Tensor | None = None,
    ) -> PolicyValueOutput:
        """PPO 整段训练（可梯度）：动作均值/方差 [B,T,N]，价值 [B,T]。"""
        self._check_robot(robot_id)
        if not isinstance(observation, torch.Tensor) or observation.ndim != 3:
            raise ValueError("PPO forward 输入必须为 [B,T,O]")
        lead = observation.shape[:-1]
        valid = self._check_mask(valid_mask, lead, observation.device, name="valid_mask", default=True)
        starts = self._check_mask(episode_start, lead, observation.device,
                                  name="episode_start", default=False)
        if (starts & ~valid).any().item():
            raise ValueError("不能在补位时刻标记新回合")
        encoded = self._sequence_encoding(robot_id, observation, reference, valid)
        backbone_output = self.backbone(encoded, valid_mask=valid, episode_start=starts)
        values = self.critic(robot_id, observation, reference, privileged=privileged,
                             valid_mask=valid)
        return self._make_policy(robot_id, backbone_output, values)

    def evaluate_actions(
        self,
        robot_id: str, observation: torch.Tensor,
        reference: ReferenceObservation, actions: torch.Tensor, *,
        privileged: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        episode_start: torch.Tensor | None = None,
    ) -> ActionEvaluation:
        """PPO 更新时把原始 action 传回，重算与采样时一致的高斯 log_prob。"""
        policy = self.forward(robot_id, observation, reference,
                              privileged=privileged, valid_mask=valid_mask,
                              episode_start=episode_start)
        if not isinstance(actions, torch.Tensor) or actions.shape != policy.mean.shape:
            raise ValueError(f"actions 必须是 {tuple(policy.mean.shape)} 的未限幅原始动作")
        if actions.device != policy.mean.device or actions.dtype != policy.mean.dtype:
            raise ValueError("actions 必须与策略均值同设备、同 dtype")
        if self.check_values and not torch.isfinite(actions).all().item():
            raise ValueError("actions 中出现 NaN/Inf")
        # 对补位位置喂 0，避免无意义数据参与概率计算；其梯度也会被屏蔽。
        safe_actions = torch.where(policy.valid_mask.unsqueeze(-1), actions, 0.0)
        dist = policy.distribution()
        log_prob = torch.where(policy.valid_mask, dist.log_prob(safe_actions), 0.0)
        entropy = torch.where(policy.valid_mask, dist.entropy(), 0.0)
        return ActionEvaluation(policy, log_prob, entropy)

    def value(
        self, robot_id: str, observation: torch.Tensor,
        reference: ReferenceObservation, *, privileged: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """单独获取价值预测，便于 PPO 为最后一个观测计算 bootstrap value。"""
        self._check_robot(robot_id)
        return self.critic(robot_id, observation, reference,
                           privileged=privileged, valid_mask=valid_mask)

    @torch.no_grad()
    def act_step(
        self, robot_id: str, observation: torch.Tensor,
        reference: ReferenceObservation, *,
        env_ids: tuple[int, ...],
        cache: TransformerCache | None = None,
        episode_start: torch.Tensor | None = None,
        privileged: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[ActionStep, TransformerCache]:
        """逐帧采样，用给环境的原始 action 同时计算 log_prob，避免 PPO 错位。"""
        self._check_robot(robot_id)
        if self.training:
            raise RuntimeError("act_step 仅用于采样/推理，调用前请 .eval()")
        if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
            raise ValueError("act_step 的观测必须是 [B,O]")
        if type(deterministic) is not bool:
            raise TypeError("deterministic 必须是 bool")
        encoded = self.encoder(robot_id, observation, reference)
        backbone_out, new_cache = self.backbone.step(
            encoded, env_ids=env_ids, cache=cache, episode_start=episode_start,
        )
        values = self.critic(robot_id, observation, reference, privileged=privileged)
        policy = self._make_policy(robot_id, backbone_out, values)
        dist = policy.distribution()
        action = policy.mean.clone() if deterministic else dist.sample()
        log_prob, entropy = dist.log_prob(action), dist.entropy()
        return ActionStep(
            robot_id, policy.joint_names, action, policy.mean, policy.std,
            log_prob, entropy, values, policy.balance_loss,
            policy.router_entropy, policy.expert_load,
        ), new_cache




