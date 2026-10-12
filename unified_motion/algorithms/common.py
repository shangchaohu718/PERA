"""共享 PPO 配置与 GAE：MLP、Dense、MoE 使用同一种奖励/终止计算约定。

这里不读取仿真，不构建 Actor，不保存 rollout。每条经验必须提供真正结束前的 next_value。
"""
from dataclasses import dataclass
import math
import torch


def diagonal_gaussian_kl(old_mu, old_sigma, new_mu, new_sigma):
    """Differentiable KL(old || new), summed over the action axis."""
    old_mu, old_sigma = old_mu.detach(), old_sigma.detach()
    return (torch.log(new_sigma / old_sigma)
            + (old_sigma.square() + (old_mu-new_mu).square())/(2*new_sigma.square())
            - .5).sum(dim=-1)

def learning_rate_scale(progress: float, *, schedule: str = "cosine",
                        min_ratio: float = 0.1) -> float:
    """按本次训练进度返回确定性学习率倍率；不需要额外 scheduler 状态。"""
    _require_number("progress", progress, lower=0, upper=1)
    _require_number("min_ratio", min_ratio, lower=0, upper=1, strict_lower=True)
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
    if schedule == "cosine":
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError("lr_schedule必须是constant/linear/cosine")

def _require_number(name: str, value: float, *, lower: float | None = None,
                    upper: float | None = None, strict_lower: bool = False) -> None:
    """检查一个训练超参数是否为合理实数。

    输入：name名称、value数值、lower/upper上下界、strict_lower是否排除下边界。输出：无；拒绝bool、NaN/Inf和越界值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    if lower is not None and (value <= lower if strict_lower else value < lower):
        raise ValueError(f"{name} 过小")
    if upper is not None and value > upper:
        raise ValueError(f"{name} 过大")


@dataclass(frozen=True)
class PPOConfig:
    """所有 PPO 训练参数集中管理；不在代码深处藏 reward 或策略限制。"""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.005
    balance_coef: float = 0.0
    value_clip: float | None = 0.2
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatch_envs: int = 32
    normalize_advantage: bool = True
    verify_old_policy: bool = True
    logprob_atol: float = 1e-3
    value_atol: float = 1e-3
    distribution_atol: float = 1e-3
    target_kl: float | None = None

    def __post_init__(self) -> None:
        """创建PPO配置时一次性检查所有超参数。

        输入：配置对象字段。输出：无；学习率/阈值/批次非法会在开始采样前报错，而不是训练中途才发现。"""
        for name in ("gamma", "gae_lambda"):
            _require_number(name, getattr(self, name), lower=0, upper=1, strict_lower=True)
        for name in ("learning_rate", "adam_eps", "max_grad_norm",
                     "logprob_atol", "value_atol", "distribution_atol"):
            _require_number(name, getattr(self, name), lower=0, strict_lower=True)
        _require_number("clip_ratio", self.clip_ratio, lower=0, upper=1, strict_lower=True)
        for name in ("value_coef", "entropy_coef", "balance_coef"):
            _require_number(name, getattr(self, name), lower=0)
        for name in ("value_clip", "target_kl"):
            value = getattr(self, name)
            if value is not None:
                _require_number(name, value, lower=0, strict_lower=True)
        for name in ("epochs", "minibatch_envs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("normalize_advantage", "verify_old_policy"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} 必须是 bool")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *, gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按时间倒序计算 GAE，输入全部为 [T,B]，返回 advantage 和 returns。

    terminated：环境真正终止（例如失败），没有未来价值。
    truncated：因时间上限截断，允许用 final observation 的 value bootstrap，
               但优势递归不能跨越到 reset 后的新回合。
    每一帧单独提供 next_values，避免跨回合取错 reset 后的 V(s)。
    """
    _require_number("gamma", gamma, lower=0, upper=1, strict_lower=True)
    _require_number("gae_lambda", gae_lambda, lower=0, upper=1, strict_lower=True)
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 2 or any(s == 0 for s in rewards.shape):
        raise ValueError("rewards 必须是非空 [T,B] Tensor")
    if not rewards.is_floating_point():
        raise TypeError("rewards 必须是浮点 Tensor")
    shape = rewards.shape
    for name, value in (("values", values), ("next_values", next_values)):
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"{name} 的形状必须与 rewards 相同")
        if value.device != rewards.device or value.dtype != rewards.dtype:
            raise ValueError(f"{name} 的设备和 dtype 必须与 rewards 相同")
    for name, value in (("terminated", terminated), ("truncated", truncated)):
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.bool:
            raise ValueError(f"{name} 必须是同形状的 bool Tensor")
        if value.device != rewards.device:
            raise ValueError(f"{name} 与 rewards 不在同一设备")
    if (terminated & truncated).any().item():
        raise ValueError("terminated 与 truncated 不能在同一步同时为 True")
    if not all(torch.isfinite(x).all().item() for x in (rewards, values, next_values)):
        raise ValueError("GAE 输入出现 NaN / Inf")

    # 有真实终止时，下一状态价值为 0；时间截断则仍使用 final observation 价值。
    bootstrap = (~terminated).to(rewards.dtype)
    # 任意一种回合结束，都不能将下一段新回合的优势传回来。
    continuation = (~(terminated | truncated)).to(rewards.dtype)
    delta = rewards + gamma * bootstrap * next_values - values
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for i in range(shape[0] - 1, -1, -1):
        carry = delta[i] + gamma * gae_lambda * continuation[i] * carry
        advantages[i] = carry
    returns = advantages + values
    if not torch.isfinite(returns).all().item():
        raise ValueError("GAE 输出出现非有限值")
    return advantages, returns
