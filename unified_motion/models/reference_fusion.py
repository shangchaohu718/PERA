"""参考融合：当前状态逐时刻查询自己的目标，返回特征与路由摘要。

统一测评：python tools/evaluate.py
不修改前五个模块；自研实现，不是 HoloMotion 官方权重兼容层。

输入合同：
- state_features：[B,D] 或 [B,T,D]，可接 Encoder 或因果历史模块的输出。
- reference_tokens：[B,H,D] 或 [B,T,H,D]，接 Encoder.reference_tokens。
- reference_mask：对应 [...,H]，True=有效；time_mask：对应 [B] 或 [B,T]。
- 每个真实时刻至少一个有效参考；无效补位只能在对应轴末尾。
- 全补位的时间行可以没有参考，但不能用它冒充“无参考任务”。
- T 不参与交叉注意力匹配，每个 (环境,时刻) 只查询自己当时的 H 个目标。
- H 是已经提供的目标，不是未来真实观测；参考已包含时间偏移特征。
- 本模块不缓存参考：第03模块的参考以当前机身为坐标基准，会随状态变化。
- 已包含交叉注意力的归一化与残差；外层不要再次加同一条残差。
- 不含 FFN/MoE、Actor、奖励或 PPO；不是完整控制器，不保证运动效果。
- 模型保持 float32/64，可外部 autocast；check_values=True 的数值检查有同步开销。
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FusedReference:
    """返回的是网络特征；reference_summary 不是专家概率。"""

    policy_features: torch.Tensor      # [...,D]：状态 + 读出的参考信息。
    reference_summary: torch.Tensor    # [...,D]：仅由有效参考生成，供后续路由使用。
    time_mask: torch.Tensor            # [...]：True=真实状态，False=时间补位。


class ReferenceFusion(nn.Module):
    """逐时刻交叉注意力；K1/G1 编码到同一 D 后可复用这一层权重。

    使用少量 K/V 头，并显式展开给 SDPA，优先保证兼容，不承诺最佳 GPU 速度。
    reference_summary 是未归一化的输入 reference_tokens 的 masked mean：
    仅辅助后续 Router；策略主分支仍使用全部有效参考的交叉注意力结果。
    """

    SCHEMA_VERSION = 1

    def __init__(
        self, d_model: int = 256, n_heads: int = 4, n_kv_heads: int = 2,
        *, check_values: bool = True,
    ) -> None:
        """创建逐时刻查询参考目标的融合层。

        输入：d_model统一特征宽度、n_heads查询头数、n_kv_heads共享KV头数、check_values检查开关。
        输出：无；状态生成Q，参考生成K/V，本层内部包含归一化和一次残差。"""
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads),
                            ("n_kv_heads", n_kv_heads)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if d_model < 2 or d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError("D至少为2；D能被Q头数整除；Q头数能被KV头数整除")
        if type(check_values) is not bool:
            raise TypeError("check_values 必须是 bool")
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.check_values = check_values
        self.query_norm = nn.LayerNorm(d_model)
        self.reference_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def get_extra_state(self) -> dict:
        """相同权重形状也可能有不同头划分；保存语义合同，严格检查加载。"""
        return dict(schema=self.SCHEMA_VERSION, d_model=self.d_model,
                    n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
                    residual="state_plus_cross_attention", time_mixing=False,
                    summary="masked_mean_input_reference", reference_time="already_encoded")

    def set_extra_state(self, state: dict) -> None:
        """加载参考融合权重时核对结构定义。

        输入：state checkpoint元数据字典。输出：无；不同头数/输入语义不能只因权重形状相同就加载。"""
        if state != self.get_extra_state():
            raise ValueError("参考融合 checkpoint 的结构或接口版本不匹配")

    def _check_tensor(
        self, name: str, value: torch.Tensor, shape: tuple[int, ...],
        *, boolean: bool = False,
    ) -> None:
        """拒绝错形状、错设备、错精度或坏数值。

        输入：name字段名、value张量、shape期望大小、boolean类型标志。输出：无；不自动广播或偷偷补齐错误数据。"""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} 必须是 Tensor")
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} 应为 {tuple(shape)}，实际 {tuple(value.shape)}")
        p = self.q_proj.weight
        if value.device != p.device:
            raise ValueError(f"{name} 与融合模块不在同一设备")
        allowed = {torch.bool} if boolean else {p.dtype}
        if not boolean and p.dtype == torch.float32 and torch.is_autocast_enabled(p.device.type):
            allowed.add(torch.get_autocast_dtype(p.device.type))
        if value.dtype not in allowed:
            raise TypeError(f"{name} 的类型与模型/autocast 不匹配")
        if self.check_values and not boolean and not torch.isfinite(value).all().item():
            raise ValueError(f"{name} 包含 NaN 或 Inf，包括补位区")

    def forward(
        self,
        state_features: torch.Tensor,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor,
        *, time_mask: torch.Tensor | None = None,
    ) -> FusedReference:
        """整段和逐帧都用同一 forward，无隐藏记忆，不需要 step 或 reset。

        历史注意力的 reset/因果屏蔽仍由第05模块处理，本层不能修复上游泄漏。
        模块不能识别张量来自哪个机器人；调用方必须对齐机器人、时刻及参考。
        """
        if not isinstance(state_features, torch.Tensor):
            raise TypeError("state_features 必须是 Tensor")
        if state_features.ndim not in (2, 3) or any(s == 0 for s in state_features.shape):
            raise ValueError("状态必须是非空 [B,D] 或 [B,T,D]")
        if not isinstance(reference_tokens, torch.Tensor):
            raise TypeError("reference_tokens 必须是 Tensor")
        if reference_tokens.ndim != state_features.ndim + 1 or reference_tokens.shape[-2] <= 0:
            raise ValueError("参考必须比状态多一个非空的 H 轴")
        if self.q_proj.weight.dtype not in (torch.float32, torch.float64):
            raise TypeError("权重保持 float32/64；混合精度请用外部 autocast")
        leading = state_features.shape[:-1]
        h, d = reference_tokens.shape[-2], self.d_model
        self._check_tensor("state_features", state_features, (*leading, d))
        self._check_tensor("reference_tokens", reference_tokens, (*leading, h, d))
        self._check_tensor("reference_mask", reference_mask, (*leading, h), boolean=True)
        if time_mask is None:
            time_mask = torch.ones(leading, dtype=torch.bool, device=state_features.device)
        self._check_tensor("time_mask", time_mask, leading, boolean=True)
        valid = reference_mask & time_mask.unsqueeze(-1)
        if self.check_values:
            if (time_mask & ~valid.any(dim=-1)).any().item():
                raise ValueError("真实状态至少需要一个有效参考；无参考任务需独立接口")
            if (valid[..., 1:] & ~valid[..., :-1]).any().item():
                raise ValueError("参考补位只能位于 H 轴末尾")
            if time_mask.ndim == 2 and (time_mask[:, 1:] & ~time_mask[:, :-1]).any().item():
                raise ValueError("时间补位只能位于 T 轴末尾")

        # 先清零再进网络，有限的补位垃圾数据不能影响输出或梯度。
        x = torch.where(time_mask.unsqueeze(-1), state_features, 0.0)
        refs = torch.where(valid.unsqueeze(-1), reference_tokens, 0.0)
        # 仅把 B/T 变成独立批次 S；绝不把不同时间点的 H 串成一条长序列。
        s = math.prod(leading)
        q = self.q_proj(self.query_norm(x)).reshape(s, self.n_heads, 1, self.head_dim)
        r = self.reference_norm(refs)
        r = torch.where(valid.unsqueeze(-1), r, 0.0)  # 屏蔽 LayerNorm 的可学习偏置。
        k = self.k_proj(r).reshape(s, h, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(r).reshape(s, h, self.n_kv_heads, self.head_dim).transpose(1, 2)
        groups = self.n_heads // self.n_kv_heads
        if groups != 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        allowed = valid.reshape(s, h)
        # 全补位时间行临时允许读第0个零参考，避免全屏蔽 softmax；最终输出清零。
        first_slot = torch.arange(h, device=x.device) == 0
        allowed = allowed | (~allowed.any(dim=-1, keepdim=True) & first_slot)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None, None, :],
            dropout_p=0.0, is_causal=False,  # H是已知目标，不对它套历史三角屏蔽。
        )
        delta = self.out_proj(y.reshape(*leading, d)).to(x.dtype)
        fused = torch.where(time_mask.unsqueeze(-1), x + delta, 0.0)

        # 辅助路由摘要：仅沿 H 求平均，不跨 T；先除再求和避免低精度累加溢出。
        # 不 detach，未来 Router 的训练信号可以回到参考 Encoder。
        sum_dtype = torch.float64 if refs.dtype == torch.float64 else torch.float32
        count = valid.sum(dim=-1).clamp_min(1).to(sum_dtype)
        summary = (refs.to(sum_dtype) / count[..., None, None]).sum(dim=-2).to(x.dtype)
        summary = torch.where(time_mask.unsqueeze(-1), summary, 0.0)
        if self.check_values:
            for name, value in (("policy_features", fused), ("reference_summary", summary)):
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"{name} 出现 NaN 或 Inf")
        return FusedReference(fused, summary, time_mask.clone())




