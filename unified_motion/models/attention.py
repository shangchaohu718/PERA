"""因果历史注意力：GQA、RoPE、回合隔离和逐环境KV缓存。

统一测评：python tools/evaluate.py
训练输入 [B,T,D]；逐帧输入 [B,D]；输出宽度仍为 D，不是关节动作。
只处理真实状态的 T 轴，不把参考目标 H 轴混进时间轴。
这是自研基础模块，不是完整 Transformer，也不兼容官方 HoloMotion 权重。

合同：
- context_len 包含当前帧；一个 token 对应一个等间隔控制时刻。
- episode_start=True 表示“当前输入已是新回合第一帧”，不是本步执行后 done。
- valid_mask=True 表示真实时间点；允许全补位行，补位只能在序列末尾。
- forward 从给出的历史起点计算；PPO 切片必须携带所需前文并单独屏蔽 burn-in 损失。
- step 仅用于无梯度采样/推理。同一缓存不能跨网络层、环境顺序或机器人分组复用。
- 更新任意上游/本层权重后必须作废并用新权重重建缓存，不能使用旧 KV 更新 PPO。
- 缓存不进入 state_dict。一次 step 会原地更新缓存，不可并发共享该对象。
- 权重保持 float32/64；可用外部 autocast，缓存建立后不得切换精度。
- GQA 显式展开 K/V 后调用 SDPA，优先保证兼容；不声称已优化到最快内核。
"""

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AttentionCache:
    """单层、单批环境的运行时记忆；由 step 自动创建，不要手工修改。"""

    keys: torch.Tensor             # [B, KV头数, C, 每头宽度]，已应用 RoPE。
    values: torch.Tensor           # 同上；只存 KV 头，不复制成 Q 头数。
    valid: torch.Tensor            # [B,C]，槽位是否属于当前回合。
    next_position: torch.Tensor    # [B]，各环境下一帧的回合内编号。
    slot: int                      # 环形数组下一次写入的位置。
    _owner: object = field(repr=False)  # 防止错误复用另一个层或过期缓存。


class CausalSelfAttention(nn.Module):
    """输入特征 → 查阅当前及历史信息 → 同宽度特征。

    不包含残差、归一化、FFN/MoE；这些在 Transformer block 中组合。
    output_gate 是特征门控，不是 MoE Router，也不是动作/力矩安全限幅。
    """

    SCHEMA_VERSION = 1

    def __init__(
        self, d_model: int = 256, n_heads: int = 4, n_kv_heads: int = 2,
        *, context_len: int = 32, rope_base: float = 10000.0,
        output_gate: bool = True, check_values: bool = True,
    ) -> None:
        """创建可读取历史的注意力层。

        输入：d_model宽度D；n_heads查询头；n_kv_heads共享KV头；context_len每层窗口含当前帧；rope_base位置编码基数；output_gate是否调节特征；check_values检查开关。
        输出：无；创建投影权重。没有FFN、没有残差、没有机器人动作输出。"""
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads),
                            ("n_kv_heads", n_kv_heads), ("context_len", context_len)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError("d_model 必须是 n_heads 的倍数，n_heads 必须是 n_kv_heads 的倍数")
        if (d_model // n_heads) % 2:
            raise ValueError("RoPE 要求每头宽度为偶数")
        if isinstance(rope_base, bool) or not isinstance(rope_base, (int, float)):
            raise TypeError("rope_base 必须是实数")
        if not math.isfinite(rope_base) or rope_base <= 1:
            raise ValueError("rope_base 必须有限且大于 1")
        if type(output_gate) is not bool or type(check_values) is not bool:
            raise TypeError("output_gate 和 check_values 必须是 bool")
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.context_len, self.rope_base = context_len, float(rope_base)
        self.check_values = check_values
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False) if output_gate else None
        if self.gate_proj is not None:
            nn.init.zeros_(self.gate_proj.weight)  # 初始 sigmoid(0)=0.5，之后可学习。
        # 整数索引不会因模块转成低精度而破坏 RoPE 的频率定义。
        self.register_buffer("pair_dims", torch.arange(0, self.head_dim, 2), persistent=False)
        self._cache_owner = object()

    def get_extra_state(self) -> dict:
        """记录注意力配置，防止同形状权重被用在不同头划分上。

        输入：无。输出：结构字典，包含头数、窗口和RoPE约定；不保存运行时历史缓存。"""
        return dict(schema=self.SCHEMA_VERSION, d_model=self.d_model,
                    n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
                    context_len=self.context_len, rope_base=self.rope_base,
                    output_gate=self.gate_proj is not None, rope="adjacent_pairs_v1")

    def invalidate_caches(self) -> None:
        """使本层之前生成的缓存全部过期。

        输入：无。输出：无；后面再使用旧缓存会被拒绝。调用方要用新权重重算真实历史，不是只把机器人记忆清零。"""
        self._cache_owner = object()

    def set_extra_state(self, state: dict) -> None:
        """加载权重时检查结构并让旧缓存失效。

        输入：checkpoint结构字典。输出：无；不兼容就报错，避免加载后混用旧KV。"""
        if state != self.get_extra_state():
            raise ValueError("注意力 checkpoint 的结构或接口版本不匹配")
        self.invalidate_caches()

    def _check_input(self, x: torch.Tensor, ndim: int) -> None:
        """检查注意力输入的维数与数值。

        输入：x数据、ndim期望维数（整段3，单帧2）。输出：无；设备/宽度/精度不一致立即拒绝。"""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x 必须是 Tensor")
        if x.ndim != ndim or x.shape[-1] != self.d_model or any(s == 0 for s in x.shape):
            raise ValueError(f"x 应是非空 {ndim} 维张量，最后一维={self.d_model}")
        p = self.q_proj.weight
        if x.device != p.device:
            raise ValueError("输入与注意力网络不在同一设备")
        if p.dtype not in (torch.float32, torch.float64):
            raise TypeError("权重保持 float32/64；混合精度使用 autocast")
        allowed_dtypes = {p.dtype}
        if p.dtype == torch.float32 and torch.is_autocast_enabled(x.device.type):
            allowed_dtypes.add(torch.get_autocast_dtype(x.device.type))
        if x.dtype not in allowed_dtypes:
            raise TypeError("输入类型与权重/autocast 不一致")
        if self.check_values and not torch.isfinite(x).all().item():
            raise ValueError("输入包含 NaN 或 Inf")

    def _bool_mask(self, mask, shape, x, *, name, default):
        """创建或核对有效性标记。

        输入：mask或None、shape期望形状、x参照设备、name错误提示、default默认值。输出：同设备bool张量，True代表该条件成立。"""
        if mask is None:
            return torch.full(shape, default, dtype=torch.bool, device=x.device)
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{name} 必须是 Tensor")
        if tuple(mask.shape) != tuple(shape) or mask.device != x.device:
            raise ValueError(f"{name} 形状或设备不匹配，应为 {tuple(shape)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} 必须是 bool")
        return mask

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """把时间编号加入Q/K，帮助网络区分先后位置。

        输入：x [B,头,T,头宽]；positions [B,T] 回合内编号。输出：同形状的位置编码结果。这里旋转的是特征，不是机器人身体。"""
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            inv_freq = self.rope_base ** (-self.pair_dims.to(dtype) / self.head_dim)
            angles = positions.to(dtype).unsqueeze(-1) * inv_freq
            cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
            even, odd = x[..., 0::2].to(dtype), x[..., 1::2].to(dtype)
            rotated = torch.stack((even*cos - odd*sin, even*sin + odd*cos), dim=-1)
            return rotated.flatten(-2).to(x.dtype)

    def _project(self, x, positions):
        """由每帧特征生成查询Q、线索K和内容V。

        输入：x [B,T,D]、positions [B,T]。输出：Q为[B,Q头,T,头宽]，K/V使用KV头数；Q/K已经加上RoPE。"""
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return self._rope(q, positions), self._rope(k, positions), v

    def _attend(self, x, q, k, v, allowed):
        # 多个 Q 头共用一组 K/V。显式展开会有临时内存开销，但缓存仍是小的。
        """按允许读取的历史位置做注意力，再投影回D维。

        输入：x用于门控，q/k/v投影特征，allowed [B,T,S]，True表示可读。
        输出：[B,T,D]特征增量；没有把原来的x加回来，残差由外层处理。"""
        groups = self.n_heads // self.n_kv_heads
        if groups != 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        # SDPA 的 bool mask：True=允许读取。已经自己构建 mask，不再开 is_causal。
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed.unsqueeze(1), dropout_p=0.0, is_causal=False,
        )
        y = y.transpose(1, 2).reshape(*x.shape[:-1], self.d_model)
        if self.gate_proj is not None:
            y = y * torch.sigmoid(self.gate_proj(x))
        return self.out_proj(y)

    def _check_output(self, y):
        """防止非有限特征流入后续网络。

        输入：y 注意力输出。输出：无；check_values打开且有NaN/Inf时立即报错。"""
        if self.check_values and not torch.isfinite(y).all().item():
            raise ValueError("注意力输出包含 NaN 或 Inf")

    def forward(
        self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None,
        episode_start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """整段训练：[B,T,D] → [B,T,D]，允许梯度通过整个给定序列。"""
        self._check_input(x, 3)
        b, t, _ = x.shape
        valid = self._bool_mask(valid_mask, (b, t), x, name="valid_mask", default=True)
        starts = self._bool_mask(episode_start, (b, t), x, name="episode_start", default=False)
        if self.check_values:
            if (valid[:, 1:] & ~valid[:, :-1]).any().item():
                raise ValueError("时间补位只能在末尾")
            if (starts & ~valid).any().item():
                raise ValueError("补位帧不能标记为新回合")
        x = torch.where(valid.unsqueeze(-1), x, 0.0)
        index = torch.arange(t, device=x.device)
        # 每次 reset 重新从位置0开始；未显式标记的片段开头也从0开始。
        last_start = torch.where(starts, index, 0).cummax(dim=-1).values
        positions = index - last_start
        segments = starts.long().cumsum(dim=-1)
        q, k, v = self._project(x, positions)
        distance = index[:, None] - index[None, :]
        causal_window = (distance >= 0) & (distance < self.context_len)
        allowed = causal_window.unsqueeze(0) & (segments[:, :, None] == segments[:, None, :])
        allowed = allowed & valid[:, :, None] & valid[:, None, :]
        # 补位 query 暂时只读自己，避免全屏蔽行；最后再把其输出清零。
        allowed = allowed | (~valid[:, :, None] & torch.eye(t, dtype=torch.bool, device=x.device))
        y = self._attend(x, q, k, v, allowed)
        y = torch.where(valid.unsqueeze(-1), y, 0.0)
        self._check_output(y)
        return y

    def _check_cache(self, cache, x):
        """核对KV缓存是否真的属于本层、这批环境和当前精度。

        输入：cache 运行时记忆；x 当前[B,D]特征。输出：无；缓存过期、宽度/设备变化或带梯度时拒绝。"""
        if not isinstance(cache, AttentionCache) or cache._owner is not self._cache_owner:
            raise ValueError("缓存属于其他层或已失效；请重新建立")
        if type(cache.slot) is not int or not 0 <= cache.slot < self.context_len:
            raise ValueError("缓存写入槽位错误")
        dtype = self.q_proj.weight.dtype
        if dtype == torch.float32 and torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
        b = x.shape[0]
        shape = (b, self.n_kv_heads, self.context_len, self.head_dim)
        for name, tensor, expected_shape, expected_dtype in (
            ("keys", cache.keys, shape, dtype), ("values", cache.values, shape, dtype),
            ("valid", cache.valid, (b, self.context_len), torch.bool),
            ("next_position", cache.next_position, (b,), torch.long),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                raise ValueError(f"缓存 {name} 形状不匹配")
            if tensor.device != x.device or tensor.dtype != expected_dtype:
                raise ValueError(f"缓存 {name} 的设备或精度已改变，请重建")
            if tensor.requires_grad:
                raise ValueError("推理缓存不能携带训练梯度")
            if self.check_values and not torch.isfinite(tensor).all().item():
                raise ValueError(f"缓存 {name} 含非有限值")
        if self.check_values and (cache.next_position < 0).any().item():
            raise ValueError("缓存时间编号不能为负数")

    @torch.no_grad()
    def step(
        self, x: torch.Tensor, cache: AttentionCache | None = None,
        *, episode_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AttentionCache]:
        """逐帧采样/推理：[B,D] → [B,D]；必须先 .eval()，返回更新后的缓存。"""
        if self.training:
            raise RuntimeError("step 只用于采样/推理，请先 .eval()；训练用 forward")
        self._check_input(x, 2)
        b = x.shape[0]
        starts = self._bool_mask(episode_start, (b,), x, name="episode_start", default=False)
        if cache is not None:
            self._check_cache(cache, x)
            positions = torch.where(starts, 0, cache.next_position)
        else:
            positions = torch.zeros(b, dtype=torch.long, device=x.device)
        q, k, v = self._project(x.unsqueeze(1), positions.unsqueeze(-1))
        if cache is None:
            shape = (b, self.n_kv_heads, self.context_len, self.head_dim)
            cache = AttentionCache(
                k.new_zeros(shape), v.new_zeros(shape),
                torch.zeros(b, self.context_len, dtype=torch.bool, device=x.device),
                torch.zeros(b, dtype=torch.long, device=x.device), 0, self._cache_owner,
            )
        # 只清空需要 reset 的环境，其他机器人仍保留历史。
        cache.valid.masked_fill_(starts[:, None], False)
        cache.keys.masked_fill_(starts[:, None, None, None], 0.0)
        cache.values.masked_fill_(starts[:, None, None, None], 0.0)
        cache.keys[:, :, cache.slot, :].copy_(k[:, :, 0, :])
        cache.values[:, :, cache.slot, :].copy_(v[:, :, 0, :])
        cache.valid[:, cache.slot] = True
        # 缓存中只有当前与过去，不能套非方形的普通三角 mask。
        y = self._attend(x.unsqueeze(1), q, cache.keys, cache.values,
                         cache.valid.unsqueeze(1)).squeeze(1)
        cache.next_position.copy_(positions + 1)
        cache.slot = (cache.slot + 1) % self.context_len
        self._check_output(y)
        return y, cache




