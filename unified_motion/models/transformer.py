"""参考条件化MoE主干：复用历史注意力、参考融合和专家，不直接控制电机。

统一测评：python tools/evaluate.py
依赖：前面已完成的 encoder.py、attention.py、reference_fusion.py、moe.py。

重要接口合同：
1. 输入 EncodedMotion，由每款机器人的输入编码器产生；所有 token 最后一维 D 一致。
2. 前向输入 [B,T,D]，用于整段训练；step 输入 [B,D]，用于无梯度逐帧执行。
3. 仅使用每一帧当时已知的参考窗口；不能读取未来真实机器人观测。
4. 整个主干只做一次参考 Cross-Attention，不在各层重复融合，降低冗余和计算。
5. 每层：因果时间注意力（自己加残差）→ MoE（自己加残差），最后 LayerNorm。
6. 跨机器人共享的是本模块权重，不是 Encoder 的权重，也不是 KV Cache。
7. step 必须传入环境 ID 列表，防止同批次行顺序改变后悄悄串用缓存。
8. 每次更新本主干或上游 Encoder 的权重后，必须调用 invalidate_caches()。
9. 返回 balance_loss 只是辅助诊断；训练器决定是否加到 PPO loss。
10. 无 Actor、Critic、动作分布、物理仿真或 PPO；不是训练成功的保证。

本模块是我们自己的组合实现，不等同于官方 HoloMotion 架构或权重格式。
"""

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .attention import AttentionCache, CausalSelfAttention
from .encoder import EncodedMotion
from .moe import MoEOutput, ReferenceRoutedMoE
from .reference_fusion import ReferenceFusion


@dataclass(frozen=True)
class TransformerConfig:
    """所有影响网络结构的设置都集中在此；方便做轻量化和消融实验。"""

    d_model: int = 256                # 隐藏特征宽度，不是机器人关节数量。
    n_layers: int = 2                # 时间注意力 + MoE 的重复层数。
    n_heads: int = 4                 # 查询头数量。
    n_kv_heads: int = 2              # K/V 头数量，少于 Q 头实现 GQA。
    context_len: int = 32           # 每步最多看最近 32 帧（包括当前帧）。
    num_fine_experts: int = 8        # 可被 Router 选择的专家总数。
    num_shared_experts: int = 1      # 每个有效 token 都用的共享专家数量。
    top_k: int = 2                  # 每一步选择几个 Fine Expert。
    expert_hidden_dim: int = 256    # 每个专家的中间宽度。
    router_hidden_dim: int = 256    # Router 小 MLP 的中间宽度。
    check_values: bool = True       # 调试时检查数值；GPU 高频执行可关闭同步检查。


@dataclass
class TransformerOutput:
    """最终得到的仍是网络特征，并不是机器人电机动作。"""

    features: torch.Tensor          # 训练 [B,T,D]，逐帧 [B,D]。
    reference_summary: torch.Tensor # 每个时刻自己对应的参考摘要 [...,D]。
    valid_mask: torch.Tensor        # [...]: True=真实状态，False=补位。
    balance_loss: torch.Tensor      # 标量：各层 MoE 辅助损失均值，不自动加入 PPO。
    router_entropy: torch.Tensor    # [L]：各层 Router 的熵，不是 PPO 动作熵。
    expert_load: torch.Tensor       # [L,E]：各层每个专家的使用比例。
    expert_counts: torch.Tensor     # [L,E]：各层各专家被选中的次数。


@dataclass
class TransformerCache:
    """采样时的运行时缓存：不能保存到 checkpoint，也不能跨机器人或乱序使用。"""

    robot_id: str
    env_ids: tuple[int, ...]
    layer_caches: list[AttentionCache]
    _owner: object = field(repr=False)


class TransformerLayer(nn.Module):
    """一个简洁的预归一化 Transformer 层：时间 Attention + 参考路由 MoE。

    参考 Cross-Attention 已在外部执行一次，本层不会重复创建参考融合器。
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.temporal_attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            context_len=config.context_len,
            check_values=config.check_values,
        )
        self.moe_norm = nn.LayerNorm(config.d_model)
        self.moe = ReferenceRoutedMoE(
            d_model=config.d_model,
            num_fine_experts=config.num_fine_experts,
            num_shared_experts=config.num_shared_experts,
            top_k=config.top_k,
            expert_hidden_dim=config.expert_hidden_dim,
            router_hidden_dim=config.router_hidden_dim,
            check_values=config.check_values,
        )

    def forward(
        self,
        x: torch.Tensor,
        router_context: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, MoEOutput]:
        """整段训练，可求梯度：x 是 [B,T,D]。"""
        # 第05模块返回的是特征增量，这里加一次残差。
        delta = self.temporal_attn(
            self.attn_norm(x), valid_mask=valid_mask, episode_start=episode_start,
        )
        x = torch.where(valid_mask[..., None], x + delta, 0.0)

        # MoE 用参考摘要选择专家，但专家真正处理的是当前状态特征。
        output = self.moe(self.moe_norm(x), router_context, valid_mask=valid_mask)
        x = torch.where(valid_mask[..., None], x + output.delta, 0.0)
        return x, output

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        router_context: torch.Tensor,
        cache: AttentionCache | None,
        *,
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, AttentionCache, MoEOutput]:
        """逐帧执行：[B,D]，只缓存真正跨时间的自注意力。"""
        delta, new_cache = self.temporal_attn.step(
            self.attn_norm(x), cache, episode_start=episode_start,
        )
        x = x + delta
        output = self.moe(self.moe_norm(x), router_context)
        return x + output.delta, new_cache, output


class ReferenceConditionedTransformer(nn.Module):
    """跨机器人共享的 Transformer 主干。每次调用对应一种机器人。

    * 一个 ReferenceFusion 读取当前时刻的已知参考；
    * 多层因果 Attention 处理时间历史；
    * 多层参考路由 MoE 处理特征；
    * 不直接处理 K1/G1 不同的关节维度，输入编码器先完成统一映射。
    """

    SCHEMA_VERSION = 1

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        config = TransformerConfig() if config is None else config
        if not isinstance(config, TransformerConfig):
            raise TypeError("config 必须是 TransformerConfig")
        if type(config.n_layers) is not int or config.n_layers <= 0:
            raise ValueError("n_layers 必须是正整数")
        self.config = config
        self.reference_fusion = ReferenceFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            check_values=config.check_values,
        )
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self._cache_owner = object()

    def get_extra_state(self) -> dict:
        """随网络权重记录配置；相同权重 shape、不同语义也不能直接混用。"""
        config = asdict(self.config)
        # 是否开启数值检查只是运行选项，不影响权重语义：部署可以关掉检查。
        config.pop("check_values")
        return {
            "schema": self.SCHEMA_VERSION,
            "config": config,
            "pipeline": "one_reference_fusion_then_temporal_attention_moe_per_layer",
            "output": "features_not_actions",
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("Transformer checkpoint 配置或接口版本不匹配")
        self.invalidate_caches()

    def invalidate_caches(self) -> None:
        """参数更新/加载后废弃当前缓存。不能只清空引用而不处理旧权重。"""
        self._cache_owner = object()
        for layer in self.layers:
            layer.temporal_attn.invalidate_caches()

    def train(self, mode: bool = True):
        """从采样模式进入训练模式时主动让旧缓存失效。"""
        if type(mode) is not bool:
            raise TypeError("mode 必须是 bool")
        was_training = self.training
        result = super().train(mode)
        if mode and not was_training and hasattr(self, "layers"):
            self.invalidate_caches()
        return result

    @staticmethod
    def _mask(
        value: torch.Tensor | None,
        shape: tuple[int, ...],
        *, name: str,
        device: torch.device,
        default: bool,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(shape, default, dtype=torch.bool, device=device)
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != torch.bool or value.device != device):
            raise ValueError(f"{name} 必须是形状 {shape} 的同设备 bool Tensor")
        return value

    def _check_encoded(self, encoded: EncodedMotion, expected_rank: int) -> tuple[int, ...]:
        if not isinstance(encoded, EncodedMotion):
            raise TypeError("encoded 必须由 MultiRobotEncoder 产生，是 EncodedMotion")
        if not isinstance(encoded.robot_id, str) or not encoded.robot_id.strip():
            raise ValueError("robot_id 必须是非空机器人标识")
        state = encoded.state_token
        if not isinstance(state, torch.Tensor):
            raise TypeError("state_token 必须是 Tensor")
        if state.ndim != expected_rank or any(s == 0 for s in state.shape):
            raise ValueError("state_token 维度或形状错误：训练需 [B,T,D]，推理需 [B,D]")
        if state.shape[-1] != self.config.d_model:
            raise ValueError(f"state_token 最后一维必须是 {self.config.d_model}")
        leading = state.shape[:-1]
        ref = encoded.reference_tokens
        if not isinstance(ref, torch.Tensor) or ref.ndim != expected_rank + 1:
            raise ValueError("reference_tokens 必须比 state_token 多一个 H 轴")
        if (ref.shape[:-2] != leading or ref.shape[-1] != self.config.d_model
                or ref.shape[-2] == 0):
            raise ValueError("reference_tokens 与状态的 batch/time/D 不匹配")
        rm = encoded.reference_mask
        if not isinstance(rm, torch.Tensor) or rm.shape != (*leading, ref.shape[-2]):
            raise ValueError("reference_mask 形状与参考不匹配")
        # dtype / device / 数值 / 参考有效性由 ReferenceFusion 进一步严格验证。
        return leading

    def _collect_output(
        self, features: torch.Tensor, summary: torch.Tensor,
        valid: torch.Tensor, outputs: list[MoEOutput],
    ) -> TransformerOutput:
        # balance_loss 保留梯度，且不自动加进 PPO；其余都是观察量。
        return TransformerOutput(
            features=features,
            reference_summary=summary,
            valid_mask=valid.clone(),
            balance_loss=torch.stack([m.balance_loss for m in outputs]).mean(),
            router_entropy=torch.stack([m.router_entropy for m in outputs]),
            expert_load=torch.stack([m.expert_load for m in outputs]),
            expert_counts=torch.stack([m.expert_counts for m in outputs]),
        )

    def forward(
        self,
        encoded: EncodedMotion,
        *,
        valid_mask: torch.Tensor | None = None,
        episode_start: torch.Tensor | None = None,
    ) -> TransformerOutput:
        """完整时间序列训练：[B,T,D] → [B,T,D]。仅在同一个回合内读过去。"""
        leading = self._check_encoded(encoded, 3)
        b, t = leading
        device = encoded.state_token.device
        valid = self._mask(valid_mask, (b, t), name="valid_mask", device=device, default=True)
        starts = self._mask(episode_start, (b, t), name="episode_start", device=device, default=False)
        if (valid[:, 1:] & ~valid[:, :-1]).any().item():
            raise ValueError("历史时间补位只能在 T 轴末尾")
        if (starts & ~valid).any().item():
            raise ValueError("补位状态不能标记新回合")

        # 只做一次参考融合，避免每层重复运行 Cross-Attention。
        fused = self.reference_fusion(
            encoded.state_token,
            encoded.reference_tokens,
            encoded.reference_mask,
            time_mask=valid,
        )
        x = fused.policy_features
        outputs: list[MoEOutput] = []
        for layer in self.layers:
            x, output = layer(x, fused.reference_summary,
                              valid_mask=valid, episode_start=starts)
            outputs.append(output)
        x = torch.where(valid[..., None], self.final_norm(x), 0.0)
        if self.config.check_values and not torch.isfinite(x).all().item():
            raise ValueError("Transformer 输出包含 NaN 或 Inf")
        return self._collect_output(x, fused.reference_summary, valid, outputs)

    @staticmethod
    def _check_env_ids(ids: tuple[int, ...], batch: int) -> tuple[int, ...]:
        if (not isinstance(ids, tuple) or len(ids) != batch
                or any(type(i) is not int for i in ids) or len(set(ids)) != batch):
            raise ValueError("env_ids 必须是长度 B、元素不重复的整数 tuple，且顺序必须固定")
        return ids

    @torch.no_grad()
    def step(
        self,
        encoded: EncodedMotion,
        *,
        env_ids: tuple[int, ...],
        cache: TransformerCache | None = None,
        episode_start: torch.Tensor | None = None,
    ) -> tuple[TransformerOutput, TransformerCache]:
        """逐帧部署/采样：[B,D] → [B,D]，必须先 .eval()。

        env_ids 要与实际并行环境的行对应；重新分组或改变排列需要重建缓存。
        改变权重后必须先调用 invalidate_caches()，否则数学上无法保证缓存有效。
        """
        if self.training:
            raise RuntimeError("step 用于推理/采样，请先调用 .eval()")
        leading = self._check_encoded(encoded, 2)
        b, = leading
        ids = self._check_env_ids(env_ids, b)
        starts = self._mask(episode_start, (b,), name="episode_start",
                            device=encoded.state_token.device, default=False)
        if cache is not None:
            if not isinstance(cache, TransformerCache) or cache._owner is not self._cache_owner:
                raise ValueError("KV Cache 不属于当前模型或已失效，请重新创建")
            if cache.robot_id != encoded.robot_id:
                raise ValueError("不能把另一种机器人的 KV Cache 拿来使用")
            if cache.env_ids != ids or len(cache.layer_caches) != len(self.layers):
                raise ValueError("环境顺序/数量或缓存层数改变，请重建 KV Cache")

        fused = self.reference_fusion(
            encoded.state_token, encoded.reference_tokens, encoded.reference_mask,
        )
        x = fused.policy_features
        caches: list[AttentionCache] = []
        outputs: list[MoEOutput] = []
        for i, layer in enumerate(self.layers):
            old_cache = cache.layer_caches[i] if cache is not None else None
            x, new_cache, out = layer.step(
                x, fused.reference_summary, old_cache, episode_start=starts,
            )
            caches.append(new_cache)
            outputs.append(out)
        x = self.final_norm(x)
        if self.config.check_values and not torch.isfinite(x).all().item():
            raise ValueError("Transformer 逐帧输出包含 NaN 或 Inf")
        output = self._collect_output(
            x, fused.reference_summary,
            torch.ones(b, dtype=torch.bool, device=x.device), outputs,
        )
        return output, TransformerCache(encoded.robot_id, ids, caches, self._cache_owner)




