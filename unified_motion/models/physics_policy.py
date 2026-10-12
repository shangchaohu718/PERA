"""物理条件化时序策略：Dense和MoE共用目标接口、独立动力学输入与动作分布。

【这个文件解决什么问题】
第14阶段已经有 DynamicsModel（看物理历史、预测下一状态）和 MLP Actor。
但第08阶段的 Transformer / MoE 还没有正式接收动力学模型的结果。
本文件把它们连接起来，并提供 Dense 和 MoE 两个可对照的策略版本。

【输入从哪里来】
1. physics：第15阶段 K1PhysicsBridge.read() 的真实物理观测。
2. dynamics：第14阶段 DynamicsModel.step() 或 forward_sequence() 的输出。
   这里会使用当前物理特征 context，以及由历史形成的动力学变量 latent。
3. goal：ControlGoal，可以是完整参考动作、接触目标、物体目标，或组合。

【输出到哪里去】
单步：输出每个关节动作的均值、标准差、Critic 价值，以及下一步要复用的 KV 缓存。
整段：输出 [B,T,N] 的动作均值/标准差和 [B,T] 的价值，供 algorithms.sequence_training 的时序 PPO 使用。

【重要边界】
- 这里只连接策略网络，不执行 mjlab、PPO 更新或动力学监督优化。
- 不能直接把本文件的 Transformer 策略塞进第15阶段的 MLP PPOUpdater：
  它的旧代码是逐帧重新评估 MLP 的，没有重建 Transformer 的历史。
- 动力学 context/latent 没有独立直连 Router。路由读取“目标token摘要”，但
  接触/物体token包含当前状态误差，完整动作目标也在当前机身系表达；因此
  不能称为与反馈完全无关的纯目标路由。这是代码的真实语义，而不是论文结论。
- 物理模型由另外的 DynamicsUpdater 独立优化，本 Actor 默认对其输入 detach。
- 真正零样本滑板需要训练任务/测试任务分离和高层接触规划，本文件不声称实现。
- 生产模块不内嵌自测；统一回归位于 tests/，运行 python tools/evaluate.py。

形状约定：
B = 同一款机器人的并行环境数，T = 时间长度，N = 机器人关节数。
D = Transformer 隐藏宽度，H = 参考动作时间点数，C = 接触槽位数，M = 物体数。
单步物理观测的前导维为 [B]，整段为 [B,T]。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.distributions import Independent, Normal

from ..robot_contract import RobotContract
from .attention import AttentionCache, CausalSelfAttention
from .encoder import EncodedMotion, MultiRobotEncoder
from .moe import SwiGLUExpert
from .reference_fusion import ReferenceFusion
from .transformer import (
    ReferenceConditionedTransformer,
    TransformerCache,
    TransformerConfig,
    TransformerOutput,
)
from .distributions import GaussianActionHead
from .recurrent import RecurrentReferenceBackbone, RecurrentCache, weight_stamp
from .lru import validate_lru_settings
from .physics import (
    CONTACT_GOAL_DIM,
    OBJECT_GOAL_DIM,
    ControlGoal,
    DynamicsEncoding,
    PhysicsMoEConditioner,
    PhysicsObservation,
    PhysicsPolicyOutput,
    _check_observation,
)


@dataclass(frozen=True)
class PhysicsPolicyConfig:
    """功能：集中设置 Dense / MoE 策略的结构。

    输入：创建配置时传入所需参数。
    输出：不可修改的配置对象；用于构造网络，也保存在 checkpoint 中。

    backbone：transformer/dense为无MoE的Transformer；gru/lstm/lru是替代时间主干，不叠加时间注意力。
    recurrent_hidden_dim是Actor记忆，不改变独立DynamicsModel的GRU。
    max_replay_steps是保护上限；循环PPO仍要求保留到真实回合起点。
    physics_width/latent_width 必须和独立 DynamicsModel 的输出维度相同。
    """

    backbone: str = "moe"
    d_model: int = 256
    input_hidden_dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    context_len: int = 32
    dense_hidden_dim: int = 512
    num_fine_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 256
    router_hidden_dim: int = 256
    physics_width: int = 128
    latent_width: int = 64
    critic_hidden_dim: int = 256
    init_std: float = 1.0
    check_values: bool = True
    # 循环Actor专用：None表示与d_model相同。不是DynamicsModel的记忆宽度。
    recurrent_hidden_dim: int | None = None
    # 完整回合历史重放的内存保护上限；超过时明确报错，不偷截断RNN历史。
    max_replay_steps: int = 4096
    # LRU专用：state_dim个复数记忆占2*state_dim个实数；不改变DynamicsModel。
    lru_state_dim: int | None = None
    lru_r_min: float = 0.9
    lru_r_max: float = 0.999
    lru_max_phase: float = 2 * math.pi
    lru_scan: str = "parallel"

    def __post_init__(self) -> None:
        """功能：创建配置时检查网络宽度和专家参数，避免计算到一半才报错。

        输入：上面每个配置字段。
        输出：无；不合法立即报错。
        """
        # transformer就是已有dense的易懂别名；不是两套可以各自改坏的实现。
        if self.backbone == "transformer":
            object.__setattr__(self, "backbone", "dense")
        if self.backbone not in ("dense", "moe", "gru", "lstm", "lru"):
            raise ValueError("backbone 只能选择 transformer/dense、moe、gru、lstm 或 lru")
        validate_lru_settings(self.lru_state_dim, self.lru_r_min, self.lru_r_max,
                              self.lru_max_phase, self.lru_scan)
        if self.backbone != "lru" and (self.lru_state_dim is not None or
                (self.lru_r_min, self.lru_r_max, self.lru_max_phase, self.lru_scan)
                != (.9, .999, 2*math.pi, "parallel")):
            raise ValueError("lru_*配置只用于LRU，不能配置后悄悄忽略")
        if self.recurrent_hidden_dim is not None and (
                type(self.recurrent_hidden_dim) is not int or self.recurrent_hidden_dim < 1):
            raise ValueError("recurrent_hidden_dim必须是正整数或None")
        if type(self.max_replay_steps) is not int or self.max_replay_steps < 1:
            raise ValueError("max_replay_steps必须是正整数")
        if self.backbone not in ("gru", "lstm") and self.recurrent_hidden_dim is not None:
            raise ValueError("只有GRU/LSTM使用recurrent_hidden_dim")
        widths = (
            "d_model", "input_hidden_dim", "n_layers", "n_heads", "n_kv_heads",
            "context_len", "dense_hidden_dim", "num_fine_experts", "top_k",
            "expert_hidden_dim", "router_hidden_dim", "physics_width", "latent_width",
            "critic_hidden_dim",
        )
        for name in widths:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if type(self.num_shared_experts) is not int or self.num_shared_experts < 0:
            raise ValueError("num_shared_experts 不能为负数")
        if self.top_k > self.num_fine_experts:
            raise ValueError("top_k 不能超过专家总数量")
        if (self.d_model % self.n_heads != 0 or
                self.n_heads % self.n_kv_heads != 0 or
                (self.d_model // self.n_heads) % 2 != 0):
            raise ValueError("注意力头数不匹配：D 必须可整除 Q 头；Q 可整除 KV 头；head_dim 为偶数")
        if isinstance(self.init_std, bool) or not isinstance(self.init_std, (float, int)):
            raise TypeError("init_std 必须是正数")
        if not math.isfinite(float(self.init_std)) or self.init_std <= 0:
            raise ValueError("init_std 必须是有限的正数")
        if type(self.check_values) is not bool:
            raise TypeError("check_values 必须是 bool")


@dataclass
class PhysicsSequenceOutput:
    """功能：保存整段策略输出，供时序 PPO 计算旧动作概率。

    mean/std：[B,T,N]，原始动作高斯分布参数，不是 PD 力矩目标。
    values：[B,T]，Critic 的价值预测。
    valid_mask：[B,T]，True 表示该时间点是真实采样。
    balance_loss：标量；只在 MoE 模式有意义，不能在本层自动加到 PPO。
    """

    robot_id: str
    joint_names: tuple[str, ...]
    mean: torch.Tensor
    std: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor
    balance_loss: torch.Tensor
    router_entropy: torch.Tensor
    expert_load: torch.Tensor

    def distribution(self) -> Independent:
        """功能：构造每个时间点的动作概率分布。

        输入：对象里的 mean/std，形状 [B,T,N]。
        输出：高斯联合分布；对 [B,T,N] 动作求 log_prob 得 [B,T]。
        """
        return Independent(Normal(self.mean, self.std), 1)

    def evaluate_actions(self, raw_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """功能：重算 PPO 中旧的原始动作的概率和分布熵。

        输入：raw_actions [B,T,N]，必须是采样时的原始动作。
        输出：log_prob [B,T] 和 entropy [B,T]。
        注意：不要传被执行器裁剪后的目标角度。
        """
        if not isinstance(raw_actions, torch.Tensor) or raw_actions.shape != self.mean.shape:
            raise ValueError("raw_actions 必须和 mean 形状相同")
        if raw_actions.device != self.mean.device or raw_actions.dtype != self.mean.dtype:
            raise ValueError("raw_actions 的设备或精度不匹配")
        dist = self.distribution()
        log_prob = dist.log_prob(raw_actions)
        entropy = dist.entropy()
        # 补位不会被 PPO 当成有效经验；真正 PPO loss 还需要使用 valid_mask。
        log_prob = torch.where(self.valid_mask, log_prob, 0.0)
        entropy = torch.where(self.valid_mask, entropy, 0.0)
        return log_prob, entropy


# --------------------------- Dense Transformer 基线 ---------------------------


class _DenseLayer(nn.Module):
    """功能：一层普通 Transformer：历史注意力 + 一层所有时刻都运行的 FFN。

    输入：状态 [B,T,D]，有效时间掩码、回合起点标记。
    输出：同形状 [B,T,D]，每帧可以利用本回合有限长度的过去信息。
    说明：FFN 使用旧 moe.py 的 SwiGLUExpert 计算结构，但此处没有 Router，
          它只是一个始终启用的普通前馈层。
    """

    def __init__(self, cfg: PhysicsPolicyConfig) -> None:
        """功能：创建本层真正可训练的神经网络。

        输入：cfg，指定隐藏特征宽度、历史帧数、注意力头数等。
        输出：无；新建的注意力和 FFN 会作为本层的可训练参数保存。
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
            context_len=cfg.context_len, check_values=cfg.check_values,
        )
        self.ff_norm = nn.LayerNorm(cfg.d_model)
        self.ff = SwiGLUExpert(cfg.d_model, cfg.dense_hidden_dim)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                starts: torch.Tensor) -> torch.Tensor:
        """功能：把本回合的历史注意力结果与普通 FFN 依次加到原特征上。

        输入：
            x：[B,T,D]，一个批次中按时间从早到晚排列的身体特征。
            valid：[B,T]，True 表示是真实采样帧，False 表示末尾补位。
            starts：[B,T]，True 表示该帧已进入新回合，禁止看之前回合的历史。
        输出：更新后的 [B,T,D] 特征，仍然不是电机动作。
        """
        attention_delta = self.attn(self.attn_norm(x), valid_mask=valid, episode_start=starts)
        x = torch.where(valid.unsqueeze(-1), x + attention_delta, 0.0)
        ff_delta = self.ff(self.ff_norm(x))
        return torch.where(valid.unsqueeze(-1), x + ff_delta, 0.0)

    @torch.no_grad()
    def step(self, x: torch.Tensor, cache: AttentionCache | None,
             starts: torch.Tensor) -> tuple[torch.Tensor, AttentionCache]:
        """输入：当前特征 [B,D]、上一 KV Cache、新回合标记 [B]。

        输出：本时刻的新特征 [B,D] 和更新后的 KV Cache。
        """
        delta, new_cache = self.attn.step(self.attn_norm(x), cache, episode_start=starts)
        x = x + delta
        return x + self.ff(self.ff_norm(x)), new_cache


class DenseReferenceTransformer(nn.Module):
    """功能：保持与旧 MoE Transformer 相同的输入输出，但换成普通 FFN。

    输入：EncodedMotion（状态、参考 token、参考掩码）。
    输出：TransformerOutput，含处理后的隐藏特征及零 MoE 诊断。
    用途：只对比 MoE 是否真的有价值，而不是改变整套任务输入。
    """

    def __init__(self, cfg: PhysicsPolicyConfig) -> None:
        """输入：网络配置。输出：一个 Dense Transformer 主干。"""
        super().__init__()
        self.cfg = cfg
        self.reference_fusion = ReferenceFusion(
            cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
            check_values=cfg.check_values,
        )
        self.layers = nn.ModuleList([_DenseLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self._cache_owner = object()

    def invalidate_caches(self) -> None:
        """功能：让权重更新前生成的 KV Cache 全部失效。

        输入：无。输出：无；旧缓存随后传入 step() 将被拒绝。
        """
        self._cache_owner = object()
        for layer in self.layers:
            layer.attn.invalidate_caches()

    def train(self, mode: bool = True):
        """功能：切换训练/推理模式，并让旧缓存失效。

        输入：mode=True 表示训练；False 表示推理。
        输出：当前网络本身。
        """
        previous = self.training
        super().train(mode)
        if previous != mode:
            self.invalidate_caches()
        return self

    def forward(self, encoded: EncodedMotion, *, valid_mask: torch.Tensor,
                episode_start: torch.Tensor) -> TransformerOutput:
        """功能：整段读参考目标与历史信息，允许反向传播。

        输入：EncodedMotion 状态 [B,T,D]、目标 [B,T,H,D]；
              valid_mask / episode_start 都是 [B,T]。
        输出：TransformerOutput.features [B,T,D]。
        """
        fused = self.reference_fusion(
            encoded.state_token, encoded.reference_tokens,
            encoded.reference_mask, time_mask=valid_mask,
        )
        x = fused.policy_features
        for layer in self.layers:
            x = layer(x, valid_mask, episode_start)
        x = torch.where(valid_mask.unsqueeze(-1), self.final_norm(x), 0.0)
        z = x.new_zeros(())
        return TransformerOutput(
            x, fused.reference_summary, valid_mask.clone(),
            z, x.new_zeros((0,)), x.new_zeros((0, 0)), x.new_zeros((0, 0)),
        )

    @torch.no_grad()
    def step(self, encoded: EncodedMotion, *, env_ids: tuple[int, ...],
             cache: TransformerCache | None,
             episode_start: torch.Tensor) -> tuple[TransformerOutput, TransformerCache]:
        """功能：逐帧读取旧 KV 和当前目标，供 mjlab 实时采样。

        输入：状态 [B,D]、目标 [B,H,D]、固定 env_ids、上一缓存、reset 标记 [B]。
        输出：TransformerOutput.features [B,D] 和新的缓存。
        """
        if self.training:
            raise RuntimeError("逐帧 step 前先 model.eval()")
        b = encoded.state_token.shape[0]
        if len(env_ids) != b or len(set(env_ids)) != b:
            raise ValueError("env_ids 数量错误或含重复环境编号")
        if cache is not None:
            if (not isinstance(cache, TransformerCache)
                    or cache._owner is not self._cache_owner
                    or cache.robot_id != encoded.robot_id
                    or cache.env_ids != env_ids
                    or len(cache.layer_caches) != len(self.layers)):
                raise ValueError("缓存已经过期，或属于其他机器人/环境排列")
        fused = self.reference_fusion(
            encoded.state_token, encoded.reference_tokens, encoded.reference_mask,
        )
        x = fused.policy_features
        caches: list[AttentionCache] = []
        for i, layer in enumerate(self.layers):
            y, c = layer.step(x, cache.layer_caches[i] if cache is not None else None,
                              episode_start)
            x = y
            caches.append(c)
        x = self.final_norm(x)
        return (TransformerOutput(x, fused.reference_summary,
                                  torch.ones(b, dtype=torch.bool, device=x.device),
                                  x.new_zeros(()), x.new_zeros((0,)),
                                  x.new_zeros((0, 0)), x.new_zeros((0, 0))),
                TransformerCache(encoded.robot_id, env_ids, caches, self._cache_owner))


# ------------------------ 多模态目标 tokens + 物理条件化 Actor ------------------------


class PhysicsConditionedActorCritic(nn.Module):
    """功能：同一个策略类切换 Dense/MoE，共用输入接口和独立动力学模型。

    输入：
        robots：如 {"k1_22dof": RobotContract(...), "g1_29dof": ...}。
        cfg：PhysicsPolicyConfig，决定 Dense 还是 MoE。
    输出：一个可训练的网络；按 robot_id 分组读取对应机器人的数据。

    模型结构：
        各机器人专用状态/运动 Encoder + 共享接触/物体目标编码器
        → 已有 PhysicsMoEConditioner（物理隐变量进入 Actor 隐藏特征）
        → 共享 Dense 或 MoE Transformer
        → 各机器人专用动作头 + 完全独立的 Critic。

    网络主干权重可以共用，但 K1/G1 的动力学隐变量语义未必天生一致。
    跨机器人联合训练是否产生迁移，需要实验，不能因为特征宽度相同就认定成功。
    """

    SCHEMA_VERSION = 1

    def __init__(self, robots: Mapping[str, RobotContract],
                 cfg: PhysicsPolicyConfig | None = None,
                 privileged_dims: Mapping[str, int] | None = None) -> None:
        """功能：创建网络、动作头和 Critic，所有参数在初始化时注册。

        输入：
            robots：机器人 ID → 关节合同。
            cfg：架构配置，缺省使用 MoE。
            privileged_dims：每种机器人 Critic 可用的额外仿真状态维数。
        输出：无；创建好的对象可放进 torch.optim.Adam。
        """
        super().__init__()
        if not isinstance(robots, Mapping) or not robots:
            raise ValueError("robots 必须是非空映射")
        if not all(isinstance(k, str) and k.strip() and isinstance(v, RobotContract)
                   for k, v in robots.items()):
            raise TypeError("机器人 ID 必须非空，配置必须是 RobotContract")
        self.cfg = cfg or PhysicsPolicyConfig()
        if not isinstance(self.cfg, PhysicsPolicyConfig):
            raise TypeError("cfg 必须是 PhysicsPolicyConfig")
        self.robots = {k: robots[k] for k in sorted(robots)}
        self.robot_ids = tuple(self.robots)
        self._keys = {k: f"robot_{i}" for i, k in enumerate(self.robot_ids)}
        dims = {} if privileged_dims is None else dict(privileged_dims)
        if set(dims) - set(self.robots):
            raise ValueError("privileged_dims 包含未注册的机器人")
        if any(type(n) is not int or n < 0 for n in dims.values()):
            raise ValueError("Critic 特权状态宽度必须是非负整数")
        self.privileged_dims = {rid: dims.get(rid, 0) for rid in self.robot_ids}
        # 复用第04个文件的两套 MLP 输入 Encoder，避免再写一份状态/动作编码。
        self.encoder = MultiRobotEncoder(
            self.robots, d_model=self.cfg.d_model,
            hidden_dim=self.cfg.input_hidden_dim,
            check_values=self.cfg.check_values,
        )
        # 接触/物体目标具有统一的物理语义，使用各机器人共用的投影。
        self.contact_goal_proj = nn.Sequential(
            nn.Linear(CONTACT_GOAL_DIM + 12 + CONTACT_GOAL_DIM, self.cfg.input_hidden_dim),
            nn.SiLU(), nn.Linear(self.cfg.input_hidden_dim, self.cfg.d_model),
            nn.LayerNorm(self.cfg.d_model),
        )
        self.object_goal_proj = nn.Sequential(
            nn.Linear(6 + 16 + 6, self.cfg.input_hidden_dim),
            nn.SiLU(), nn.Linear(self.cfg.input_hidden_dim, self.cfg.d_model),
            nn.LayerNorm(self.cfg.d_model),
        )
        # 区分“来自 MoCap/接触/物体”的 token，不强迫 Router 自己猜是哪种输入。
        self.modality_embeddings = nn.Parameter(torch.zeros(3, self.cfg.d_model))
        nn.init.normal_(self.modality_embeddings, std=0.01)
        self.conditioner = PhysicsMoEConditioner(
            self.cfg.d_model, self.cfg.physics_width, self.cfg.latent_width,
            hidden_dim=self.cfg.input_hidden_dim,
        )
        if self.cfg.backbone == "moe":
            transformer_cfg = TransformerConfig(
                d_model=self.cfg.d_model, n_layers=self.cfg.n_layers,
                n_heads=self.cfg.n_heads, n_kv_heads=self.cfg.n_kv_heads,
                context_len=self.cfg.context_len,
                num_fine_experts=self.cfg.num_fine_experts,
                num_shared_experts=self.cfg.num_shared_experts,
                top_k=self.cfg.top_k,
                expert_hidden_dim=self.cfg.expert_hidden_dim,
                router_hidden_dim=self.cfg.router_hidden_dim,
                check_values=self.cfg.check_values,
            )
            self.backbone = ReferenceConditionedTransformer(transformer_cfg)
        elif self.cfg.backbone == "dense":
            self.backbone = DenseReferenceTransformer(self.cfg)
        else:
            self.backbone = RecurrentReferenceBackbone(self.cfg)
        self.action_heads = nn.ModuleDict({
            self._keys[rid]: GaussianActionHead(
                self.cfg.d_model, contract.dof,
                self.cfg.init_std, check_values=self.cfg.check_values,
            ) for rid, contract in self.robots.items()
        })
        # Critic 完全独立。它的梯度不会改 Actor 的主干或者 DynamicsModel。
        self.critics = nn.ModuleDict()
        for rid, contract in self.robots.items():
            dim = (6 + 3 * contract.dof + self.cfg.physics_width
                   + self.cfg.latent_width + self.cfg.d_model
                   + self.privileged_dims[rid])
            self.critics[self._keys[rid]] = nn.Sequential(
                nn.Linear(dim, self.cfg.critic_hidden_dim), nn.SiLU(),
                nn.Linear(self.cfg.critic_hidden_dim, self.cfg.critic_hidden_dim),
                nn.SiLU(), nn.Linear(self.cfg.critic_hidden_dim, 1),
            )

    def get_extra_state(self) -> dict:
        """功能：保存网络接口信息，避免错把不同关节顺序的权重互相加载。

        输入：无。输出：包含机器人和配置的版本字典，跟随 state_dict 存储。
        """
        config = asdict(self.cfg)
        if self.cfg.backbone not in ("gru", "lstm", "lru"):
            config.pop("recurrent_hidden_dim"); config.pop("max_replay_steps")
        if self.cfg.backbone != "lru":
            # 保留R20五种旧策略的state_dict元数据，未启用LRU不改变旧权重结构。
            for key in ("lru_state_dim", "lru_r_min", "lru_r_max", "lru_max_phase", "lru_scan"):
                config.pop(key)
        return dict(schema=self.SCHEMA_VERSION, config=config,
                    robots={k: (v.root_name, v.joint_names, v.root_qpos_dim)
                            for k, v in self.robots.items()},
                    privileged_dims=self.privileged_dims,
                    goal_tokens="motion_contact_object_separate_v1")

    def set_extra_state(self, state: dict) -> None:
        """功能：加载 checkpoint 时核对网络结构与机器人合同。

        输入：state：此前保存的接口版本字典。输出：无；不一致直接报错。
        """
        if state != self.get_extra_state():
            raise ValueError("物理控制策略 checkpoint 的结构/机器人/目标接口不匹配")
        self.invalidate_caches()

    @property
    def history_requirement(self) -> int | None:
        """输出需要保留的Actor前文长度。

        Transformer返回有限帧数；GRU/LSTM/LRU返回None表示必须从真实回合起点重放。
        这不是说RNN推理要重算全历史：逐帧执行只更新h/c，PPO改权重后才重放。
        """
        if self.cfg.backbone in ("gru", "lstm", "lru"):
            return None
        return self.cfg.n_layers * (self.cfg.context_len - 1)

    def invalidate_caches(self) -> None:
        """功能：声明网络权重已经改变，旧时序注意力缓存不再可信。

        输入：无。输出：无；优化器更新后由训练器显式调用。
        """
        self.backbone.invalidate_caches()

    def train(self, mode: bool = True):
        """功能：切换训练/推理模式。

        输入：True 开启训练，False 开启推理。输出：模型自身。
        说明：底层 Transformer 会在模式改变时清除旧缓存。
        """
        return super().train(mode)

    @staticmethod
    def _valid_flag(x: torch.Tensor | None, shape: tuple[int, ...],
                    device: torch.device, *, default: bool) -> torch.Tensor:
        """功能：检查时间 mask。输入：mask、应有形状、设备；输出：bool Tensor。"""
        if x is None:
            return torch.full(shape, default, dtype=torch.bool, device=device)
        if not isinstance(x, torch.Tensor) or x.shape != shape or x.dtype != torch.bool or x.device != device:
            raise ValueError(f"时间 mask 必须是形状 {shape} 的同设备 bool 张量")
        return x

    def _check_field(self, name: str, value: torch.Tensor, shape: tuple[int, ...],
                     device: torch.device, dtype: torch.dtype,
                     *, boolean: bool = False) -> None:
        """功能：检查目标是否来自同一批环境、同一设备和相同时间点。

        输入：字段名、实际张量、期望形状/设备/精度。
        输出：无；检查失败抛异常。避免广播把不同机器人的目标混起来。
        """
        expected = torch.bool if boolean else dtype
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"{name} 必须为形状 {shape} 的 Tensor")
        if value.device != device or value.dtype != expected:
            raise ValueError(f"{name} 的设备或数据类型不一致")
        if not boolean and self.cfg.check_values and not torch.isfinite(value).all().item():
            raise ValueError(f"{name} 包含 NaN/Inf")

    def _build_encoded(self, robot_id: str, physics: PhysicsObservation,
                       dynamics: DynamicsEncoding, goal: ControlGoal,
                       valid: torch.Tensor) -> EncodedMotion:
        """功能：整理所有任务目标并给当前状态加入动力学特征，送给 Transformer。

        输入：
            robot_id：当前正在处理的机器人类型。
            physics：当前身体、接触点、物体的真实观测；[B,...] 或 [B,T,...]。
            dynamics：同一时刻的物理 context [..,P] 和隐变量 z [..,Z]。
            goal：当前和未来的动作参考，或目标接触/目标物体信息。
            valid：[B] 或 [B,T]，是否为真实采样时间点。
        输出：EncodedMotion：state_token [...,D]，reference_tokens [...,H总,D]，
              reference_mask [...,H总]。H总 是全部有效任务 token 的合并个数。

        关键：接触/物体每个槽位都单独保留一个 token，不简单平均掉位置细节。
        只有任务目标进入 Router 摘要；动力学条件通过 state_token 进入 Expert。
        """
        if robot_id not in self.robots:
            raise KeyError(f"未知机器人 {robot_id!r}")
        robot = self.robots[robot_id]
        p = self.action_heads[self._keys[robot_id]].mean_head.weight
        lead = _check_observation(physics, robot, p.device, p.dtype, self.cfg.check_values)
        if valid.shape != lead:
            raise ValueError("valid_mask 的 B/T 维度与物理状态不一致")
        if not isinstance(goal, ControlGoal):
            raise TypeError("goal 必须是 ControlGoal")
        if not isinstance(dynamics, DynamicsEncoding):
            raise TypeError("dynamics 必须来自 DynamicsModel 的 DynamicsEncoding")
        for name, value, width in (("context", dynamics.context, self.cfg.physics_width),
                                   ("latent", dynamics.latent, self.cfg.latent_width),
                                   ("prediction_error", dynamics.prediction_error,
                                    9 + 2 * robot.dof)):
            self._check_field("dynamics." + name, value, (*lead, width), p.device, p.dtype)
        if not isinstance(dynamics.hidden, torch.Tensor):
            raise TypeError("dynamics.hidden 必须是独立动力学模型的历史记忆张量")

        # 身体状态 encoder 因为是独立分支，可以在无参考 motion 时照样计算。
        # 这里复用第04个文件创建的子网络；不用伪造一个“假 MoCap”骗过它的 forward。
        key = self._keys[robot_id]
        state_token = self.encoder.state_encoders[key](
            torch.where(valid.unsqueeze(-1), physics.body_obs, 0.0)
        )
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        if goal.motion is not None:
            motion = goal.motion
            if not hasattr(motion, "features") or not hasattr(motion, "valid_mask"):
                raise TypeError("goal.motion 必须是 ReferenceObservation")
            features, mask = motion.features, motion.valid_mask
            h = features.shape[-2] if isinstance(features, torch.Tensor) and features.ndim == len(lead) + 2 else -1
            if h <= 0:
                raise ValueError("motion.features 应为 [...,H,16+2N]，H 不能为0")
            self._check_field("motion.features", features, (*lead, h, 16 + 2 * robot.dof), p.device, p.dtype)
            self._check_field("motion.valid_mask", mask, (*lead, h), p.device, p.dtype, boolean=True)
            mask = mask & valid.unsqueeze(-1)
            item = self.encoder.reference_encoders[key](
                torch.where(mask.unsqueeze(-1), features, 0.0)
            ) + self.modality_embeddings[0]
            tokens.append(torch.where(mask.unsqueeze(-1), item, 0.0))
            masks.append(mask)

        if goal.contact_features is not None:
            c = physics.contact_features.shape[-2]
            self._check_field("contact_goal", goal.contact_features,
                              (*lead, c, CONTACT_GOAL_DIM), p.device, p.dtype)
            self._check_field("contact_goal_mask", goal.contact_mask,
                              (*lead, c), p.device, p.dtype, boolean=True)
            if (goal.contact_mask & ~physics.contact_mask).any().item():
                raise ValueError("接触目标指向了不存在的接触槽位")
            mask = goal.contact_mask & physics.contact_mask & valid.unsqueeze(-1)
            wanted = goal.contact_features
            real = physics.contact_features
            # 第7个接触目标为 -1 时，代表不指定接触状态，不应计算接触差异。
            contact_diff = torch.cat((wanted[..., :3] - real[..., :3],
                                      wanted[..., 3:6] - real[..., 3:6],
                                      torch.where(wanted[..., 6:7] < 0, 0.0,
                                                  wanted[..., 6:7] - real[..., 10:11])), dim=-1)
            features = torch.cat((wanted, real, contact_diff), dim=-1)
            item = self.contact_goal_proj(torch.where(mask.unsqueeze(-1), features, 0.0))
            item = item + self.modality_embeddings[1]
            tokens.append(torch.where(mask.unsqueeze(-1), item, 0.0))
            masks.append(mask)
        elif goal.contact_mask is not None:
            raise ValueError("给了 contact_mask，但没有 contact_features")

        if goal.object_features is not None:
            m = physics.object_features.shape[-2]
            self._check_field("object_goal", goal.object_features,
                              (*lead, m, OBJECT_GOAL_DIM), p.device, p.dtype)
            self._check_field("object_goal_mask", goal.object_mask,
                              (*lead, m), p.device, p.dtype, boolean=True)
            if (goal.object_mask & ~physics.object_mask).any().item():
                raise ValueError("物体目标指向了不存在的对象槽位")
            mask = goal.object_mask & physics.object_mask & valid.unsqueeze(-1)
            wanted = goal.object_features
            real = physics.object_features
            object_diff = torch.cat((wanted[..., :3] - real[..., :3],
                                     wanted[..., 3:6] - real[..., 7:10]), dim=-1)
            item = self.object_goal_proj(torch.where(mask.unsqueeze(-1),
                                                       torch.cat((wanted, real, object_diff), dim=-1), 0.0))
            item = item + self.modality_embeddings[2]
            tokens.append(torch.where(mask.unsqueeze(-1), item, 0.0))
            masks.append(mask)
        elif goal.object_mask is not None:
            raise ValueError("给了 object_mask，但没有 object_features")

        if not tokens:
            raise ValueError("至少提供一种真实目标：动作参考、接触目标或物体目标")
        combined_mask = torch.cat(masks, dim=-1)
        if (valid & ~combined_mask.any(dim=-1)).any().item():
            raise ValueError("每个真实控制时刻至少要有一个有效任务目标")
        joined = torch.cat(tokens, dim=-2)
        # 多种目标各有自己的补位；拼接后可能是 [真,假,真]。
        # 稳定地把真实目标移到前面，并同步重排 token。不能只改 mask。
        # 每个 token 自带时间/部位信息，本层不使用 token 槽位位置编码。
        order = torch.argsort((~combined_mask).to(torch.int64), dim=-1, stable=True)
        joined = joined.gather(-2, order.unsqueeze(-1).expand_as(joined))
        combined_mask = combined_mask.gather(-1, order)
        encoded = EncodedMotion(robot_id, state_token, joined, combined_mask)
        # 复用第14阶段的物理特征注入层，DynamicsModel 的权重并不属于本策略。
        detached = DynamicsEncoding(dynamics.latent.detach(), dynamics.context.detach(),
                                    dynamics.hidden.detach(), dynamics.prediction_error.detach())
        return self.conditioner(encoded, detached)

    def _value(self, robot_id: str, physics: PhysicsObservation,
               dynamics: DynamicsEncoding, encoded: EncodedMotion,
               privileged: torch.Tensor | None, valid: torch.Tensor) -> torch.Tensor:
        """功能：用独立 Critic 评价当前状态的未来收益，避免特权信息泄漏给 Actor。

        输入：机器人ID、物理输入、动力学估计、任务 tokens、可选额外仿真真值。
        输出：[B] 或 [B,T] 价值预测；与每个当前时间点对应。
        """
        lead = physics.body_obs.shape[:-1]
        p = self.action_heads[self._keys[robot_id]].mean_head.weight
        pdim = self.privileged_dims[robot_id]
        if pdim > 0:
            self._check_field("privileged", privileged, (*lead, pdim), p.device, p.dtype)
        elif privileged is not None:
            raise ValueError("privileged_dim=0，不允许传入特权物理数据")
        # 目标摘要仅来自真实任务 token；不会让未来真实状态进入 Critic。
        mask = encoded.reference_mask
        summary = (encoded.reference_tokens * mask.unsqueeze(-1)).sum(dim=-2) / (
            mask.sum(dim=-1, keepdim=True).clamp_min(1)
        )
        parts = (physics.body_obs.detach(), dynamics.context.detach(),
                 dynamics.latent.detach(), summary.detach())
        extras = (privileged,) if pdim > 0 else ()
        critic_input = torch.cat((*parts, *extras), dim=-1)
        value = self.critics[self._keys[robot_id]](critic_input).squeeze(-1)
        return torch.where(valid, value, 0.0)

    def forward_sequence(self, robot_id: str, physics: PhysicsObservation,
                         dynamics: DynamicsEncoding, goal: ControlGoal, *,
                         valid_mask: torch.Tensor | None = None,
                         episode_start: torch.Tensor,
                         privileged: torch.Tensor | None = None,
                         require_fresh_start: bool = True) -> PhysicsSequenceOutput:
        """功能：整段计算动作分布和价值，供时序 PPO 重新评估历史采样动作。

        输入：
            robot_id：当前是哪种机器人，例如 'k1_22dof'。
            physics：物理观测 [B,T,...]，必须按采样发生的时间顺序排列。
            dynamics：采样期间形成的同一时刻的 context/latent [B,T,P/Z]。
            goal：每一时刻的目标，可以是 motion 或 contact/object Goal。
            valid_mask：[B,T]，True 是真实控制步；后面的补位 False。
            episode_start：[B,T]，True 表示该帧已经是新回合。
            privileged：可选 [B,T,Q]，只用于 Critic。
            require_fresh_start：默认要求第0帧是各环境新回合起点。
        输出：动作 mean/std [B,T,N]、values [B,T]、Router 诊断。

        为什么要求新回合起点：
            Transformer 有历史 KV。若从回合中间截取一段，却丢掉前面历史，
            重算的 log-prob 就不等于采样时的 log-prob，PPO 将出现隐蔽错误。
            回合中段采样由 algorithms.sequence_training 提供前缀并核对历史，不能直接关闭保护绕过它。
        """
        if robot_id not in self.robots:
            raise KeyError(f"未知机器人 {robot_id!r}")
        if not isinstance(physics, PhysicsObservation) or physics.body_obs.ndim != 3:
            raise ValueError("整段前向必须提供 PhysicsObservation，身体观测为 [B,T,...]")
        lead = physics.body_obs.shape[:-1]
        b, t = lead
        valid = self._valid_flag(valid_mask, (b, t), physics.body_obs.device, default=True)
        starts = self._valid_flag(episode_start, (b, t), physics.body_obs.device, default=False)
        if (valid[:, 1:] & ~valid[:, :-1]).any().item():
            raise ValueError("时间补位只能在末尾")
        if (starts & ~valid).any().item():
            raise ValueError("补位时间点不能标记为新回合")
        if require_fresh_start and not starts[:, 0].all().item():
            raise ValueError("序列从回合中间开始，缺少 KV 历史；不能直接重算 PPO")
        encoded = self._build_encoded(robot_id, physics, dynamics, goal, valid)
        features = self.backbone(encoded, valid_mask=valid, episode_start=starts)
        mean, std = self.action_heads[self._keys[robot_id]](features.features)
        mean = torch.where(valid.unsqueeze(-1), mean, 0.0)
        std = torch.where(valid.unsqueeze(-1), std, 1.0)
        values = self._value(robot_id, physics, dynamics, encoded, privileged, valid)
        return PhysicsSequenceOutput(
            robot_id, self.robots[robot_id].joint_names, mean, std, values,
            valid.clone(), features.balance_loss,
            features.router_entropy, features.expert_load,
        )

    def value(self, robot_id: str, physics: PhysicsObservation,
              dynamics: DynamicsEncoding, goal: ControlGoal, *,
              privileged: torch.Tensor | None = None) -> torch.Tensor:
        """只读取独立 Critic，不运行时序主干、不消耗动作采样随机数、不推进 KV。

        输入：单帧物理观测 [B,...]、该帧动力学特征、当前目标和可选特权状态。
        输出：[B] 价值；专门用于 reset 之前的最终状态 bootstrap。
        不能用 policy.step() 代替本函数，否则会把“偷看一帧”的 KV 带到下一动作。
        """
        if robot_id not in self.robots or physics.body_obs.ndim != 2:
            raise ValueError("value 需要已注册机器人和单帧物理观测")
        valid = torch.ones(physics.body_obs.shape[0], device=physics.body_obs.device, dtype=torch.bool)
        encoded = self._build_encoded(robot_id, physics, dynamics, goal, valid)
        return self._value(robot_id, physics, dynamics, encoded, privileged, valid)

    @torch.no_grad()
    def step(self, robot_id: str, physics: PhysicsObservation,
             dynamics: DynamicsEncoding, goal: ControlGoal, *,
             env_ids: Sequence[int], episode_start: torch.Tensor,
             cache: TransformerCache | RecurrentCache | None = None,
             privileged: torch.Tensor | None = None,
             deterministic: bool = False,
             ) -> tuple[PhysicsPolicyOutput, torch.Tensor, torch.Tensor, TransformerCache | RecurrentCache]:
        """功能：逐帧生成一个控制动作，供 mjlab 真正执行。

        输入：
            robot_id：当前机器人 ID。
            physics：当前一帧观测 [B,...]。
            dynamics：独立 DynamicsModel.step() 当前时刻返回的 z/context。
            goal：当前已知参考/接触/物体目标。
            env_ids：B 个环境固定 ID，行顺序变化必须重新建立缓存。
            episode_start：[B]，True 表示当前状态是 reset 后第一帧。
            cache：上一帧返回的缓存；首次必须是 None 且 episode_start 全 True。
            privileged：如果提供，仅交给 Critic。
            deterministic：True 直接执行均值；False 随机采样（供 PPO）。
        输出：
            policy_output：mean/std/value；其中动作不是执行后的角度。
            action：[B,N]，未裁剪的原始动作。
            log_prob：[B]，采样动作在当前策略下的对数概率。
            cache：更新后的历史 KV Cache，下次步进继续使用。
        """
        if self.training:
            raise RuntimeError("推理和 rollout 采样前必须 model.eval()")
        if robot_id not in self.robots:
            raise KeyError(f"未知机器人 {robot_id!r}")
        if not isinstance(physics, PhysicsObservation) or physics.body_obs.ndim != 2:
            raise ValueError("step 必须是单帧 PhysicsObservation [B,...]")
        b = physics.body_obs.shape[0]
        starts = self._valid_flag(episode_start, (b,), physics.body_obs.device, default=False)
        ids = tuple(env_ids)
        if len(ids) != b or len(set(ids)) != b or any(type(i) is not int for i in ids):
            raise ValueError("env_ids 必须是当前批次 B 个无重复整数")
        if cache is None and not starts.all().item():
            raise ValueError("首次推理没有历史缓存，只能从所有环境的新回合第一帧开始")
        if isinstance(cache, RecurrentCache) and cache.policy_stamp != weight_stamp(self):
            raise ValueError("Actor输入编码或其他参数已改变，必须重建循环记忆")
        encoded = self._build_encoded(robot_id, physics, dynamics, goal,
                                      torch.ones(b, device=physics.body_obs.device, dtype=torch.bool))
        result, next_cache = self.backbone.step(
            encoded, env_ids=ids, cache=cache, episode_start=starts,
        )
        if isinstance(next_cache, RecurrentCache):
            next_cache.policy_stamp = weight_stamp(self)
        mean, std = self.action_heads[self._keys[robot_id]](result.features)
        values = self._value(robot_id, physics, dynamics, encoded, privileged,
                             torch.ones(b, device=physics.body_obs.device, dtype=torch.bool))
        policy = PhysicsPolicyOutput(mean, std, values, self.robots[robot_id].joint_names)
        dist = policy.distribution()
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return policy, action, log_prob, next_cache

# 兼容既有配置名称；新文档统一用 PhysicsPolicyConfig。
PhysicsPolicyV2Config = PhysicsPolicyConfig
