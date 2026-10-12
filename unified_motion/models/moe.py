"""混合专家层：条件路由、Top-K、共享专家、负载统计。

运行：python -m unified_motion.models.moe
依赖：PyTorch、前面统一的 Encoder/ReferenceFusion 接口（本文件本身不直接导入它们）。

设计合同：
- x：ReferenceFusion.policy_features，[B,D] 或 [B,T,D]。
- router_context：ReferenceFusion.reference_summary，必须与 x 的时间/环境轴一致。
- True 的 valid_mask 表示真实时间点，False 表示补位。
- Router 根据参考摘要选择专家；Fine Expert 处理 x，Shared Expert 每帧都参与。
- 返回 delta，不含残差！之后的 Transformer Block 负责执行 x + delta。
- 输出包含负载均衡辅助损失，但不自动把它加进 PPO 的优化目标。
- 不提供控制力矩/关节限位，也不实现 PPO、Actor 或训练环境。
- 每种机器人先由前面的 Encoder 映射到相同宽度 D，再复用本模块权重。
- 默认 Top-K=2；Top-K=1 时主任务损失一般不能通过选中权重训练 Router，
  需要单独研究辅助损失/可微路由，不能误以为参数会自动得到充分训练。
- 本版按选中专家做稀疏计算，优先保证数学/梯度正确；Python 专家循环及
  nonzero 可能在 GPU 产生同步开销，不能把“稀疏”当作实测更快的证明。
- checkpoint 保存结构版本和含义；不同 Expert 配置不能静默混用权重。
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MoEOutput:
    """MoE 输出以及训练诊断；除 delta 之外，其他项不等于机器人动作。"""

    delta: torch.Tensor                # [B,D] 或 [B,T,D]；由后续 Block 加残差
    balance_loss: torch.Tensor         # 标量；只有训练器显式使用它才生效
    router_entropy: torch.Tensor       # 标量；用于观察专家分布，非 PPO entropy
    router_probs: torch.Tensor         # [..., E]；补位的整行概率置 0
    topk_indices: torch.Tensor         # [..., K]；补位设成 -1
    topk_weights: torch.Tensor         # [..., K]；补位权重为 0
    expert_counts: torch.Tensor        # [E]；一个 token 选中 K 个专家，计 K 次
    expert_load: torch.Tensor          # [E]；各专家占全部有效选择的比例


class SwiGLUExpert(nn.Module):
    """一个小型专家：把 D 维特征处理后，仍输出 D 维。"""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """创建一个独立专家的前馈网络。

        输入：d_model输入输出宽度D，hidden_dim中间宽度。输出：无；每个实例的权重独立，不预先规定负责哪个技能。"""
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model 必须是正整数")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim 必须是正整数")
        # 一次线性运算得到两组中间向量，减少额外模块定义。
        self.up_gate = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate 决定哪些特征更重要，value 提供要处理的信息。
        """让一个专家处理收到的特征。

        输入：x [...,D]。输出：同形状特征增量；SwiGLU用一组中间值调节另一组，不选择其他专家也不加残差。"""
        gate, value = self.up_gate(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class ReferenceRoutedMoE(nn.Module):
    """给参考目标打分，选 K 个 Fine Expert，再加上 Shared Expert。"""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        d_model: int = 256,
        num_fine_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        expert_hidden_dim: int = 512,
        router_hidden_dim: int = 256,
        *,
        check_values: bool = True,
    ) -> None:
        """创建Router、可选择专家和始终参与的共享专家。

        输入：d_model隐藏宽度；num_fine_experts候选数；num_shared_experts公共专家数；top_k每帧选几个；expert_hidden_dim/router_hidden_dim各自宽度；check_values检查开关。
        输出：无；模块输出特征增量，由外层加残差。"""
        super().__init__()
        sizes = {
            "d_model": d_model,
            "num_fine_experts": num_fine_experts,
            "top_k": top_k,
            "expert_hidden_dim": expert_hidden_dim,
            "router_hidden_dim": router_hidden_dim,
        }
        for name, value in sizes.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if type(num_shared_experts) is not int or num_shared_experts < 0:
            raise ValueError("num_shared_experts 必须是非负整数")
        if top_k > num_fine_experts:
            raise ValueError("top_k 不能超过 Fine Expert 数量")
        if type(check_values) is not bool:
            raise TypeError("check_values 必须是 bool")

        self.d_model = d_model
        self.num_fine_experts = num_fine_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim
        self.router_hidden_dim = router_hidden_dim
        self.check_values = check_values

        # Router 是可训练的 MLP，它不是 if/else 规则。
        self.router = nn.Sequential(
            nn.Linear(d_model, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, num_fine_experts),
        )
        # Fine Experts：每个真实 token 只计算分数最高的 K 个。
        self.fine_experts = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_hidden_dim)
             for _ in range(num_fine_experts)]
        )
        # Shared Experts：每个真实 token 都经过；允许设成 0 做消融对照。
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_hidden_dim)
             for _ in range(num_shared_experts)]
        )

    def get_extra_state(self) -> dict:
        """保存专家数量与路由定义。

        输入：无。输出：结构字典；用于拒绝形状碰巧相同但算法含义不同的checkpoint。"""
        return {
            "schema": self.SCHEMA_VERSION,
            "d_model": self.d_model,
            "fine": self.num_fine_experts,
            "shared": self.num_shared_experts,
            "top_k": self.top_k,
            "expert_hidden": self.expert_hidden_dim,
            "router_hidden": self.router_hidden_dim,
            "routing": "reference_only_softmax_topk_renorm",
            "expert": "swiglu_sparse_fine_mean_shared",
            "residual": "not_included",
        }

    def set_extra_state(self, state: dict) -> None:
        """加载MoE权重前检查它是不是同一套专家配置。

        输入：state结构字典。输出：无；不一致立即报错。"""
        if state != self.get_extra_state():
            raise ValueError("MoE checkpoint 的网络结构或接口版本不匹配")

    def _validate_inputs(self, x, router_context, valid_mask):
        """严格检查 shape、dtype、设备；拒绝把不同时间或环境混在一起。"""
        if not isinstance(x, torch.Tensor) or x.ndim not in (2, 3):
            raise TypeError("x 必须是 [B,D] 或 [B,T,D] Tensor")
        if any(v == 0 for v in x.shape) or x.shape[-1] != self.d_model:
            raise ValueError(f"x 非空，最后一维必须为 {self.d_model}")
        if not isinstance(router_context, torch.Tensor):
            raise TypeError("router_context 必须是 Tensor")
        if router_context.shape != x.shape:
            raise ValueError("router_context 必须与 x 完全同形状")
        p = self.router[0].weight
        if p.dtype not in (torch.float32, torch.float64):
            raise TypeError("权重应使用 float32/64，混合精度请用 autocast")
        allowed = {p.dtype}
        if p.dtype == torch.float32 and torch.is_autocast_enabled(p.device.type):
            allowed.add(torch.get_autocast_dtype(p.device.type))
        for name, value in (("x", x), ("router_context", router_context)):
            if value.device != p.device:
                raise ValueError(f"{name} 与网络不在同一设备")
            if value.dtype not in allowed:
                raise TypeError(f"{name} dtype 与模型/autocast 不匹配")
            if self.check_values and not torch.isfinite(value).all().item():
                raise ValueError(f"{name} 包含 NaN 或 Inf")
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        elif (not isinstance(valid_mask, torch.Tensor)
              or valid_mask.shape != x.shape[:-1]
              or valid_mask.dtype != torch.bool
              or valid_mask.device != x.device):
            raise ValueError("valid_mask 必须是同设备、同前导形状的 bool Tensor")
        if (self.check_values and x.ndim == 3
            and (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any().item()):
            raise ValueError("时间补位必须位于序列末尾")
        return valid_mask

    def forward(
        self,
        x: torch.Tensor,
        router_context: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> MoEOutput:
        """让Router按条件选专家，再组合专家的输出。

        输入：x [B,D]或[B,T,D]要处理的特征；router_context同形状的路由条件；valid_mask表示哪些环境/时刻是真实数据。
        输出：MoEOutput，包含delta同形状特征增量、Top-K结果、专家负载和可选balance_loss。
        注意：top_k=1时当前选中后softmax恒为1，任务损失不能通过组合权重训练Router；默认用top_k=2。"""
        valid = self._validate_inputs(x, router_context, valid_mask)
        leading = x.shape[:-1]
        # 先把 [B,T,D] 摊平为 [B*T,D]；不混合不同环境的数值。
        flat_x = torch.where(valid.unsqueeze(-1), x, 0.0).reshape(-1, self.d_model)
        flat_ref = torch.where(valid.unsqueeze(-1), router_context, 0.0).reshape(-1, self.d_model)
        flat_valid = valid.reshape(-1)
        n_tokens = flat_x.shape[0]
        n_valid = flat_valid.sum()  # Tensor；统计时不用强制同步成 Python 数字。

        # Router 先给所有 Fine Expert 打分，训练时通过梯度学习打分方法。
        logits = self.router(flat_ref)
        # 低精度训练时，用 float32 计算 Softmax，减少数值误差。
        logits = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.where(flat_valid[:, None], probs, 0.0)

        # 按每个 token 的分数选 K 个专家，并把选中权重归一化到和为 1。
        picked_logits, picked = torch.topk(logits, k=self.top_k, dim=-1)
        picked_weights = torch.softmax(picked_logits, dim=-1)
        picked_weights = torch.where(flat_valid[:, None], picked_weights, 0.0)
        picked = torch.where(flat_valid[:, None], picked, -1)

        # Fine 专家只执行被选中的 token；index_add 可以反向传播到专家和权重。
        fine_delta = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.fine_experts):
            rows, slots = (picked == expert_id).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            input_for_expert = flat_x.index_select(0, rows)
            expert_output = expert(input_for_expert)
            weight = picked_weights[rows, slots].to(expert_output.dtype)
            fine_delta = fine_delta.index_add(
                0, rows, (expert_output * weight[:, None]).to(fine_delta.dtype)
            )

        # Shared 专家始终参与真实 token；采用平均，避免 shared 数量改变输出量级。
        shared_delta = torch.zeros_like(flat_x)
        if self.num_shared_experts:
            for expert in self.shared_experts:
                shared_delta = shared_delta + expert(flat_x)
            shared_delta = shared_delta / self.num_shared_experts
        delta = torch.where(flat_valid[:, None], fine_delta + shared_delta, 0.0)

        # 负载诊断：占全部 Top-K 选择的比例。均匀使用时每个专家接近 1/E。
        counts = torch.stack([((picked == i) & flat_valid[:, None]).sum()
                              for i in range(self.num_fine_experts)])
        load = counts.to(probs.dtype) / (n_valid * self.top_k).clamp_min(1)
        mean_probs = probs.sum(dim=0) / n_valid.clamp_min(1)
        # Switch-style 负载均衡项；硬选择的 load 不传梯度，概率分布传梯度。
        # 均匀选中通常接近 1，集中到少数专家可能更大；训练器决定是否采用。
        balance = self.num_fine_experts * (load.detach() * mean_probs).sum()
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum()
        entropy = entropy / n_valid.clamp_min(1)

        # 开发时尽早发现坏权重、溢出或错误的 Softmax；正式性能测试可关闭同步检查。
        if self.check_values:
            for name, value in (("delta", delta), ("router_probs", probs),
                                ("topk_weights", picked_weights),
                                ("balance_loss", balance), ("router_entropy", entropy)):
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"MoE {name} 包含 NaN 或 Inf")

        return MoEOutput(
            delta=delta.reshape(*leading, self.d_model),
            balance_loss=balance,
            router_entropy=entropy.detach(),
            router_probs=probs.reshape(*leading, self.num_fine_experts),
            topk_indices=picked.reshape(*leading, self.top_k),
            topk_weights=picked_weights.reshape(*leading, self.top_k),
            expert_counts=counts.detach(),
            expert_load=load.detach(),
        )




