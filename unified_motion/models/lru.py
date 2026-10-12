"""LRU（Linear Recurrent Unit，线性循环单元）的独立网络插口。

这里不是“最近最少使用”的缓存算法，而是 Orvieto 等人在 ICML 2023 的 LRU。
来源：Resurrecting Recurrent Neural Networks for Long Sequences，论文附录 A。
https://proceedings.mlr.press/v202/orvieto23a.html

它做什么：把过去的信息放进固定大小的记忆，再结合当前输入产生新特征。
核心递推是 h[t] = lambda * h[t-1] + gamma * B * u[t]，输出是 Re(C*h[t])+D*u[t]。
- lambda 是复数对角系数；用两个实数分量计算，不创建复杂的复数网络权重。
- lambda = exp(-exp(nu_log)) * (cos(exp(theta_log)) + i*sin(exp(theta_log)))。
- gamma 可训练，初始化为 sqrt(1-|lambda|²)，初始化以后不是强制绑在 lambda 上。
- 只有记忆的递推是线性的；完整网络在各 LRU 层之间使用非线性和残差。
- 并行模式使用可反向传播的配对前缀扫描，不调用不稳定的私有 PyTorch API。
- 每台环境单独 reset，末尾补位输出为零且保持旧记忆；不读未来真实状态。

这是自研的机器人接线版本，不是论文实验结果复现，也不保证一定比 GRU 更快。
LRU 的记忆属于 Actor；不会替换或共用独立 DynamicsModel 的 GRU 记忆。
B=环境数量，T=时间长度，D=特征宽度，N=复数记忆宽度，L=LRU 层数。
本文件不自动训练；测评集中在 tests/。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LRUState:
    """保存所有层的复数记忆，不是 LSTM 的 h/c。

    输入/输出：hidden 为 [L,B,N,2] 浮点张量；最后的两个数分别是实部、虚部。
    状态跟随网络使用 float32/64。N个复数占2N个实数，参数统计不能漏掉这点。
    这是随交互变化的运行时状态，不写入模型的 state_dict。
    """
    hidden: torch.Tensor


def validate_lru_settings(state_dim: int | None, r_min: float, r_max: float,
                          max_phase: float, scan_mode: str) -> None:
    """检查 LRU 配置，错误立即报出，不让配置拼错后悄悄换算法。

    输入：记忆宽度、初始特征值半径范围、初始相位范围、扫描方式。
    输出：无；合法时返回，不合法抛异常。r_min/r_max仅控制初始化，不是训练时硬范围。
    """
    if state_dim is not None and (type(state_dim) is not int or state_dim < 1):
        raise ValueError("lru_state_dim 必须是正整数或 None")
    for name, value in (("r_min", r_min), ("r_max", r_max), ("max_phase", max_phase)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"LRU {name} 必须是有限实数")
    if not 0 <= r_min < r_max < 1:
        raise ValueError("LRU 初始化要求 0 <= r_min < r_max < 1")
    if not 0 < max_phase <= 2 * math.pi:
        raise ValueError("LRU max_phase 必须在 (0, 2*pi]，单位弧度")
    if scan_mode not in ("parallel", "sequential"):
        raise ValueError("lru_scan 必须为 parallel 或 sequential")


def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """用两个实数表示复数乘法；输入 [...,2]，输出同形状。

    第0列是实部，第1列是虚部。普通实数运算可以直接被 PyTorch 反向传播。
    """
    return torch.stack((a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1],
                        a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]), dim=-1)


def _parallel_prefix(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """合并一段逐时刻的仿射变换，得到每个时刻的累计变换。

    输入：a/b 为 [B,T,N,2]，含义是每一步 h_new=a*h_old+b。
    输出：累计 a/b，形状不变，因此 h[t]=a_prefix[t]*h_initial+b_prefix[t]。
    先合并相邻两步，再递归处理一半长度，最后补回偶数位置。
    T不需要是2的幂。张量没有原地修改，梯度能沿全部历史回传。
    理论扫描工作量为O(B*T*N)，递归深度O(log T)；不是已优化的CUDA内核。
    """
    t = a.shape[1]
    if t == 1:
        return a, b
    # 两步的先后不能颠倒：先执行左边，再执行右边。
    right_a, right_b = a[:, 1::2], b[:, 1::2]
    pair_a = _complex_mul(right_a, a[:, :t-1:2])
    pair_b = _complex_mul(right_a, b[:, :t-1:2]) + right_b
    odd_a, odd_b = _parallel_prefix(pair_a, pair_b)
    extra = a[:, 2::2].shape[1]
    even_a = torch.cat((a[:, :1], _complex_mul(a[:, 2::2], odd_a[:, :extra])), dim=1)
    even_b = torch.cat((b[:, :1], _complex_mul(a[:, 2::2], odd_b[:, :extra]) + b[:, 2::2]), dim=1)
    pairs = t // 2
    all_a = torch.stack((even_a[:, :pairs], odd_a), dim=2).flatten(1, 2)
    all_b = torch.stack((even_b[:, :pairs], odd_b), dim=2).flatten(1, 2)
    if t % 2:
        all_a = torch.cat((all_a, even_a[:, -1:]), dim=1)
        all_b = torch.cat((all_b, even_b[:, -1:]), dim=1)
    return all_a, all_b


def _mask(mask: torch.Tensor, shape: tuple[int, ...], device: torch.device, name: str) -> None:
    """输入布尔标记和期望形状；检查设备、形状与类型，输出无。

    不允许借助广播把一台环境的 reset 错用到全部环境。
    """
    if (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != shape
            or mask.dtype != torch.bool or mask.device != device):
        raise ValueError(f"{name} 必须是同设备 {shape} bool")


class LRULayer(nn.Module):
    """单层线性递推：输入/输出 [B,T,D]，复数记忆 [B,N,2]。

    本层不放非线性到 h 的时间递推里；LRUCore负责层间非线性。
    所有可训练参数为实数。只有记忆和系数通过最后一维的两个数表示复数。
    """
    def __init__(self, width: int, state_dim: int, *, r_min: float = .9,
                 r_max: float = .999, max_phase: float = 2*math.pi) -> None:
        """创建稳定对角递推和输入/输出投影，不启动训练。

        输入：D=width、N=state_dim，以及初始化区间。
        输出：可训练层。半径按圆环面积均匀初始化，gamma按初始半径缩放输入。
        """
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("LRU width 必须是正整数")
        validate_lru_settings(state_dim, r_min, r_max, max_phase, "parallel")
        if state_dim is None:
            raise ValueError("LRULayer 必须明确 state_dim")
        self.width, self.state_dim = width, state_dim
        eps = torch.finfo(torch.float32).eps
        u = torch.rand(state_dim).clamp(min=eps, max=1-eps)
        r2 = r_min*r_min + (r_max*r_max - r_min*r_min)*u
        if ((r2 <= 0) | (r2 >= 1)).any().item():
            raise ValueError("LRU 半径区间在float32精度下过于靠近0或1，请缩小范围")
        decay = -.5 * r2.log()
        phase = max_phase * torch.rand(state_dim).clamp_min(eps)
        self.nu_log = nn.Parameter(decay.log())
        self.theta_log = nn.Parameter(phase.log())
        self.gamma_log = nn.Parameter(.5 * torch.log1p(-r2))
        self.b_real = nn.Parameter(torch.randn(state_dim, width) / math.sqrt(2*width))
        self.b_imag = nn.Parameter(torch.randn(state_dim, width) / math.sqrt(2*width))
        self.c_real = nn.Parameter(torch.randn(width, state_dim) / math.sqrt(state_dim))
        self.c_imag = nn.Parameter(torch.randn(width, state_dim) / math.sqrt(state_dim))
        self.skip = nn.Parameter(torch.randn(width))

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """把可学习参数转换成复数对角 lambda 和正输入缩放 gamma。

        输入：本层参数。输出：lambda [N,2]，gamma [N]。
        不把谱稳定说成机器人不会摔倒。极端参数导致浮点溢出/半径舍入到1时直接报错。
        """
        decay = self.nu_log.exp()
        phase = self.theta_log.exp()
        radius = (-decay).exp()
        gamma = self.gamma_log.exp()
        if (not torch.isfinite(decay).all().item() or not torch.isfinite(phase).all().item()
                or not torch.isfinite(gamma).all().item() or (gamma <= 0).any().item()
                or (radius >= 1).any().item() or not torch.isfinite(radius).all().item()):
            raise ValueError("LRU 系数溢出、下溢或失去严格衰减；检查学习率和参数范围")
        return torch.stack((radius*phase.cos(), radius*phase.sin()), dim=-1), gamma

    def _inputs(self, x, initial, starts, valid) -> None:
        """检查线性层的一段输入；x [B,T,D]、initial [B,N,2]、掩码 [B,T]。

        输出无。拒绝错误设备、非有限数、半精度参数和补位中间挖洞。
        """
        p = self.nu_log
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or min(x.shape[:2]) < 1
                or x.shape[-1] != self.width):
            raise ValueError("LRU 输入必须是非空 [B,T,width]")
        if p.dtype not in (torch.float32, torch.float64) or x.dtype != p.dtype or x.device != p.device:
            raise ValueError("LRU 输入/权重需同设备float32/64；不要把记忆改为half")
        if (not isinstance(initial, torch.Tensor) or initial.shape != (x.shape[0], self.state_dim, 2)
                or initial.dtype != p.dtype or initial.device != p.device):
            raise ValueError("LRU 初始记忆必须是同设备同精度 [B,N,2]")
        if not torch.isfinite(x).all().item() or not torch.isfinite(initial).all().item():
            raise ValueError("LRU 输入或记忆包含 NaN/Inf")
        _mask(starts, tuple(x.shape[:2]), x.device, "episode_start")
        _mask(valid, tuple(x.shape[:2]), x.device, "valid_mask")
        if (starts & ~valid).any().item():
            raise ValueError("补位不能标记为新回合")
        if (valid[:, 1:] & ~valid[:, :-1]).any().item():
            raise ValueError("补位只能在末尾")

    def forward_sequence(self, x: torch.Tensor, *, initial_state: torch.Tensor | None = None,
                         episode_start: torch.Tensor, valid_mask: torch.Tensor | None = None,
                         scan_mode: str = "parallel") -> tuple[torch.Tensor, torch.Tensor]:
        """计算所有时刻的线性状态和读出，不修改传入的记忆。

        输入：x [B,T,D]，初始复数记忆 [B,N,2]，reset/有效标记 [B,T]。
        输出：y [B,T,D]、最后记忆 [B,N,2]。parallel/ sequential 使用同一数学公式。
        reset切断过去；无效步的变换是h_new=h_old，最终输出清零。
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("LRU 输入必须是 [B,T,D]")
        if scan_mode not in ("parallel", "sequential"):
            raise ValueError("scan_mode 必须是 parallel 或 sequential")
        shape = tuple(x.shape[:2])
        valid = torch.ones(shape, dtype=torch.bool, device=x.device) if valid_mask is None else valid_mask
        initial = self.nu_log.new_zeros(shape[0], self.state_dim, 2) if initial_state is None else initial_state
        self._inputs(x, initial, episode_start, valid)
        with torch.autocast(device_type=x.device.type, enabled=False):
            u = torch.where(valid[..., None], x, 0.)
            lam, gamma = self.coefficients()
            drive = torch.stack((F.linear(u, self.b_real), F.linear(u, self.b_imag)), dim=-1)
            drive = drive * gamma[:, None]
            a = lam.expand(shape[0], shape[1], -1, -1)
            a = torch.where(episode_start[..., None, None], 0., a)
            identity = a.new_tensor([1., 0.])
            a = torch.where(valid[..., None, None], a, identity)
            if scan_mode == "parallel":
                pa, pb = _parallel_prefix(a, drive)
                states = _complex_mul(pa, initial[:, None]) + pb
            else:
                history, h = [], initial
                for t in range(shape[1]):
                    h = _complex_mul(a[:, t], h) + drive[:, t]
                    history.append(h)
                states = torch.stack(history, dim=1)
            y = F.linear(states[..., 0], self.c_real) - F.linear(states[..., 1], self.c_imag)
            y = torch.where(valid[..., None], y + self.skip*u, 0.)
        if not torch.isfinite(y).all().item() or not torch.isfinite(states).all().item():
            raise ValueError("LRU 输出/记忆含 NaN或Inf")
        return y, states[:, -1]

    def forward(self, x: torch.Tensor, **kwargs):
        """标准PyTorch调用入口；输入与forward_sequence相同，输出(y, 最后记忆)。"""
        return self.forward_sequence(x, **kwargs)

    def step(self, x: torch.Tensor, state: torch.Tensor | None, *,
             episode_start: torch.Tensor, valid_mask: torch.Tensor | None = None):
        """一帧 [B,D] -> [B,D] 和 [B,N,2] 记忆；不回放全部历史。

        输入可带训练梯度；部署由调用方使用no_grad。当前状态不会原地覆盖旧状态。
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("LRU step 输入必须是 [B,D]")
        _mask(episode_start, (x.shape[0],), x.device, "episode_start")
        if valid_mask is not None:
            _mask(valid_mask, (x.shape[0],), x.device, "valid_mask")
        y, h = self.forward_sequence(x[:, None], initial_state=state,
            episode_start=episode_start[:, None],
            valid_mask=None if valid_mask is None else valid_mask[:, None], scan_mode="sequential")
        return y[:, 0], h


class _LRUBlock(nn.Module):
    """一层Pre-LN → LRU → 逐帧GLU → 残差；非线性不在复数时间递推内部。"""
    def __init__(self, width, state_dim, r_min, r_max, max_phase):
        """输入特征/记忆宽度和初始化范围；输出可学习block，无训练副作用。"""
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lru = LRULayer(width, state_dim, r_min=r_min, r_max=r_max, max_phase=max_phase)
        self.glu = nn.Linear(width, 2*width)

    def finish(self, x, readout, valid):
        """输入原特征、LRU读出、有效标记；输出同宽度特征。

        GLU根据当前读出调节信息，不选专家。仅本时刻计算，不窥视其他时刻。
        """
        value, gate = self.glu(F.gelu(readout)).chunk(2, dim=-1)
        return torch.where(valid[..., None], x + value*torch.sigmoid(gate), 0.)


class LRUCore(nn.Module):
    """可独立复用的多层LRU。

    输入 [B,T,input_dim] 或一帧 [B,input_dim]；输出 [...,output_dim]。
    state_dim 是每层复数记忆数，不是关节数；n_layers层共有2*L*N个实数记忆/环境。
    参数稳定只限制线性递推系数；输入、输出投影和机器人闭环仍需训练与验证。
    """
    def __init__(self, input_dim: int, state_dim: int, output_dim: int, *, n_layers: int = 1,
                 r_min: float = .9, r_max: float = .999, max_phase: float = 2*math.pi,
                 scan_mode: str = "parallel") -> None:
        """创建输入投影及若干LRU块。

        输入：特征宽度、复数记忆宽度、层数、初始化参数、整段扫描方式。
        输出：一个可交给优化器的网络。step始终单步递推，scan_mode只选择整段实现。
        """
        super().__init__()
        for name, value in (("input_dim", input_dim), ("output_dim", output_dim), ("n_layers", n_layers)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if output_dim < 2:
            raise ValueError("LRUCore 的 output_dim 至少为2，避免一维LayerNorm抹掉输入")
        validate_lru_settings(state_dim, r_min, r_max, max_phase, scan_mode)
        if state_dim is None:
            raise ValueError("LRUCore 必须明确 state_dim")
        self.input_dim, self.state_dim, self.output_dim = input_dim, state_dim, output_dim
        self.n_layers, self.scan_mode = n_layers, scan_mode
        self.r_min, self.r_max, self.max_phase = float(r_min), float(r_max), float(max_phase)
        self.input_projection = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([_LRUBlock(output_dim, state_dim, r_min, r_max, max_phase)
                                     for _ in range(n_layers)])

    def get_extra_state(self) -> dict:
        """输出LRU结构合同；保存权重时带上它，防止同尺寸却不同公式的模型被混用。"""
        return dict(schema=1, kind="lru", input_dim=self.input_dim, state_dim=self.state_dim,
            output_dim=self.output_dim, n_layers=self.n_layers, r_min=self.r_min, r_max=self.r_max,
            max_phase=self.max_phase, scan_mode=self.scan_mode,
            recurrence="complex_diagonal_gamma_trainable_v1", block="preln_lru_gelu_glu_residual_v1")

    def set_extra_state(self, state: dict) -> None:
        """输入checkpoint结构字典；结构不符立即拒绝。裸Core的外部缓存由调用者管理。"""
        if state != self.get_extra_state():
            raise ValueError("LRU checkpoint结构不匹配")

    def initial_state(self, batch_size: int) -> LRUState:
        """输入环境数B；输出同设备同精度全零 [L,B,N,2]，用于真正新回合。"""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        p = next(self.parameters())
        return LRUState(p.new_zeros(self.n_layers, batch_size, self.state_dim, 2))

    def _check(self, x, starts, valid, state):
        """核对整段输入和全部层的状态。输入[B,T,I]、[B,T]、LRUState；输出无。

        裸Core允许显式传入有梯度的初态；正式PPO不能随便拿旧权重的记忆当初态。
        """
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or min(x.shape[:2]) < 1
                or x.shape[-1] != self.input_dim):
            raise ValueError("LRUCore 输入必须是非空 [B,T,input_dim]")
        p = next(self.parameters())
        if p.dtype not in (torch.float32, torch.float64) or x.dtype != p.dtype or x.device != p.device:
            raise ValueError("LRUCore输入/权重必须同设备float32/64")
        if not torch.isfinite(x).all().item():
            raise ValueError("LRUCore 输入含NaN/Inf，包括补位区")
        _mask(starts, tuple(x.shape[:2]), x.device, "episode_start")
        _mask(valid, tuple(x.shape[:2]), x.device, "valid_mask")
        if (starts & ~valid).any().item() or (valid[:, 1:] & ~valid[:, :-1]).any().item():
            raise ValueError("补位只能在末尾且不能标记新回合")
        if not isinstance(state, LRUState):
            raise TypeError("state 必须是LRUState，不是GRU/LSTM记忆")
        h = state.hidden
        if (not isinstance(h, torch.Tensor) or h.shape != (self.n_layers, x.shape[0], self.state_dim, 2)
                or h.device != p.device or h.dtype != p.dtype or not torch.isfinite(h).all().item()):
            raise ValueError("LRU记忆应为同设备同精度、有限的[L,B,N,2]")

    def forward_sequence(self, x: torch.Tensor, *, episode_start: torch.Tensor,
                         valid_mask: torch.Tensor | None = None, initial_state: LRUState | None = None,
                         scan_mode: str | None = None) -> tuple[torch.Tensor, LRUState]:
        """整段训练输入[B,T,I]，返回全部特征[B,T,O]和最终LRUState。

        reset按环境/时间分别生效。并行只改变计算顺序，不改变因果关系。
        scan_mode=None使用配置值，也可指定sequential进行数值对照，不改参数。
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("LRUCore输入必须是[B,T,I]")
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if valid_mask is None else valid_mask
        state = self.initial_state(x.shape[0]) if initial_state is None else initial_state
        self._check(x, episode_start, valid, state)
        mode = self.scan_mode if scan_mode is None else scan_mode
        if mode not in ("parallel", "sequential"):
            raise ValueError("scan_mode必须是parallel/sequential")
        hs = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.input_projection(torch.where(valid[..., None], x, 0.))
            for i, block in enumerate(self.layers):
                y, h = block.lru.forward_sequence(block.norm(z), initial_state=state.hidden[i],
                    episode_start=episode_start, valid_mask=valid, scan_mode=mode)
                z = block.finish(z, y, valid)
                hs.append(h)
        if not torch.isfinite(z).all().item():
            raise ValueError("LRUCore输出非有限")
        return z, LRUState(torch.stack(hs))

    def forward(self, x: torch.Tensor, **kwargs):
        """标准PyTorch调用入口；输入整段特征，输出全部特征和LRUState。"""
        return self.forward_sequence(x, **kwargs)

    def step(self, x: torch.Tensor, state: LRUState | None, *, episode_start: torch.Tensor,
             valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, LRUState]:
        """一帧[B,I]进入，返回[B,O]与新[L,B,N,2]记忆；每次只消费当前帧。

        输入：x、上一次LRUState、新回合标记；输出不原地修改输入状态。
        它不是KV Cache，也不因为只看当前帧就忘掉更早历史。
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("LRUCore.step输入必须是[B,I]")
        _mask(episode_start, (x.shape[0],), x.device, "episode_start")
        if valid_mask is not None:
            _mask(valid_mask, (x.shape[0],), x.device, "valid_mask")
        y, next_state = self.forward_sequence(x[:, None], initial_state=state,
            episode_start=episode_start[:, None],
            valid_mask=None if valid_mask is None else valid_mask[:, None], scan_mode="sequential")
        return y[:, 0], next_state
