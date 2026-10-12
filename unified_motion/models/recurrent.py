"""可选的 GRU / LSTM 时间记忆，不把两种网络强行串在一起。

这个文件做什么：
    RecurrentCore 是独立插口：输入特征 [B,T,D]，输出同一时间序列的新特征。
    RecurrentReferenceBackbone 把这个插口接到现有物理 Actor，复用参考融合。

三个容易混淆的概念：
    1. 本文件的记忆属于 Actor，用来决定动作。
    2. DynamicsModel 的 GRU 记忆属于独立动力学模型，用来辨识和预测物理变化。
    3. 两者不是同一个 hidden，不能共用缓存，也不能相互覆盖。

实现约定：
    - 只有单向网络，没有 Dropout；不能偷看未来真实状态。
    - episode_start=True 清空该环境的全部层；LSTM 同时清空 h 和 c。
    - 推理使用 step，每步只更新一次记忆；PPO 用真实历史重新计算。
    - 权重更新后旧记忆失效。Actor 的循环记忆没有有限窗口假设，采样器必须
      保留至真实回合起点的前缀，不能只保留 context_len 帧。
    - 这里没有训练入口和内嵌自测，回归统一放在 tests/。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import torch
from torch import nn

from .encoder import EncodedMotion
from .reference_fusion import ReferenceFusion
from .transformer import TransformerOutput
from .lru import LRUCore, LRUState

if TYPE_CHECKING:
    from .physics_policy import PhysicsPolicyConfig


@dataclass
class RecurrentState:
    """一份循环记忆。

    hidden：[L,B,R]，L层、B个环境、每层R维记忆。
    cell：LSTM 额外的 [L,B,R] 单元记忆；GRU 必须为 None。
    这是运行时状态，不是网络参数，不写入模型 state_dict。
    """
    hidden: torch.Tensor
    cell: torch.Tensor | None = None


@dataclass
class RecurrentCache:
    """Actor 推理缓存，绑定机器人、环境行顺序和创建它的网络版本。

    state 是GRU/LSTM的h/c，或LRU的实部/虚部；owner/stamp 防止旧权重的记忆被继续使用。
    policy_stamp 由上层 Actor 写入，连输入 Encoder 改权重也能检查出来。
    """
    robot_id: str
    env_ids: tuple[int, ...]
    state: RecurrentState | LRUState
    owner: object = field(repr=False)
    stamp: tuple[int, ...] = field(repr=False)
    policy_stamp: tuple[int, ...] | None = field(default=None, repr=False)


def weight_stamp(module: nn.Module) -> tuple[int, ...]:
    """功能：记录当前参数的身份与原地修改计数。

    输入：一个网络。输出：仅本进程使用的元组，不是可携带的 checkpoint。
    optimizer.step/load_state_dict 后计数会改变；禁止用 .data 绕过这个检查。
    """
    return tuple(v for p in module.parameters() for v in (id(p), p._version))


class RecurrentCore(nn.Module):
    """真正的 GRU/LSTM 插口，可以脱离机器人项目单独使用。

    输入宽度 input_dim，中间记忆宽度 hidden_dim，输出宽度 output_dim。
    forward_sequence：整段 [B,T,input_dim] -> [B,T,output_dim]，允许反传。
    step：一帧 [B,input_dim] -> [B,output_dim]，返回更新后的 RecurrentState。
    reset 使用每个环境自己的标记，不会因一台机器人重置而清空全部环境。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 *, kind: Literal['gru', 'lstm'] = 'gru', n_layers: int = 1) -> None:
        """创建循环单元与输出投影。

        输入：三个正整数宽度、kind、层数。输出：可学习模块，无自动训练副作用。
        nn.GRUCell/nn.LSTMCell 只管一帧；本类显式管理回合边界和补位。
        """
        super().__init__()
        if kind not in ('gru', 'lstm'):
            raise ValueError("kind 必须是 gru 或 lstm")
        for name, value in dict(input_dim=input_dim, hidden_dim=hidden_dim,
                                output_dim=output_dim, n_layers=n_layers).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.output_dim, self.n_layers, self.kind = output_dim, n_layers, kind
        cell_type = nn.GRUCell if kind == 'gru' else nn.LSTMCell
        self.cells = nn.ModuleList([
            cell_type(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])
        self.projection = nn.Linear(hidden_dim, output_dim)

    def get_extra_state(self) -> dict:
        """输出结构定义；GRU/LSTM 即使碰巧尺寸一样，也不能静默互载权重。"""
        return dict(schema=1, kind=self.kind, input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim, output_dim=self.output_dim,
                    n_layers=self.n_layers, direction='causal')

    def set_extra_state(self, state: dict) -> None:
        """输入旧结构字典；相同才允许继续载入，不推测用户意图。"""
        if state != self.get_extra_state():
            raise ValueError("循环网络 checkpoint 结构不一致")

    def initial_state(self, batch_size: int) -> RecurrentState:
        """生成空记忆，设备和精度跟随网络。

        输入：B个环境。输出：全零 h，以及 LSTM 的全零 c。
        零记忆只用于真正开始回合或明确的历史回放起点，不应随便清空。
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        p = self.projection.weight
        h = p.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        return RecurrentState(h, torch.zeros_like(h) if self.kind == 'lstm' else None)

    def _check_state(self, state: RecurrentState, batch: int) -> None:
        """检查 h/c 是否属于当前形状、设备和浮点类型；错误时立即报错。"""
        if not isinstance(state, RecurrentState):
            raise TypeError("state 必须是 RecurrentState")
        if (self.kind == 'lstm') != (state.cell is not None):
            raise ValueError("LSTM 必须有 cell；GRU 不允许传 cell")
        p = self.projection.weight
        for value in (state.hidden,) + ((state.cell,) if state.cell is not None else ()):
            if not isinstance(value, torch.Tensor) or value.shape != (self.n_layers, batch, self.hidden_dim):
                raise ValueError("循环记忆必须是 [层数,环境数,hidden_dim]")
            if value.device != p.device or value.dtype != p.dtype:
                raise ValueError("记忆的设备或精度不一致，请重新建立")
            if not torch.isfinite(value).all().item():
                raise ValueError("循环记忆含 NaN/Inf")

    @staticmethod
    def _check_mask(mask: torch.Tensor, shape: tuple[int, ...], device: torch.device,
                    name: str) -> None:
        """输入布尔标记与预期形状；输出无，禁止广播错环境或错时间。"""
        if (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != shape
                or mask.dtype != torch.bool or mask.device != device):
            raise ValueError(f"{name} 必须是同设备 {shape} bool")

    def step(self, x: torch.Tensor, state: RecurrentState | None, *,
             episode_start: torch.Tensor, valid_mask: torch.Tensor | None = None,
             ) -> tuple[torch.Tensor, RecurrentState]:
        """计算一帧，既可用于训练中的逐帧展开，也可在 no_grad 下推理。

        输入：x [B,D]；state 为上一时刻 h/c；episode_start [B]。
        valid_mask [B]：False 表示补位，该行输出零，且不更新原记忆。
        输出：y [B,output_dim] 与新的 RecurrentState；不原地修改输入记忆。
        LSTM 的 hidden 与 cell 都在新回合开始前清零。
        """
        p = self.projection.weight
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] < 1
                or x.shape[1] != self.input_dim):
            raise ValueError("循环单步输入应为非空 [B,input_dim]")
        # 物理策略权重维持float32/64；不把半精度旧记忆偷偷转成另一个状态。
        if x.device != p.device or x.dtype != p.dtype:
            raise ValueError("循环输入必须与网络在同一设备和浮点类型")
        if not torch.isfinite(x).all().item():
            raise ValueError("循环输入含 NaN/Inf")
        b = x.shape[0]
        valid = torch.ones(b, device=x.device, dtype=torch.bool) if valid_mask is None else valid_mask
        self._check_mask(valid, (b,), x.device, 'valid_mask')
        self._check_mask(episode_start, (b,), x.device, 'episode_start')
        if (episode_start & ~valid).any().item():
            raise ValueError("补位帧不能标记新回合")
        state = self.initial_state(b) if state is None else state
        self._check_state(state, b)
        keep = ~episode_start[:, None]
        z = torch.where(valid[:, None], x, 0.0)
        hs, cs = [], []
        # 显式保持记忆与参数的精度；外层使用 autocast 也不积累低精度 h/c。
        with torch.autocast(device_type=x.device.type, enabled=False):
            for i, cell in enumerate(self.cells):
                old_h = state.hidden[i]
                h = torch.where(keep, old_h, 0.0)
                if self.kind == 'lstm':
                    old_c = state.cell[i]
                    c = torch.where(keep, old_c, 0.0)
                    h, c = cell(z, (h, c))
                    cs.append(torch.where(valid[:, None], c, old_c))
                else:
                    h = cell(z, h)
                h = torch.where(valid[:, None], h, old_h)
                hs.append(h)
                z = h
            y = self.projection(z)
        y = torch.where(valid[:, None], y, 0.0)
        if not torch.isfinite(y).all().item():
            raise ValueError("循环输出含 NaN/Inf")
        return y, RecurrentState(torch.stack(hs), torch.stack(cs) if cs else None)

    def forward_sequence(self, x: torch.Tensor, *, episode_start: torch.Tensor,
                         valid_mask: torch.Tensor | None = None,
                         initial_state: RecurrentState | None = None,
                         ) -> tuple[torch.Tensor, RecurrentState]:
        """按真实顺序处理整段，不能打乱 T。

        输入：x [B,T,D]，episode_start / valid_mask [B,T]。
        initial_state 可用于有明确历史的训练；上层 PPO 默认从真实回合前文重算。
        输出：全部 y [B,T,output_dim] 和末尾 h/c。无效补位只能在末尾。
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or min(x.shape[:2]) < 1:
            raise ValueError("整段循环输入必须是非空 [B,T,D]")
        shape = tuple(x.shape[:2])
        valid = torch.ones(shape, dtype=torch.bool, device=x.device) if valid_mask is None else valid_mask
        self._check_mask(valid, shape, x.device, 'valid_mask')
        self._check_mask(episode_start, shape, x.device, 'episode_start')
        if (valid[:, 1:] & ~valid[:, :-1]).any().item():
            raise ValueError("时间补位只能出现在末尾")
        state, ys = initial_state, []
        for t in range(x.shape[1]):
            y, state = self.step(x[:, t], state, episode_start=episode_start[:, t],
                                 valid_mask=valid[:, t])
            ys.append(y)
        return torch.stack(ys, dim=1), state


class RecurrentReferenceBackbone(nn.Module):
    """把可选循环网络接入已有 Actor，不复制目标编码、动作头或 PPO。

    结构：现有 ReferenceFusion -> GRU / LSTM / LRU 之一 -> LayerNorm。
    这里没有时间 self-attention，也没有 MoE；共享的是输入目标接口。
    它可以像 Dense/MoE 主干一样接收 EncodedMotion 并返回 TransformerOutput。
    """

    def __init__(self, cfg: PhysicsPolicyConfig) -> None:
        """输入统一策略配置，创建指定的一种循环主干，不同时建立 GRU、LSTM 和 LRU。"""
        super().__init__()
        self.cfg = cfg
        self.reference_fusion = ReferenceFusion(cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                                                check_values=cfg.check_values)
        if cfg.backbone == "lru":
            self.core = LRUCore(cfg.d_model, cfg.lru_state_dim or cfg.d_model, cfg.d_model,
                n_layers=cfg.n_layers, r_min=cfg.lru_r_min, r_max=cfg.lru_r_max,
                max_phase=cfg.lru_max_phase, scan_mode=cfg.lru_scan)
        else:
            width = cfg.recurrent_hidden_dim or cfg.d_model
            self.core = RecurrentCore(cfg.d_model, width, cfg.d_model,
                                       kind=cfg.backbone, n_layers=cfg.n_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self._owner = object()

    def invalidate_caches(self) -> None:
        """权重变化后作废旧 h/c；Collector 会用新权重重放真实历史。"""
        self._owner = object()

    def train(self, mode: bool = True):
        """切换运行模式并废弃旧记忆；返回模型自身，保持 PyTorch 接口。"""
        old = self.training
        super().train(mode)
        if old != mode:
            self.invalidate_caches()
        return self

    @staticmethod
    def _output(x: torch.Tensor, summary: torch.Tensor,
                valid: torch.Tensor) -> TransformerOutput:
        """包装与旧主干相同的输出；没有专家，所以专家统计是空张量而不是假概率。"""
        return TransformerOutput(x, summary, valid.clone(), x.new_zeros(()),
                                 x.new_zeros(0), x.new_zeros((0, 0)), x.new_zeros((0, 0)))

    def forward(self, encoded: EncodedMotion, *, valid_mask: torch.Tensor,
                episode_start: torch.Tensor) -> TransformerOutput:
        """整段 [B,T,D] -> [B,T,D]，允许梯度通过此前真实历史。

        每个学习时刻是否有完整前文由 Collector/PPO 检查；不接受过期隐藏状态。
        """
        fused = self.reference_fusion(encoded.state_token, encoded.reference_tokens,
                                      encoded.reference_mask, time_mask=valid_mask)
        p = next(self.core.parameters())
        x, _ = self.core.forward_sequence(fused.policy_features.to(p.dtype),
                    episode_start=episode_start, valid_mask=valid_mask)
        x = torch.where(valid_mask[..., None], self.final_norm(x), 0.0)
        return self._output(x, fused.reference_summary, valid_mask)

    @torch.no_grad()
    def step(self, encoded: EncodedMotion, *, env_ids: tuple[int, ...],
             cache: RecurrentCache | None, episode_start: torch.Tensor,
             ) -> tuple[TransformerOutput, RecurrentCache]:
        """单步推理；只消费当前一帧，不反复跑历史。

        输入：EncodedMotion 单帧、固定环境身份、上次 h/c、新回合标记。
        输出：统一特征输出和新 RecurrentCache。权重变化/环境换行会拒绝旧缓存。
        """
        if self.training:
            raise RuntimeError("循环采样前必须 .eval()")
        b = encoded.state_token.shape[0]
        ids = tuple(env_ids)
        if len(ids) != b or len(set(ids)) != b or any(type(i) is not int for i in ids):
            raise ValueError("env_ids 必须与批次一致，且是无重复整数")
        stamp = weight_stamp(self)
        if cache is not None:
            if (not isinstance(cache, RecurrentCache) or cache.owner is not self._owner
                    or cache.stamp != stamp or cache.robot_id != encoded.robot_id
                    or cache.env_ids != ids):
                raise ValueError("循环缓存过期，或属于另一机器人/环境排列")
        elif not episode_start.all().item():
            raise ValueError("没有循环缓存时必须从新回合或已校验的历史重放边界开始")
        fused = self.reference_fusion(encoded.state_token, encoded.reference_tokens,
                                      encoded.reference_mask)
        p = next(self.core.parameters())
        x, state = self.core.step(fused.policy_features.to(p.dtype),
                    None if cache is None else cache.state, episode_start=episode_start)
        x = self.final_norm(x)
        valid = torch.ones(b, device=x.device, dtype=torch.bool)
        return self._output(x, fused.reference_summary, valid), RecurrentCache(
            encoded.robot_id, ids, state, self._owner, stamp)
