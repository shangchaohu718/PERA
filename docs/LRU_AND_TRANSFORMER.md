# R21：LRU 与不含 MoE 的 Transformer

## 先分清两个概念

`transformer` 是原 `dense` 主干的正式易读别名。两种名字建立同一个类，权重结构完全相同，只有一份实现。
这个分支每层是因果 Attention + 一个始终工作的 SwiGLU FFN，没有 Router、Top-K 或多个稀疏专家。
FFN 复用了 `moe.py` 中名为 `SwiGLUExpert` 的计算单元，不等于使用了 MoE。可用 `inspect` 查看 `moe_layer_count: 0`。

这里的 **LRU = Linear Recurrent Unit（线性循环单元）**，不是“最近最少使用”的缓存淘汰算法，也不是 RecurrentGemma 的 RG-LRU。
本轮按 Orvieto 等人的 ICML 2023 LRU 设计线性递推；机器人接线、reset/padding 和层间网络由本项目实现，不复制论文数据或宣称复现论文分数。

## 1. 最终有哪些选择

```text
物理观测 → set 或 GNN → 独立 DynamicsModel（原 GRU）
                          ↓ context / latent
任务目标 ────────────────→ Actor主干六选一：
                          mlp / transformer / moe / gru / lstm / lru
                          ↓
                      原始关节动作
```

`dense` 和 `transformer` 是同一种，不算第七种主干。
GNN 是空间关系编码选项，不是一定要和 LRU 捆绑。
新 LRU 与纯 Transformer 选择都不构造 MoE。独立动力学模型仍有自己的 GRU；“纯 Transformer”指 Actor 时间主干，不是把整个系统的记忆全部替换。
所有分支都保留已有目标融合，因此 LRU/GRU/LSTM Actor 的目标读取仍使用交叉注意力，但它们没有时间 self-attention。

## 2. LRU 在做什么

给每一帧特征 `u[t]` 留一份复数记忆：

```text
h[t] = lambda * h[t-1] + gamma * B * u[t]
y[t] = Real(C * h[t]) + D * u[t]
```

`lambda` 是对角系数：每个记忆分量单独保留和旋转过去的信息，不使用完整的 N×N 递归矩阵。
代码不创建复数参数，使用两个普通浮点数表示实部/虚部；它与复数乘法数学一致，测试另用真正的 complex 公式核对。

稳定参数化：

```text
radius = exp(-exp(nu_log))
phase  = exp(theta_log)
lambda = radius * (cos(phase) + i*sin(phase))
gamma  = exp(gamma_log)
```

`gamma_log` 初始化为 `log(sqrt(1-radius²))`，随后单独学习。
`r_min/r_max` 只限定初始化的圆环，不是训练后的半径硬上下限。极端参数在有限精度中产生溢出或半径舍入为1时会报错。
**线性递推的谱稳定不保证整个非线性网络稳定，更不保证机器人不会摔倒。**

完整 LRU 块：`LayerNorm → LRU → GELU/GLU → 残差`。非线性在同一时刻的读出上，不放进本层线性记忆递推里。
我们保留这种结构以便表达非线性动作，但没有复现原论文全部训练超参数、Dropout 或分类任务。

## 3. 独立插口：不依赖仿真器

```python
import torch
from unified_motion.api import LRUCore

# B=2台环境，T=20个时间点，每点32个特征。
# state_dim=64是64个复数记忆，需要128个实数存储。
core = LRUCore(input_dim=32, state_dim=64, output_dim=32,
               n_layers=2, scan_mode="parallel")
x = torch.randn(2, 20, 32)
starts = torch.zeros(2, 20, dtype=torch.bool)
starts[:, 0] = True
starts[0, 10] = True    # 第0台环境这时进入新回合，不影响第1台。
y, memory = core.forward_sequence(x, episode_start=starts)
# y: [2,20,32]；memory.hidden: [2层,2环境,64记忆,2实虚分量]
```

逐帧接口：`y, state = core.step(x_t, state, episode_start=starts_t)`。
裸 Core 的状态不自动绑定环境ID和参数版本；正式 Actor 使用原 `RecurrentCache` 对这些信息做检查。
不要把 `LRUState.hidden` 当成 LSTM 的 cell；两种状态对象不同，错误混用会拒绝。

## 4. 并行与顺序扫描

- `parallel`：通过相邻仿射变换的结合律计算前缀，使用普通 PyTorch 张量算子，可反向传播。
- `sequential`：按时间逐步展开相同递推，便于数值核对。
- `step()`：始终只处理一帧，不重放已经执行的全部历史。

配对扫描的总算术工作量随 T 线性增长、递归深度对数增长，**不意味着每个 GPU 上都更快**。
目前不是 fused CUDA 内核，Python递归、投影、严格数值检查和PPO历史重放都有成本。
两条计算路径会有浮点求和顺序差异，测评使用明确容差，不要求逐位相同。
测试比较了独立复数公式、参数和输入梯度、有限差分、513帧序列、reset与padding。

## 5. YAML 怎么改

LRU：

```yaml
model:
  backbone: lru
  d_model: 32
  hidden_dim: 32
  n_layers: 2
  lru_state_dim: 32
  lru_r_min: 0.9
  lru_r_max: 0.999
  lru_max_phase: 6.283185307179586
  lru_scan: parallel
  max_replay_steps: 512
```

纯 Transformer：

```yaml
model:
  backbone: transformer   # dense也合法，指向同一实现
  d_model: 32
  hidden_dim: 32
  n_layers: 2
  context_len: 32
```

这些是字段示例。完整可运行诊断配置在 `configs/diagnostic_lru.yaml` 和 `configs/diagnostic_transformer.yaml`。
GNN组合分别在 `diagnostic_gnn_lru.yaml`、`diagnostic_gnn_transformer.yaml`。
`lru_*` 字段不能放到其它主干下；`recurrent_hidden_dim`仍只用于GRU/LSTM。配置拼错直接报错。

## 6. 时序PPO与长期记忆

LRU与GRU/LSTM一样有无限历史形式的记忆。谱衰减不能用来假装某个截断长度已经“完全无影响”。
本项目先保留严格方案：PPO更新权重后，从每个环境可追溯的真实回合起点重放，并且前缀不参与本批主PPO损失。
`max_replay_steps`限制保留历史，不会静默伪造reset。超限会报错，需要用户配置合理上限或另行设计明确的近似训练算法。
并行扫描只加速同一公式计算，不取消这个历史一致性要求。
LRU的运行时记忆为 `2*n_layers*state_dim` 个实数/环境；“32复数状态”不能只计32个float。

## 7. 兼容与范围

- R20五种旧主干与set/GNN的参数结构保持；本轮用旧包中的10份checkpoint实际做严格加载。
- `transformer`/`dense`模型权重互通；完整实验续跑/评测仍要求保留checkpoint内原始配置，不能擅自把配置字符串改名后绕过全配置检查。
- LRU有新结构与参数，不把GRU或Transformer旧权重强行迁移到它。
- 没有新增私有数据需求，没有开启AMP，没有自动堆叠全部网络。
- 当前所有短程接线实验仍为CPU解析诊断，不是K1/G1物理仿真。诊断checkpoint禁止用于实机。
- 本轮不补真实运动平台、滑板机构、G1物理环境或混合任务调度；这些旧缺项继续明确保留。

## 来源

Orvieto et al., *Resurrecting Recurrent Neural Networks for Long Sequences*, ICML 2023，尤其附录A中的初始化与复数对角递推。
https://proceedings.mlr.press/v202/orvieto23a.html
https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf

论文来源支持数学设计，不是本项目的机器人性能或速度证据。
