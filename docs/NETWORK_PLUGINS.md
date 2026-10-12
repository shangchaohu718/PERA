# R21补充

现在共有六种Actor：mlp、transformer（旧名dense）、moe、gru、lstm、lru；set/GNN仍独立选择。
LRU实现与数学、状态和YAML说明见[LRU_AND_TRANSFORMER.md](LRU_AND_TRANSFORMER.md)。
以下R20部分保留用于解释既有GRU/LSTM和GNN插口；不应将其中“五种”当作R21的当前总数。

---

# 网络插口说明：先有替换能力，不预定最终组合

## 1. 五种Actor、两种物理编码，不是一次叠上七个模块

统一的数据流：

```text
真实物理观测
   → set或GNN空间编码
   → 原独立动力学GRU + 身体/接触/物体预测误差
   → context与latent
   → MLP / Dense / MoE / GRU / LSTM 之一
   → 原始动作 → 物理执行 → 新观测
```

GNN目前由DynamicsUpdater训练。策略PPO使用冻结的采样时context/latent。
切换Actor时不重写PPO和动力学模型；切换图编码时不更换目标与奖励。
LSTM/GRU Actor仍复用现有参考交叉注意力输入，**没有再叠一层时间Transformer**。

## 2. GRU/LSTM独立接口

```python
import torch
from unified_motion.models.recurrent import RecurrentCore

core = RecurrentCore(input_dim=32, hidden_dim=64, output_dim=32,
                     kind="lstm", n_layers=1)
# B=2：两台并行机器人；T=10：十个真实时间点；D=32：每帧特征。
x = torch.randn(2, 10, 32)
starts = torch.zeros(2, 10, dtype=torch.bool)
starts[:, 0] = True
starts[0, 6] = True  # 第0台机器人此时已经reset，不清空第1台机器人的历史。
y, memory = core.forward_sequence(x, episode_start=starts)
# y: [2,10,32]；memory.hidden: [1,2,64]；LSTM的memory.cell同形状。
```

`RecurrentCore`本身不自动执行环境或PPO。正式Actor缓存由
`RecurrentReferenceBackbone`绑定机器人ID、环境ID和权重版本。
用户自行在其它模块复用裸Core时，也必须自行管理数据时间与记忆版本。

只有单向计算，没有Dropout。逐帧和整段调用执行相同循环方程。
LSTM必须同时保留/重置h和c；GRU不使用c。

## 3. 通用GNN接口

```python
import torch
from unified_motion.models.graph_encoder import GraphBatch, MessagePassingEncoder

# 一幅图有3个节点，网络不自行解释它们是什么。
nodes = torch.randn(2, 3, 16)  # 两个环境，三节点，每节点16维
mask = torch.ones(2, 3, dtype=torch.bool)
# 显式指定0↔1、1↔2；第一行为消息来源，第二行为消息去向。
edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
types = torch.tensor([0, 0, 1, 1], dtype=torch.long)
gnn = MessagePassingEncoder(width=16, n_layers=2, num_edge_types=3)
y = gnn(GraphBatch(nodes, mask, edges, types))  # [2,3,16]
```

该例只是张量接口，不是K1真实骨架。正式接线`PhysicsRelationGraph`把已有body/
contact/object tokens转成图，再按原顺序拆回。图边数随固定槽位近似线性增加；
不会建立每个环境的全连接V×V关系矩阵。当前Python拓扑构造和数值检查仍有开销，
没有GPU吞吐测试，不宣称最优kernel。

无效节点在进入消息MLP之前屏蔽，边的分母只计算有效邻居；空边和空物体合法。
节点重编号时必须同步重编号边与类型。实际物体身份变化还要清理相应历史。

## 4. YAML关系映射为什么不能猜

`contact_object_indices`的长度等于C个接触关系。每个值是：

- `-1`：静态世界/地面，没有对应外部物体节点。
- `0,1,...`：`PhysicsObservation.object_features`的对象槽位。

`experiment.build_models()`同时检查后端的关系名和`object_names`顺序。
例如诊断场景`probe_surface → surface`必须用`[0]`，不能用`[-1]`。
真实K1默认左右脚都接触地面，使用`[-1,-1]`。
检查的是配置是否一致，不是自动视觉识别或物理可行性判断。

## 5. 历史、更新与性能边界

推理：每一步只更新当前h/c，成本不随历史长度增长。
PPO：网络更新后旧h/c不再属于新参数，不能直接当作准确起点。
当前选择从每个环境最近的真实reset历史重算，因此长回合重放成本会上升。

Collector保留共同前缀，前缀本身不计算本轮策略/价值损失。
某个环境在前缀中晚一些reset是合法的：它reset之前的暖机输出不参与本轮loss。
每个参与学习的环境必须能在前缀/当前首帧找到真实起点。
`max_replay_steps`限制缓存占用：超限报错，不用伪造reset继续训练。

以后可以研究完整回合minibatch、显式截断BPTT、burn-in近似，但那是不同训练假设，
不能在本版无提示混入。

## 6. 兼容性

- 默认`scene_encoder=set`和原三种Actor的权重结构保留。
- 新GNN有自己的参数和拓扑元数据，不能静默载入set模型当作GNN。
- 新GRU/LSTM有不同参数和状态合同，不能把Transformer checkpoint改名载入。
- R19 YAML省略新字段时沿用旧行为。
- `recurrent_hidden_dim`只允许GRU/LSTM；`set`编码器不接受伪装有效的图映射。
- 图类型/边算法可通过独立类扩展；目前没有自动插件加载或任意组合搜索器。

## 7. 主要参考（定义，不是本项目的性能证据）

- PyTorch 2.10 GRUCell：https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRUCell.html
- PyTorch 2.10 LSTMCell：https://docs.pytorch.org/docs/2.10/generated/torch.nn.LSTMCell.html
- Gilmer等，Neural Message Passing for Quantum Chemistry，ICML 2017：
  https://proceedings.mlr.press/v70/gilmer17a.html

本项目实现的是自己的轻量消息传递网络，没有复用该论文的化学数据或任务代码。
这些引用说明方法出处，不能当成我们在机器人上已经验证有效的证据。
