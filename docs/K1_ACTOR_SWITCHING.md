# 同一 K1 任务下切换 HoloMotion 与自定义 Actor

现在使用同一个训练入口 `tools/train_k1_mjlab.py`、同一个冻结评测入口
`tools/evaluate_k1_mjlab.py`。Actor 可选 HoloMotion 和六种自定义网络，Physics 独立开启。
本轮发布 CPU 回归 **752 passed、0 failed、18 CUDA skipped，445 subtests passed**。

## 选择网络和物理模块

| 参数 | 实际创建的网络 |
|---|---|
| `--actor holomotion` | 原版 HoloMotion `ReferenceRoutedGroupedMoETransformerPolicy`，从原始 run 加载 |
| `--actor mlp` | MLP，共用本项目 MLP 构造器 |
| `--actor transformer` | 已有 `DenseReferenceTransformer`；`dense` 为同义选项 |
| `--actor moe` | 已有 `ReferenceConditionedTransformer`，含专家路由与负载均衡辅助损失 |
| `--actor gru` | 已有 `RecurrentReferenceBackbone` + GRU |
| `--actor lstm` | 已有 `RecurrentReferenceBackbone` + LSTM |
| `--actor lru` | 已有 `RecurrentReferenceBackbone` + `LRUCore` |

不加 Physics 选项即训练基础 Actor；`--physics` 加直接参数辨识与动作修正；
`--physics-teacher` 加可选 Teacher 辅助目标。开启 Physics 时，
`--physics-scene-encoder set` 或 `gnn` 选择物理空间编码。K1 的两个脚部接触槽位均映射到静态地面。

所有 Actor 在这个入口中共用 **506 维 Actor 观测、843 维 Critic 观测、22 维动作、同一 mjlab 任务、
同一 PPO 和同一种动作残差 Physics 接法**。相同的 `--physics` / `--matched-physics-task` 设置下，
奖励、物理随机化和监督保持一致。切换网络不会自动切换到不同的诊断任务。

## 新建训练

以下资产、参考动作、ONNX 和源 run 路径均需替换为自己的真实文件。mjlab 入口要求 CUDA。

HoloMotion：

```text
python tools/train_k1_mjlab.py --actor holomotion --run-dir runs/original_holomotion --checkpoint-name model_9999 --onnx exports/model_9999.onnx --asset-root assets/K1 --motion motions/reference.npz --output-dir runs/k1_holomotion --iterations 100 --physics
```

自定义 LRU：

```text
python tools/train_k1_mjlab.py --actor lru --actor-config configs/actors/k1_small.yaml --onnx exports/model_9999.onnx --asset-root assets/K1 --motion motions/reference.npz --output-dir runs/k1_lru --iterations 100 --physics
```

把第二条的 `--actor lru` 改为 `mlp`、`transformer`、`moe`、`gru`、`lstm` 即切换其他自定义主干；
同时更换输出目录。删除 `--physics` 可训练对应基础 Actor；添加 `--matched-physics-task` 可匹配
Physics 实验的物理随机化。所有后端始终使用原版 HoloMotion v1.4 奖励，不附加脚滑或冲击奖励。
`--physics --physics-scene-encoder gnn` 使用 GNN 物理编码。

自定义 Actor 从零初始化，默认参与训练；HoloMotion 从预训练权重初始化，Physics 实验默认冻结
基础 Actor，`--finetune-base-actor` 可开启微调。对比结果时应明确初始化、冻结策略和训练步数。
不会默认冻结随机初始化的自定义 Actor。

ONNX 在这里提供关节顺序、PD、默认关节位置、动作缩放和观测合同。新自定义策略不用它生成动作。
也可以用 `--task-contract path/to/contract.json` 替代 `--onnx`；JSON 必须包含
`joint_names`、`joint_stiffness`、`joint_damping`、`default_joint_pos`、`action_scale` 五个长度22的列表，
以及可选的 `obs_dim=506`、`future_frames=10`，数值应来自实际任务合同，不能随意编造。

## 恢复和冻结评测

每种 Actor，无论是否开启 Physics，都会保存能独立加载的完整 `.pt`。恢复时自动重建原架构，
恢复权重、优化器、PPO 配置；Physics 实验同时恢复 online/EMA Dynamics、校准和发布配置。
从 checkpoint 迭代的下一次开始训练，从新的真实回合开始采集。

```text
python tools/train_k1_mjlab.py --resume runs/k1_lru/model_99.pt --asset-root assets/K1 --motion motions/reference.npz --output-dir runs/k1_lru_continue --iterations 20

python tools/evaluate_k1_mjlab.py --checkpoint runs/k1_lru/model_99.pt --asset-root assets/K1 --motion motions/reference.npz --output runs/k1_lru_eval.json --steps 500 --num-envs 2
```

新保存的真实任务 checkpoint 已含任务合同，因此恢复/评测通常无需再次提供 ONNX。
`--resume` 自动恢复 Actor 类型、Physics 开关、rollout 长度及训练配置，禁止同时传入覆盖这些设置的
选项。可修改新增训练迭代数、输出目录、环境数量、设备、日志频率和历史内存预算。
若额外写 `--actor`，它必须与 checkpoint 相同；更换架构应建立新实验。

Physics bundle 的评测可加 `--ablation no_physics/no_latent/shuffle_latent/no_history`；
普通 Actor bundle 只接受 `full`。冻结评测核对参数和持久缓冲区指纹，禁止更新权重与校准。

旧命令 `train_holomotion_k1_mjlab.py` 和 `evaluate_physics_holomotion_k1_mjlab.py` 仍可使用。
历史 Physics schema 4/5 继续按既有兼容规则加载；schema 4 仍为冻结评测用途。

## 和原来的多网络接口有什么关系

`api.build_policy()`、`experiment.py`、原有 `diagnostic_*` / 接触任务 YAML 继续保留，
它们使用 `PhysicsObservation + ControlGoal`，支持原生物理特征融合与 MoE 物理路由。

新统一 K1 入口增加一个观测适配器：完整读取 HoloMotion 当前106维与未来10×40维参考信息，
再交给已有 Transformer/MoE/GRU/LSTM/LRU 主干或 MLP。这样所有网络能在同一个任务输入合同下比较。
没有把 HoloMotion 没提供的关节参考速度伪造成 `ControlGoal` 的真实输入。

统一入口的 Physics 采用相同的动作残差接法。原通用接口的物理 token/路由融合仍属于原来的
`experiment.py` 路线；这两种物理融合方式不能被当成相同实验。
新输入适配器的参数结构与原 `ControlGoal` Actor 不同，旧通用 Actor checkpoint 不直接兼容新入口。
这里复用的是已实现的核心网络，不是把不同输入合同的权重直接互换。

统一入口所有 Actor 的记忆在 rollout 开头清空，PPO 从同一边界重算，冻结评测使用保存的同一节奏；
回合结束时只清理对应环境行。Dynamics 物理历史独立保留，权重更新通过真实历史重建，不重置环境。
原通用接口中的循环 Actor 完整历史策略仍按其原有合同运行。

## 代码与验证

| 文件 | 职责 |
|---|---|
| `unified_motion/k1_policy_backends.py` | 严格网络配置、506维输入适配、七类网络加载分发 |
| `unified_motion/models/physics_policy.py` | 提取共用主干工厂；旧接口与新接口复用同一实现 |
| `unified_motion/k1_training.py` | 基础/Physics 训练器构造、自定义 Actor 默认可训练 |
| `unified_motion/k1_policy_runtime.py` | 完整加载、恢复、基础/Physics 冻结评测 |
| `tools/train_k1_mjlab.py` / `tools/evaluate_k1_mjlab.py` | 统一用户入口 |
| `tests/test_k1_actor_switching.py` | 七类 Actor × Physics 开关的完整生命周期测试 |

```text
python -m pytest tests/test_k1_actor_switching.py -q
python tools/evaluate.py --output runs/local_validation/k1_actor_switching_final
```

本轮新增32项测试：14种组合在异步回合边界下采样、核对旧策略均值/log-prob、PPO 更新、
Dynamics 连续更新、独立保存加载、恢复后再次训练、冻结动作一致；另外覆盖全部506维输入梯度、
Critic 特权隔离、实际主干类型、错误架构拒绝以及 CLI 创建/恢复（显式模拟硬件/任务，含 GNN）。
发布检查还覆盖了随仓库提供的 Actor YAML 在六种自定义主干上的真实构造与前向计算。
Actor 专用配置位于 `configs/actors/`，不与根目录的完整 `experiment.py` 任务配置混用。

这证明统一代码链路的 CPU 正确性，不代表正式宽度模型、CUDA 性能或机器人运动效果已验证。
18项 CUDA 测试因当前环境缺少 CUDA/mjlab/MuJoCo/Warp 而跳过。
