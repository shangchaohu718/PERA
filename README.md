# Unified Motion R21 · LRU与无MoE Transformer

本版本在用户提供的R20工程上继续修改，**不另起项目、不重复写PPO或动力学模型**。
新增LRU（Linear Recurrent Unit）Actor插口；`transformer`是原`dense`的易读别名，真实结构没有MoE/Router。

## 当前开发状态（2026-10-13）

**2026-10-13 更新：[Transformer 物理教师与 DeepSeek／GLM 训练顾问](docs/PRIVILEGED_TEACHER_AND_LLM.md)**。
新增训练专用因果 Transformer 教师、完整保存恢复与异步 LLM 调参建议。本轮进一步修正：
物理门控保留小幅反馈；教师通过留出质量检查后才蒸馏状态／接触预测；接通四步预测监督；
LLM 保留时间趋势和分动作证据，并调整真实优化器学习率。部署仍运行学生与可选 Actor。
进一步修正随机 Actor 默认冻结、教师重力方向约束、动作敏感性误拒绝，以及 LLM 候选局部改善掩盖总体退步的问题。
本次本地回归 **843 项通过、0 失败、18 项跳过，另有 450 个子测试通过**；
见 [修正验证摘要](reports/logic_fixes_20261013.json) 和
[三种子合成学习报告](reports/logic_fixes_learning_20261013.json)。合成数据上的改善不代表机器人控制收益；
真实 CUDA/mjlab 训练与 API 调用尚未验证。上一阶段报告保留在 `reports/teacher_llm_validation_20261013.json`。

**先读 [当前算法结构与物理反馈研究路线](docs/ARCHITECTURE_AND_RESEARCH_20261012.md)**：
包含运行与训练结构图、Actor/Physics 的分工、信息泄露边界、本轮改动，以及太空步、滑板、舞蹈
和 sim2real 榜单所需的后续验证。文档明确区分已实现功能与尚未验证的研究设想。

项目已经完成原版 HoloMotion K1 Actor、Critic、MoE Transformer 和 PPO 到 mjlab
任务接口的第一阶段迁移，并接入了可选的 Physics residual、DynamicsModel 与
privileged Teacher。仓库此前记录 **665 项测试通过（另有 18 项按环境跳过）**，也完成了
真实 mjlab/CUDA 的短程冒烟；这些结果只证明代码链路可执行，**尚不代表已经具备正式
长训练条件或机器人步态已经收敛**。

本地正确性审查及结构调整见 `docs/CORRECTNESS_REVIEW_20261012.md`。
本轮已修复 HoloMotion 的 rollout 缓存边界、timeout bootstrap、单样本优势归一化，
并统一资产清单的跨平台路径格式；本轮验证使用 CPU，不替代历史 CUDA/真实仿真验收。
后续已修复 Dynamics 更新的环境连续性，补齐完整 Physics 策略加载和冻结评测。
实现与使用方式见 [Physics 运行时说明](docs/PHYSICS_RUNTIME_AND_CONTINUITY.md)。
随后已修复缺失测量反馈、KL 梯度、残差目标条件、校准门控与 Dynamics 发布约束，
默认直接 ID、Teacher 显式可选；现仅默认冻结已加载的预训练 HoloMotion Actor，随机初始化 Actor 默认参与训练。此前修复见 [算法修复和实验证据](docs/ALGORITHM_CORRECTIONS_20261012.md)。
2026-10-12 发布回归为 **752 项通过、0 失败、18 项 CUDA 跳过**，另有 445 个子测试通过。
已补齐 Windows UTF-8 报告读取和独立 Actor YAML 校验，见 [发布验证摘要](reports/local_validation_20261012.json)。

HoloMotion 和六种自定义 Actor 现可在同一个 K1 训练入口切换：
`tools/train_k1_mjlab.py --actor holomotion/mlp/transformer/moe/gru/lstm/lru`（选择其中一个值），
加 `--physics` 开启统一物理残差，`--resume` 完整恢复，`tools/evaluate_k1_mjlab.py` 冻结评测。
完整命令、配置、原接口与新输入适配器的区别见 [Actor 切换指南](docs/K1_ACTOR_SWITCHING.md)。

### 已完成

- HoloMotion K1 的 506 维 Actor、843 维 Critic 观测合同和 22 关节映射。
- 原版 HoloMotion Actor/Critic 权重严格加载，以及 mjlab 上的 PPO 采样、更新和断点保存。
- 左右脚独立物理特征、8 维学生物理 latent、privileged Teacher、预测不确定度和置信度门控。
- 固定物理量归一化、实际执行动作监督、Dynamics 多 epoch 更新、学习率调度和 MoE 物理路由接口。
- 可复现的 CPU 回归测试、GPU 冒烟记录和预检报告：
  `reports/holomotion_physics_preflight_20261012.md`。
- 官方 HoloMotion commit `cf2018a282249a04df1db4fa639760884f97b41b` 中本项目实际使用的
  源码与配置已固化到 `third_party/HoloMotion/source/`，不再依赖某台机器上的相邻源码目录。

### 正式训练前的验证门槛

1. **Dynamics 更新连续性已修复并通过 CPU 回归。** 两条采样路径保留环境状态，
   用当前回合的真实观测和执行动作，在新 Dynamics/EMA 权重下重算记忆和预测反馈。
   历史容量不足会在推进环境前报错，不会隐式截断或 reset；真实 CUDA 仿真仍需复验。
2. **复验已修复的 Transformer rollout 边界。** 两条 HoloMotion 采样路径现已在每个
   rollout 开头清空 Actor cache；缩小宽度的原版网络通过连续采样/PPO 重算测试。
   正式 checkpoint 和 CUDA 仿真仍需复验；未来若保留长历史，必须保存并重放真实历史前缀。
3. **复验已修复的 timeout bootstrap。** 两条 HoloMotion 路径已保留 reset 前的
   timeout Critic 值，共用 GAE；timeout 中断优势链但允许 bootstrap，真正终止关闭
   bootstrap。CPU 混合终止/截断用例通过，实际 mjlab timeout 仍需验收。
4. **完整 Physics 加载和冻结评测已补齐。** 新 schema 5 `.pt` 独立包含完整配置、
   基础 Actor/Critic、residual、online/EMA Dynamics、校准和优化器；冻结推理使用 EMA。
   schema 4 保留原算法语义用于冻结评测，新算法训练恢复要求 schema 5。
   保存加载一致性、权重冻结和错误合同拒绝均通过 CPU 回归。正式权重需用
   `tools/evaluate_physics_holomotion_k1_mjlab.py` 验收；普通 Actor safetensors 不含完整物理分支。
5. **建立零训练基线。** 修完上述训练语义后，先在当前 mjlab 中冻结评测原始 HoloMotion
   checkpoint，确认基础策略表现，再比较 Physics residual 和 Teacher，最后才扩大训练轮次。

在真实仿真复验及零训练基线完成之前，不建议启动正式长训练。`runs/`、模型权重和本地数据集均被
`.gitignore` 排除，不随源码仓库发布。

## 1. 原通用入口的网络组合

本节对应 `api.build_policy()` / `experiment.py` 的原生特征融合路线。
统一 `train_k1_mjlab.py` 入口采用七选一 Actor 加可选动作残差，见上方结构说明与切换指南。

```text
身体/接触/物体 → set或GNN空间编码 → 独立动力学GRU及预测误差反馈
                                        ↓ context / latent
任务目标 → MLP / Transformer / MoE / GRU / LSTM / LRU 六选一 → 原始动作
```

GNN是独立开关。选择Transformer不构造MoE；选择LRU不构造时间Transformer/MoE。
LRU仍复用当前目标交叉注意力输入，不是说整个Actor完全没有任何Attention。
独立DynamicsModel没有改成LRU，它继续使用自己的GRU记忆。这保证主干对照只改变Actor时间网络。

## 2. 直接运行已经检查的CPU代码链路

需要已有相容PyTorch和NumPy。不要为了安装项目覆盖GPU机的CUDA PyTorch。

```bash
cd unified_motion_release21
python -m pip install PyYAML packaging pytest safetensors loguru "tensordict>=0.11,<0.13"
python -m pip install --no-build-isolation --no-deps -e .

# 查看真实网络参数和MoE层数，不运行物理。
python -m unified_motion.experiment --config configs/diagnostic_lru.yaml --mode inspect
python -m unified_motion.experiment --config configs/diagnostic_transformer.yaml --mode inspect

# LRU：采样、PPO、独立动力学更新、权重保存。
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m unified_motion.experiment --config configs/diagnostic_lru.yaml --mode train

# 冻结权重评测（允许GRU/LRU记忆随新观测更新）。
python -m unified_motion.experiment --config configs/diagnostic_lru.yaml --mode evaluate \
  --checkpoint runs/diagnostic_lru/checkpoint.pt

# 完整回归：包含以前全部测试与LRU的独立公式/梯度/接线测试。
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python tools/evaluate.py

# 可复现全部12种组合的短程训练、保存加载、冻结检查。
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python tools/check_plugins.py
```

`diagnostic`明确是解析质点场景，不是K1或MuJoCo。程序跑通不等于机器人掌握技能。
包内`runs/`只有诊断权重，不应加载到实体机器人。

## 3. YAML与关键代码

| 位置 | 作用 |
|---|---|
| `configs/diagnostic_lru.yaml` | LRU Actor，不使用GNN |
| `configs/diagnostic_gnn_lru.yaml` | GNN物理编码 + LRU Actor |
| `configs/diagnostic_transformer.yaml` | 无MoE的Transformer，旧名dense继续兼容 |
| `configs/diagnostic_gnn_transformer.yaml` | GNN物理编码 + 无MoE Transformer |
| `configs/k1_lru.yaml`, `configs/k1_transformer.yaml` | 真实K1候选配置；现有mjlab 1.3已做兼容性冒烟，目标mjlab 1.6锁版和长训练仍待验收 |
| `models/lru.py` | 可独立使用的LRULayer、LRUCore、LRUState |
| `models/recurrent.py` | 复用原缓存与参考融合，统一接入GRU/LSTM/LRU |
| `models/physics_policy.py` | 原Actor接线，新增LRU分支和Transformer别名 |
| `api.py`, `experiment.py` | 工厂、YAML校验、参数检查、训练与冻结入口 |

所有代码仍在`unified_motion/`包中。配置文件中的LRU初始化范围、复数状态宽度、扫描方式都有对应真实计算，不是未使用的装饰项。
详细说明与独立用法见 **`docs/LRU_AND_TRANSFORMER.md`**。

## 4. 本轮测评证据

原R21交付回归为 **607项通过、18项CUDA跳过、0失败（共625项）**；本地兼容修复后为
**608项通过、18项CUDA跳过、427个子测试通过、0失败**。原始交付证据仍见
`reports/current/delivery.json`，本机新增物理实测见 `reports/current/local_mjlab_compatibility_smoke.json`。

- 完整回归包含旧套件、新LRU解析公式、并行/顺序梯度、有限差分、padding/reset、旧概率重放、旧checkpoint兼容、纯Transformer结构检查。
- 12种组合分别实际完成3轮CPU解析训练，执行独立动力学更新、保存、加载及冻结评测。
- 短程诊断任务成功率全部为0，不包装成“机器人已学会”或“算法无效”的结论。
- 旧R20包内10份模型和动力学checkpoint严格加载通过；这不等于逐位恢复真实仿真时钟。
- 本机现有 mjlab 1.3 + CUDA 已跑通 K1 物理桥和一次最小训练更新；它不满足项目声明的
  mjlab 1.6 锁版。真实K1长训练、反馈消融、吞吐、控制精度及G1仍没有通过结论。
- 不宣称LRU一定更快：它有可反传配对扫描，但还没有fused CUDA内核；长回合PPO历史重放仍有成本。

重要报告：

```text
reports/current/evaluation.json
reports/current/cpu_regression.txt
reports/current/clean_extract_evaluation.json
reports/current/clean_extract_pytest.txt
reports/current/plugin_matrix.json
reports/current/r20_checkpoint_compatibility.json
reports/current/delivery.json
```

`reports/r20_history/`是明确归档的旧报告，不能拿旧数字替代本轮执行。

## 5. 兼容与使用边界

- 旧`dense`保持合法，`transformer`创建同一实现，不存在两份独立纯Transformer代码。
- 所谓“纯”指Actor主干没有MoE/Router；目标输入、独立DynamicsModel和Critic仍保留。
- GRU/LSTM/LRU必须重建权重更新后的记忆；LRU并行扫描不意味着可以丢掉历史。
- `lru_state_dim`是复数宽度，每个环境的Actor状态占`2*层数*宽度`个实数。
- 旧MoE在有历史前缀时`balance_coef=0`的限制没有在本轮偷偷取消。
- 真实移动平台、滑板机构、自主接触规划、G1物理场景和MoCap/交互混合调度仍非已完成项。

R21完成的是本轮要求的网络选项与训练接口，不是对整个通用零样本项目的最终验收。

## 6. 官方 HoloMotion K1 → mjlab

新增的 `unified_motion.holomotion_mjlab` 不再近似重写 HoloMotion Actor，而是读取
官方 K1 导出 ONNX 的元数据合同，构造其真实 506 维观测、维护 22 关节名字映射，
并为 mjlab 生成与训练时一致的 PD/action-scale 配置。训练时使用的原版网络源码位于
`third_party/HoloMotion/source/holomotion/`。真实评测入口是：

```bash
python tools/evaluate_holomotion_k1_mjlab.py --help
```

参考轨迹转换、四元数顺序、短程 GPU 预检以及当前正式训练门禁见
`docs/HOLOMOTION_MJLAB.md` 和本 README 顶部的“当前开发状态”。模型与数据因体积及
许可边界不进入 Git，需要由运行者通过命令行参数提供。
