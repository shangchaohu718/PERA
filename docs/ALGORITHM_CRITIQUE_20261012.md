# 算法反例审查：Physics 分支到底在学什么？

> 本文保留修复前的审查记录和数值；后续实现已变化。当前修复、720 项 CPU 回归和对照探针见
> [算法修复与验证](ALGORITHM_CORRECTIONS_20261012.md)。下文行号属于审查时版本。

2026-10-12。本轮审查当前工作树，包含上一轮连续 Dynamics 更新和 schema 4 加载修复。
只新增审查脚本和本文，没有修改训练实现或已有权重。上一轮 703 项回归通过证明已覆盖接口满足
测试要求，不证明算法有效。本轮另跑 6 组 CPU 探针；没有真实 mjlab、CUDA 或正式 checkpoint 实验。

判断：当前最大问题是，状态预测、参数重建、latent 对齐、置信度门控与动作控制之间缺少经验证的
联系。它们可以同时产生“很好看”的 loss，却没有让机器人更会适应。继续增加骨干、GNN、Teacher
和额外预测头，不能填补这个证据缺口。

## 已复现的具体问题

### P1：不可测的速度被当成真实零速度，形成伪预测误差

位置：`envs/mjlab_physics.py:330`、`models/physics.py:912`、
`holomotion_physics_training.py:271` 和 `:392`。

HoloMotion 的 deployable 观测把不可测的根部线速度填成 0，下一状态监督却使用包含真实速度的
privileged 观测。随后 GRU 将当前 deployable core 减去上一帧预测，缺少按分量的“是否可观测”掩码。
这里混淆了“没有测量”和“测到了零”。

反例调用真实 `K1PhysicsBridge.read_deployable()`（底层数据为 CPU 合成状态）及真实 Dynamics：
身体匀速 1 m/s，预测器对全部 core 状态的平方误差为 **0**，下一步传给 GRU 的线速度创新却是
**[-1, 0, 0] m/s**。预测越准确，也不会使该分量的所谓预测误差归零。

这不证明策略必然失败——网络可能把它重新当成速度估计使用——但它确实破坏了当前误差反馈的
语义。应拆开观测值、有效性与监督真值：只对实际可测的分量计算 innovation；未观测的速度预测
可以作为显式估计特征，不能冒充测量误差，也不能把仿真真值塞回 Actor 来“修复”。

### P1：没有任何辨识信息，也能得到 84.6% 的门控值

位置：`holomotion_physics_runtime.py:41`、`holomotion_physics_training.py:398`。

当前门控是 `exp(-0.5 * mean(variance)) * min((age+1)/128, 1)`。
实验让所有输入 latent 完全相同，标签按归一化摩擦/质量的 Uniform(-1,1) 先验分布变化。
用当前代码的 NLL 表达式训练真实辨识头 400 次，它正确地学会了“我只能猜均值”：

| 指标 | CPU 结果 |
|---|---:|
| 标签方差 | 0.333333 |
| 均值预测 MSE | 0.333333 |
| 预测方差 | 0.333333 |
| age+1 = 1 的门控 | 0.006613 |
| age+1 = 32 的门控 | 0.211621 |
| age+1 = 128 的门控 | **0.846482** |

这是门控映射的问题，甚至不需要模型失准：对完全没有新信息的先验，公式就给出 `exp(-1/6)`。
**经过 128 步不能作为已经获得 128 步有效辨识证据的证明。** age 在此探针中被固定，以隔离门控
计算；没有假装跑过 128 步真实接触。这个数值也不是“84.6% 的预测正确率”。

单个异方差回归头估计的条件方差，也不能直接当成分布外模型知识的可靠度；两类不确定度的区别见
[Kendall 与 Gal](https://arxiv.org/abs/1703.04977)。建议先校准留出集误差和覆盖率，再测试相对先验
的信息增益、可观测创新一致性等候选门控；还需验证“门控越高，开残差越有益”，不能只看 ID MSE。

### P2：可选 KL 惩罚的梯度不是 KL 梯度

位置：`holomotion_mjlab_ppo.py:852`。相同写法来自 vendored `ppo_tf.py:2306`。

损失为 `mean(ratio.detach() * (new_log_prob - old_log_prob))`。脚本从当前源码 AST 提取并执行
这一表达式，用旧 N(0,1)、新 N(1,1) 的确定性分位点积分验证：

| 指标 | 数值 |
|---|---:|
| 当前表达式数值 | 0.499962 |
| 当前 autograd 对新均值的梯度 | **-0.000035** |
| 同一标量的数值差分梯度 | 0.999772 |
| 正确解析 KL 梯度 | **1.0** |

其期望梯度是 `E_old[ratio * grad(log pi_new)] = E_new[grad(log pi_new)] = 0`。
因此 loss 可以显示明显 KL，优化器却没有对应的恢复力。默认 `kl_coef=0`，此项当前不启用；
独立的 analytic-KL 监控和 early stop 仍存在，不能据此宣称整个 PPO 裁剪或默认训练失效。
修复应使用具有正确梯度的解析 Gaussian KL，并明确选取 KL 方向。

## 结构上的算法风险（尚未证明会降低真实机器人表现）

### Teacher 目前只是参数自编码器，没有教“如何控制”

位置：`models/physics.py:671`、`holomotion_physics_training.py:400`。

当前 HoloMotion Teacher 将两个归一化物理参数编码成 latent，再重建参数；学生匹配这个 latent。
CPU 梯度探针确认重建损失只更新 Teacher encoder/decoder。Teacher 不接收控制收益或动作蒸馏目标。
不能因为叫 Teacher，就把它理解成更强的控制策略。原版 Actor 自身还预测线速度/接触，新增 Dynamics
也预测这些量，现有证据没有说明重复表征是否比直接复用 Actor 特征更划算。

RMA 的环境编码器与控制策略通过 RL 联合学习，随后历史适应器预测该表示；这说明其表示的价值
由控制目标约束，并不意味着必须照搬 RMA。[原论文 III 节](https://arxiv.org/html/2107.04034v1)

现在不能断言 Teacher 没用，但要让它与直接监督 2 维参数/分布、完全无 Teacher、奖励训练的
privileged 策略上限做消融。当前 trainer 强制三个 Teacher/ID 权重大于 0，连干净关闭 Teacher
都不方便；应显式支持训练配方开关。

### 物理残差不知道目标，却被要求修正目标相关的动作

位置：`holomotion_physics_actor.py:136`。

残差只读取 physics_features，没有当前 reference 或基础 action。相同物理历史下，目标从加速
切到减速，残差仍相同。本地非零 residual 反例中，两行相同特征输出差小于 1e-8。
若目标所需补偿分别为 +0.2 和 -0.2，共享同一个补偿至少留下 0.04 的平均平方误差。

这是分支表达能力的反例，不是整个 Actor 的不可能性证明：基础 Actor 能看到目标。但基础 Actor
同时被 PPO 全量更新，所以收益可能来自它重新学了控制，而非 Physics 提供了可用适应信息。
建议先测试 `residual(proprio, belief, reference_or_base_action)`；初期固定基础策略测增益，再
小步解冻，记录 Physics 消融前后的任务表现，防止用全量微调掩盖弱分支。

### Dynamics 更新仍能绕过 PPO 的策略变化监控

位置：`holomotion_mjlab_runner.py:74`、`holomotion_physics_training.py:426`。

PPO 使用采样时冻结的特征；这对本次 PPO 重算是正确的。但随后 EMA 更新、历史重放使 Actor 输入
变化，整个闭环策略也随之变化。这一变化不包含在刚记录的 PPO KL 中。EMA 减小参数更新不等于保证
动作分布的 KL 小。应在相同观测历史上比较更新前后的完整动作分布，把 Actor 更新造成的 KL 与
Dynamics 更新造成的 KL 分开记录，再决定是否缩小、延后或拒绝特征更新。这是待实测的稳定性风险。

### 三个时间尺度缺少共同依据

默认门控 age 在 128 步达到满系数，第一次 Dynamics 更新需累计 64+128=192 步；门控放开时，
Dynamics 尚未完成第一次监督更新。EMA 每 192 步更新一次，decay=0.99 的参数权重半衰期约为
69 次更新，即 **13,242 仿真步**，约 414 个 32 步 PPO rollout。这个半衰期不是每回合适应延迟，
GRU 仍逐帧更新，但它说明训练中的 EMA 跟随速度很慢。

历史 preflight 只有 3 次 EMA 更新，因此初始化参数的线性混合系数仍为 `0.99^3≈97.0%`；同时
平均门控约 0.02。那组短程结果不足以归因于“已经学会物理辨识”。建议将预训练完成度、每回合观测
证据、EMA 训练步半衰期分别定义，不共用一个模糊的 warmup 概念。

### 上轮连续性修复正确，但完整重放不是免费的

位置：`algorithms/dynamics_history.py:71`、`algorithms/sequence_training.py:966`。

完整历史重放避免复用过期 hidden，保住了环境连续性；代价是随最长未结束回合增长的存储和计算。
当前 HoloMotion 每个 Dynamics epoch 都重放 prefix 加窗口，结束后还要重放 EMA。

本机默认宽度 Dynamics、batch=8、2 个 CPU 线程，每个长度预热后取 3 次中位数：

| 历史步数 | 一次 rebuild |
|---|---:|
| 32 | 0.043 s |
| 128 | 0.162 s |
| 512 | 0.651 s |

无物体 K1 帧、执行动作和起点标志的原始 tensor 为每环境每步 487 字节。
若扩到 4096 环境，1000 步约 **1.86 GiB**，容量上限 16384 步约 **30.44 GiB**，还不含 PPO
rollout、Python/tensor 对象及训练激活。这是容量计算，不是实际 GPU 占用测量，也不是当前默认 1 环境的内存。
建议先压缩/分块存储和测量完整 update 开销，减少重复编码；若采用有限历史窗口，需明确变更记忆合同，
用完整重放对照误差，不能暗中截断或重新 reset 世界。

## 如何优化代码结构

先停止扩大模型组合，保留一条主验证链。目标是让以下合同显式可测，不是机械地拆文件：

1. **观测与监督分离。** `ObservedPhysics` 携带逐量有效性；`PhysicsTargets` 只供损失；
   `DynamicsBelief` 包含估计值、方差及可观测 innovation。不可测量不能靠零值传递语义。
2. **动力学目标集中管理。** 共用一套明确定义的预测/ID/Teacher loss、掩码和指标；
   当前通用与 HoloMotion 训练器的 Teacher、NLL 和首帧掩码行为不同，不能靠相似命名保证一致。
3. **策略条件化有独立接口。** 基础 Actor、任务输入、belief、门控各自明确输入输出，允许
   no-physics、direct-ID、no-Teacher、oracle 等配方。冻结/解冻和优化器归属显式配置。
4. **更新边界形成一次可审计操作。** PPO 更新、Dynamics 更新、EMA 更新、记忆重建、完整策略
   KL 检查分别记录。保持现有 checkpoint 与 rollout 数据合同，分阶段迁移。

`models/physics.py` 现有 1416 行，混合数据类型、编码器、GRU、预测损失、目标编码、MLP 策略和 MoE
条件化；`sequence_training.py` 1216 行同时管理任务、采集、PPO 和 Dynamics。以后可按上述职责拆分。
`physics_dynamics.py` / `physics_policy_v2.py` 已是兼容别名，不应误报为重复实现。

## 下一轮实验顺序与失败标准

先修复 innovation 合同和可选 KL，再处理门控校准。所有性能比较使用同任务、同奖励、同 domain
randomization、同环境步预算与固定评测轨迹；短程筛选至少 3 个 seed，正式结论扩大 seed/轨迹覆盖。

| 对照 | 主要回答的问题 |
|---|---|
| 原版 Actor + matched task | Physics 之外的微调能做到什么？ |
| 加相同容量但无历史的 residual | 收益是否只来自更多参数/当帧信息？ |
| 历史 GRU + 直接 ID，不加 Teacher | Teacher 是否值得保留？ |
| 完整 Physics | 全部复杂度是否换来增益？ |
| 部署中去掉或打乱 latent/history | 策略是否真的依赖适应信息？ |
| 训练期可见真值的 oracle 条件策略 | 知道这些物理参数本身是否改善控制？ |

打乱潜变量可能制造分布外输入，所以它只能作为依赖性诊断，还要结合“重新训练一个无该分支的对照”
判断因果收益。先比较动力学与恒等预测、无 action、无 history 基线：20 ms 单步预测变好不代表学到了
有效动力学。再比较跟踪误差、存活/成功率、左右脚滑移、冲击、动作平滑和参数突变后的恢复时间。
报告最差域与多 seed 分布、吞吐量和显存；不能仅凭总 reward 或训练 loss 宣布改进。

若 oracle 都不优于 matched base，先检查任务是否需要适应；若 direct-ID 不差于 Teacher，删掉
Teacher；若去掉历史重训后不差，撤回“历史辨识带来收益”的主张；若只有训练域好而留出域差，不能称为
泛化适应。PPO 的裁剪目标本身也不能代替这些实验。[PPO 原论文](https://arxiv.org/abs/1707.06347)

## 复现和证据

```text
python tools/audit_physics_algorithm.py --output runs/local_validation/algorithm_audit_20261012/evidence.json
```

本机解释器：`runs/local_validation/dev_env/Scripts/python.exe`。
Python 3.12.13，Torch 2.10.0+cpu，随机种子 20261012。JSON 保存六组探针和时序计算。
审查时公式探针直接提取源码表达式；网络探针使用真实模块及合成张量，不能替代机器人动力学验证。
原始 JSON 已保留在上述目录。当前探针改为调用修复后的共用训练函数并比较无条件/条件残差，
再次运行不会复现本文旧缺陷数值；请用后续修复报告中的独立输出目录。
