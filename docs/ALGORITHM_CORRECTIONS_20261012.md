# Physics 算法修复与本地验证

本文保留算法修复阶段的结果。后续统一 K1 入口已支持七种 Actor，累计回归为 751 项通过；
HoloMotion 加 Physics 默认冻结基础 Actor，自定义 Actor 从零初始化并参与训练。
最新使用方式见 [Actor 切换指南](K1_ACTOR_SWITCHING.md)，整体结构与研究范围见
[当前算法结构与物理反馈研究路线](ARCHITECTURE_AND_RESEARCH_20261012.md)。

本轮落实 [算法反例审查](ALGORITHM_CRITIQUE_20261012.md) 中可由实现和 CPU 实验验证的问题。
最终回归：**720 passed、0 failed、18 CUDA skipped，442 subtests passed**。
Python 3.12.13、Torch 2.10.0+cpu。真实 mjlab/MuJoCo/Warp、正式权重与机器人运动收益未验证。

## 已修复的算法合同

| 原问题 | 当前行为 | 验证 |
|---|---|---|
| 缺失根线速度被当成零测量，完美预测仍产生伪误差 | `PhysicsObservation.core_valid` 显式标记可测分量；编码器、单步/多步预测输入和创新反馈屏蔽缺失值，预测损失按目标有效性计算。可测性有独立编码，不把仿真真值送入 Actor | 完美预测的速度创新由 −1 变为 0；缺失速度占位值从 0 改成 100，latent 和预测不变；mjlab 与通用交互入口均覆盖 |
| 只有先验、无辨识信息也能打开 84.6% 门控 | 用未参与该次梯度损失的时间窗口估计 MSE、方差温度与覆盖率；误差必须优于配置先验，且证据量足够，才能打开门控 | 无信息/未校准/虚假低方差均得到 0；有辨识能力的合成模型能打开门控 |
| 可选 PPO KL 损失的梯度近乎为零 | 改成可微的对角高斯 `KL(old || new)`，旧分布停止梯度 | 均值偏移 1、方差 1 的反例：KL=0.5，梯度=1，与有限差分一致 |
| residual 只看物理状态，无法区分同状态的相反目标 | 增加基础 Actor 动作均值作为条件，保留零初始化和动作限幅 | 相同状态、相反指令，3 个随机种子下无条件 residual 的 MSE≈0.04，条件 residual 为 4.24e−9、5.45e−11、4.03e−10 |
| Teacher 参数自编码被误当成控制教师；两条训练器辅助目标不一致 | 共用 `identification_loss` 的 NLL/MSE、掩码和可选 Teacher 辅助目标；默认直接 ID，Teacher 显式启用 | 无 Teacher 的 PPO→Dynamics→保存→冻结加载闭环通过；Teacher 停止梯度与重建合同仍通过 |
| EMA/门控变化绕过 PPO 的策略变化监控 | 暂存候选 EMA 与校准，在相同当前 Actor 下比较更新前后完整动作分布的 KL，超阈值拒绝发布 | 人为制造超限候选：部署模型、校准、Actor、环境与回合年龄均不变；online Dynamics 仍可继续学习 |
| EMA 的更新次数时钟与回合证据年龄混用 | EMA 默认以采样控制步计半衰期 1024；门控独立读取回合年龄及校准资格 | 不同更新分块得到相同累计衰减；年龄足够本身不能开启未校准门控 |
| 完整历史重放重复编码、长期占用设备内存 | 历史默认保存在 CPU，64 帧批量编码，按原顺序逐帧重建递归与预测反馈；有步数和字节预算 | 异步 reset、多 epoch、权重更新后与独立逐帧重放一致；容量不足在环境推进前报错 |

基础 Actor 在 Physics 训练中默认冻结，只优化 residual 与 Critic；`--finetune-base-actor`
显式开启基础 Actor 微调。真实 PPO 测试确认默认情况下基础 Actor 指纹不变，residual 确实更新。
已有 Dynamics 更新不 reset 世界、timeout bootstrap、rollout cache 边界修复继续通过完整回归。

## 结构调整

- `physics_data.py`：统一观测的堆叠、连接、切片与设备迁移，新增可测性字段不再被手写字段列表遗漏。
- `algorithms/dynamics_objective.py`：共用 ID 与可选 Teacher 目标，不改变 Teacher 是参数自编码器的事实。
- `physics_confidence.py`：独立、可保存、冻结评测时只读的校准与门控。
- `physics_update.py`：步数半衰期、保留校准比例、完整策略 KL 阈值和基础 Actor 冻结配置。
- `algorithms/dynamics_history.py`：真实历史保留、内存预算与精确重放，两条采样路径共用。
- `holomotion_physics_runtime.py`：完整策略重建、训练/推理共用特征计算、冻结评测与消融入口。

## 默认训练与发布行为

默认累计 192 个控制步，前 64 步用于 burn-in，接下来的 96 步用于梯度训练，最后 32 步用于
候选模型校准。若累计窗口超过最低长度，使用所有已收集的步数，并按 burn-in 后的 25% 保留
末尾校准窗口；没有静默丢弃最后一个 rollout。污染保留窗口的参数标签不会改变 online 模型的
优化结果，但会改变候选校准统计，这一隔离已有回归测试。

门控默认要求至少 16 个保留样本、每个辨识维度相对于先验的 MSE 改善至少 5%。
归一化 Uniform(−1,1) 的先验方差默认为 1/3；其他任务的参数分布应显式配置
`PhysicsConfidenceConfig.prior_variance`。实际门控同时受经验误差下限、预测方差、辨识改善和回合
年龄约束。它是经验权重，**不是安全概率或分布外检测器**。同回合相邻样本相关，16 个样本
不等于 16 个独立试验，时间保留窗口也不能替代跨 seed、跨轨迹、跨物理域的独立评测。

发布阈值默认完整动作分布 KL 的均值 ≤0.01、单个时刻最大值 ≤0.1。比较两侧使用相同的
当前 PPO 后 Actor、相同观测和相同 rollout 注意力边界，单独衡量此次 Dynamics/校准发布的变化。
通过后才一起替换 EMA 与校准；拒绝时保持部署分支，保留 online 学习结果。
`deployment_update_accepted`、`deployment_mean_kl`、`deployment_max_kl` 记录结果。
`calibration_samples`、`calibration_skill` 描述的是候选校准，拒绝时不能当成已部署状态。
这个检查只约束已收集状态上的经验 KL，不能保证所有未来状态的策略变化都受限。

## 历史预算和速度

默认上限为 16384 步、512 MiB **活跃历史原始张量**。HoloMotion CLI 可用
`--physics-history-mib` 调整后者。容量不够明确报错，保留世界；不会改成重置或近似截断。
该预算不包括尚被 rollout 前缀引用的旧快照、Python/tensor 对象、训练激活、模型或其他内存，
不是进程总内存上限。

batch=8、CPU 两线程、默认宽度、每个长度重放 3 次取中位数：

| 历史步数 | 审查时逐帧编码 | 当前分块编码 |
|---|---:|---:|
| 32 | 0.043 s | 0.031 s |
| 128 | 0.162 s | 0.125 s |
| 512 | 0.651 s | 0.473 s |

这是本机观测值，不是 GPU 吞吐保证；递归部分仍随历史长度线性增长。新增布尔掩码后，探针中
每环境每步原始历史为 540 字节。4096 环境×1000 步约 2.06 GiB，超过默认活跃历史预算，
实际大批量任务必须据最长回合和整机内存另行设置，并测量完整更新开销。

## 使用与格式兼容

训练入口仍是 `tools/train_holomotion_k1_mjlab.py`，必须提供既有基础策略、ONNX 合同、资产和动作数据。

| 对照 | 加到同一套训练参数的选项 |
|---|---|
| 基础 Actor 微调，匹配奖励和物理随机化 | `--matched-physics-task` |
| 默认 Physics：直接 ID、基础 Actor 冻结 | `--physics` |
| 加参数自编码 Teacher 辅助目标 | `--physics-teacher` |
| Physics 同时微调基础 Actor | `--physics --finetune-base-actor` |
| 关闭门控/完全打开门控的明确消融 | `--physics --physics-gate off` / `--physics --physics-gate ungated` |

默认 half-life 可用 `--physics-ema-half-life-steps` 调整；显式给出 `--physics-ema-decay`
会覆盖它，恢复“每次更新一个固定衰减”的计算方式。不要把两者当成同时生效。

新 `.pt` 为 **schema 5**，包含基础 Actor/Critic、条件 residual、online/EMA Dynamics、校准配置与
统计、更新配置和优化器。完整恢复用 `HoloMotionPhysicsMJLabPPO.from_checkpoint(path, task)`；
新进程从新真实回合开始，不恢复运行中的世界/RNG 状态。

schema 4 保留原架构、原门控和原创新语义，允许冻结评测；不会静默套用新算法。
实测上一轮真实保存的 `blockers_smoke/complete_physics.pt`，3 环境 15 步动作 SHA256 仍为
`55615ef44fce4d2d4b4e66e65dc7e23ca13a1a43aa06fecb7d4a2f7f503a924b`，与修复前完全一致。
schema 4 不直接恢复为新训练器，新算法应从基础策略建立 schema 5 新实验。
schema 3 缺少重建配置，只能通过原版本训练器和确切原配置先迁移为 schema 4；当前训练器
不会猜测缺失配置，也不会把旧优化器状态装进新架构。

冻结评测示例（需真实 CUDA 环境）：

```text
python tools/evaluate_physics_holomotion_k1_mjlab.py --checkpoint model_5.pt --asset-root assets/K1 --motion motions/reference.npz --output runs/physics_eval.json --steps 500 --num-envs 2 --seed 42 --ablation full
```

`--ablation` 支持 `full`、`no_physics`、`no_latent`、`shuffle_latent`、`no_history`；
打乱 latent 要求至少两个环境，按环境维度循环移位。`no_history` 只清空 Dynamics 的历史，
基础 Actor 的历史合同保持训练设置。所有模式禁止优化权重/校准，并检查前后指纹。
这些推理消融用于依赖性诊断，不等价于重新训练容量匹配的无历史/无 Physics 对照。

## 本地证据与仍需完成的科学验证

```text
python tools/evaluate.py --output runs/local_validation/algorithm_corrections_final
python tools/audit_physics_algorithm.py --output runs/local_validation/algorithm_corrections_final/probes.json
```

本机解释器为 `runs/local_validation/dev_env/Scripts/python.exe`。证据位于上述目录的
`evaluation.json`、`probes.json`、`legacy_checkpoint.json`，均按既有规则被 Git 忽略。
核心回归包括：新 schema 5 活跃门控完整保存加载、旧架构加载、校准缺失/损坏拒绝、
冻结不读取 Teacher/特权目标、留出标签隔离、候选更新拒绝、无 Teacher 完整训练和环境连续性。

本轮解决实现错误、表达能力缺口和更新合同；没有证明参数 Teacher 有控制价值，也没有证明
Physics 能改善真实步态。正式收益仍需同奖励、同随机化、同环境步预算、多 seed 的冻结评测，
比较原始/微调基础策略、直接 ID、Teacher、重新训练的无历史对照与可见真值 oracle。
应报告跟踪误差、存活/成功率、滑移、冲击、动作平滑、参数突变恢复时间、最差物理域和资源开销。
没有这些结果，就不能用预测 loss、此次合成任务 MSE 或测试数量代替“机器人适应能力提升”。
