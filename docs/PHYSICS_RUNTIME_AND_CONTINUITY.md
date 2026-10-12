# Dynamics 连续更新与完整 Physics 运行时

2026-10-12：两项代码阻塞及后续算法问题已修复。最新 CPU 回归为 **720 passed、0 failed、18 CUDA skipped**，
另有 **442 subtests passed**。本次没有运行真实 mjlab/MuJoCo/Warp 仿真，也没有验证正式权重的运动技能。
新门控、目标条件残差、默认冻结和发布 KL 合同见 [算法修复报告](ALGORITHM_CORRECTIONS_20261012.md)。

## 更新 Dynamics 时保留环境

`algorithms/dynamics_history.py` 集中维护真实观测、episode_start 和执行动作的不可变快照。
Unified collector 和 HoloMotion Physics trainer 共用此历史。更新后保持环境位置、回合计数、
当前观测和年龄，只在新 online/EMA 权重下重算隐状态与预测误差反馈；Unified 同时重建 Actor 前缀。
多 epoch 的 Dynamics 训练也逐次重算历史输入，避免第二个 epoch 沿用旧权重产生的 hidden 或预测。

保留历史从每个仍在运行的回合的真实起点开始。异步结束时，只删除所有行最近真实 reset 之前的
公共前缀；不会用固定长度近似 burn-in 代替完整历史。构造参数 `dynamics_history_steps=16384`
限制保留步数，`dynamics_history_bytes=512*1024**2` 限制活跃历史原始张量字节数。默认 CPU 存储、
64帧批量编码后精确递归重放。若单次追加超出容量，会在推进环境之前显式报错；应按任务最长回合
和内存预算设置容量。字节预算不含 rollout 保留的旧快照、对象与激活，不是进程总内存上限。
重放成本仍随保留历史增长。显式 `reset_all()` 和任务真实终止/截断仍可正常重置。

Dynamics 监督与反馈使用任务提供的 `last_executed_action`；未提供时按任务直接执行采样动作的合同处理。
PPO 的动作和 log-prob 始终对应采样命令，不会被 Dynamics 的执行动作替换。

## 完整 checkpoint

新 Physics 保存使用 schema 5。单个 `.pt` 包含已解析的原版网络配置、基础 Actor/Critic、
Physics residual、online Dynamics、EMA Dynamics、校准配置和统计、更新配置、训练配置和优化器状态；真实任务还保存关节顺序、
PD 与观测合同、奖励和随机化配置。原有 Actor/Critic safetensors 输出仍保留，但基础 Actor 文件
不能单独代表完整 Physics 策略。缺少网络配置时保存立即失败。

`holomotion_physics_runtime.load_physics_checkpoint()` 严格校验分支、配置、权重形状及有限值，
创建完整模型，供冻结评测和训练恢复共用。冻结推理只使用 EMA Dynamics，保留 residual、置信度门控
和训练时 rollout 长度对应的 Actor cache 清理节奏。

```python
from unified_motion.holomotion_physics_runtime import (
    FrozenPhysicsPolicy, evaluate_physics_policy,
)

policy = FrozenPhysicsPolicy.from_checkpoint("model_4.pt", device="cuda:0")
report = evaluate_physics_policy(policy, task, steps=500)
```

自定义控制循环用 `policy.act(actor_observation, deployable_physics, episode_start)` 获取确定性动作，
环境执行后调用 `policy.record_action(executed_action)`，随后才能处理下一帧。首次必须从真实回合
起点开始；新进程不恢复已运行的模拟器状态。低层控制循环应自行保证任务合同与 checkpoint 一致，
上述评测入口和训练恢复入口会在 reset/加载应用前检查任务合同。

冻结期间只更新运行时记忆，不更新权重或观测归一化统计，不调用 Teacher 或特权监督目标。
评测前后对参数及持久缓冲区计算 SHA256；变化即报错。报告包含动作指纹、奖励均值、终止/截断计数。

schema 5 训练恢复使用 `HoloMotionPhysicsMJLabPPO.from_checkpoint(path, task)`，会恢复完整模型、校准及优化器；
runner 的 `start_iteration` 应设为文件中 `iter + 1`。此入口开启新的真实回合，不声称恢复世界或 RNG 状态。

schema 4 仍可冻结加载，恢复原架构和原公式；实测旧检查点的动作指纹与修复前完全一致。
新架构新增可测性编码、动作条件和校准状态，所以 schema 4 不能直接恢复到 schema 5 训练器。
旧 schema 3 缺少网络配置和 residual 限幅等信息，独立加载明确拒绝。它只能使用原版本训练器、
原始配置和 sidecar 显式迁移为 schema 4；此后可冻结评测。当前训练器不支持这一旧格式迁移，
也不会猜测丢失的配置。采用新算法时从基础 Actor 创建 schema 5 新实验。

## 真实仿真入口

在安装好 CUDA、mjlab/MuJoCo/Warp 并准备资产、参考动作与正式 schema 4/5 checkpoint 的环境运行：

```text
python tools/evaluate_physics_holomotion_k1_mjlab.py --checkpoint model_4.pt --asset-root assets/K1 --motion motions/reference.npz --output runs/physics_eval.json --steps 500 --num-envs 1 --seed 42
```

`--motion` 可重复传入，文件需满足已有 MotionLibrary 的命名 mjlab 数据合同。
`--ablation` 支持 full、no_physics、no_latent、shuffle_latent、no_history；打乱 latent 要求至少两个环境。
若 checkpoint 未保存任务合同，需额外提供 `--onnx` 读取元数据；该 ONNX 不参与动作推理。
入口不会回退到合成环境。命令帮助已在 CPU 环境验证，实际 CUDA 路径仍需验收。

## 验证证据

- `tests/test_dynamics_continuity.py`：异步 reset、反馈开关、更新后与未裁剪完整历史重放一致；
  多 epoch、执行动作区分、容量检查、HoloMotion 连续多次 EMA 更新。
- `tests/test_physics_runtime.py`：非零 residual、不同 online/EMA 权重、无 sidecar 的独立加载，
  15 步多环境输出一致和冻结指纹不变；Teacher/特权目标访问会直接失败的评测用例；
  schema 5 完整训练恢复、缺失分支/校准拒绝、任务合同拒绝、schema 4 原行为兼容。
- 既有 GRU/LSTM/LRU 和交互反馈测试改为要求更新后持续同一回合；继续验证 PPO 重算一致。
- 独立冒烟观察到 Unified 环境 ticks 从 32 连续到 33，HoloMotion 从 6 连续到 7，
  Dynamics 更新不增加 reset 次数。

本地证据（被 Git 忽略）为 `runs/local_validation/blockers_smoke/evidence.json`、
`runs/local_validation/blockers_smoke/complete_physics.pt` 和
`runs/local_validation/20261012_blockers_verified/evaluation.json`。
后续算法修复和最新全量结果见 `runs/local_validation/algorithm_corrections_final/`。
完整回归命令：

```text
python tools/evaluate.py --output runs/local_validation/algorithm_corrections_final
```

本次解释器为 `runs/local_validation/dev_env/Scripts/python.exe`，Python 3.12.13、Torch 2.10.0+cpu。
这些结果验证代码语义与保存加载闭环；正式长训练仍应以真实参考轨迹的冻结基线、CUDA 下的
连续更新及 timeout 复验为前提。
