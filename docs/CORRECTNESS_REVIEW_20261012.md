# 本地正确性审查与结构调整（2026-10-12）

本文保留首轮审查及复现记录。后续已修复下文两项 P1 阻塞，并将 checkpoint 元数据校验
移到状态加载之前；最新回归为 **703 passed、0 failed、18 skipped、438 subtests passed**。
当前实现、验证证据与仍待真实仿真验收的范围见 [后续修复说明](PHYSICS_RUNTIME_AND_CONTINUITY.md)。

审查基线：`ecd7e30`（`main`）。本次检查训练语义、回归覆盖和代码复用；没有启动正式训练。
环境为 Windows、CPython 3.12.13、PyTorch 2.10.0+cpu、NumPy 2.5.4。
机器虽有 NVIDIA GPU，本次独立测试环境使用 CPU，未安装 mjlab/MuJoCo/Warp，也没有正式权重或运动数据。

## 结论

已有测试主要证明张量公式和局部接口，不能直接证明长训练语义正确。原始回归在这台电脑上为
664 passed、1 failed、18 skipped；失败项来自 Windows 路径表示。
另发现已有用例未覆盖的 rollout 缓存、timeout、单样本归一化问题，并完成下面的修复。
最终在仅安装 CPU Torch 和 `.[dev]` 的独立环境中，完整回归为 **681 passed、0 failed、
18 skipped、436 subtests passed**（699 个顶层用例；新增 16 个）。18 个跳过项均需 CUDA。
`uv pip check` 通过，声明依赖无冲突；`git diff --check` 通过。
首轮结束时，Dynamics 更新导致环境重置、Physics 完整策略的冻结评测/部署加载尚未修复；
这两项现已在后续修复中完成 CPU 验证。

## 已修复

| 优先级 | 问题与影响 | 修改和验证 |
|---|---|---|
| P1 | HoloMotion 采样保留上一 rollout 的 KV cache，PPO `sequence_logp` 仅重算当前 rollout；旧策略本身就不能重现采样分布。 | 普通和 Physics collector 在每段 rollout 起点清空 Actor cache，不重置环境或 Dynamics。与 vendored `PPOTF._reset_rollout_forward_state` 的边界语义一致。 |
| P1 | 将 `terminated \| truncated` 直接用于 GAE 的 bootstrap 掩码，timeout 丢失末状态价值。 | 共用 `algorithms.common.compute_gae`；只在真正终止时关闭 bootstrap，两种边界均切断优势递归。timeout 的 Critic 值在 reset 前读取。 |
| P2 | 只有一个环境、一个时间步时，`std()` 的样本标准差为 NaN。 | 共用 `normalize_advantages`，单样本使用零自由度修正；正常批次仍保持原样本标准差语义。普通/Physics 采样和 mini-batch 归一化均使用同一函数。 |
| P2 | 资产清单使用系统路径分隔符，Windows 与 Linux 的报告键及指纹不同。 | 相对资源名使用 `as_posix()`；不改变真实文件访问方式，现有缺失 mesh 测试通过。 |
| P2 | `[dev]` 未声明测试直接使用的 safetensors、tensordict；原版网络集成测试还需要 loguru。 | 补全测试依赖，并在独立、只装 CPU Torch 和 `.[dev]` 的环境执行完整回归。 |

结构调整保持原模型和优化器不变：GAE 数值计算复用已有公共实现；两条 HoloMotion 路径共用
`_finish_transition`，集中维护“读最终价值→清理结束行缓存→reset/observe”的顺序。
新增 `task.critic_observation(env_ids)` 专门读取指定环境的 Critic 输入，不推进 Actor 的参考历史，
也不修改 observe/step 的 pending-reference 状态。这样同一批中其他环境参考片段结束时，不会阻止
仍有有效参考的 timeout 环境进行 bootstrap。

`HoloMotionPPORollout` 新增可选 `truncated` 和 `timeout_values` 字段，保留原调用兼容性。
未提供 timeout 信息的旧 GAE 调用仍将 `dones` 视为真正终止。

## 验证方法与证据

新增测试位于 `tests/test_holomotion_rollout_boundaries.py`：

- 手算非平凡 gamma/lambda 的 timeout 回报，故意令 reset 后的值与 final value 不同，排除串回合。
- 同一批包含 timeout、真正终止、持续运行三种环境；覆盖最后一步 timeout 和单样本归一化。
- 历史敏感的确定性 Actor 验证第二段 rollout 的采样/重算一致性。
- 实际导入仓库固化的 `PPOTFRefRouterActor`、`PPOCritic`，缩小网络宽度并使用随机初始化权重；
  普通和 Physics 两条路径均完成两段采样、log-prob/mu 重算和 PPO 更新。
- 验证 Critic 单独读取不改变任务 pending 状态，且仅选择需要 bootstrap 的行。

原始源码单独解包到 `runs/local_validation/baseline_source/`，没有回滚当前工作目录。
原版网络回归在原始代码中两条路径均失败：mu 最大绝对差分别约 0.006417 和 0.007896；
修复后通过 `atol=2e-5, rtol=2e-5` 的 mu 比较和 `atol=2e-4, rtol=2e-5` 的 log-prob 比较，
且首次 PPO ratio 均约为 1。这里验证的是原版网络计算路径，不是正式 checkpoint 表现或物理技能。

原始日志和本次生成的报告保存在本地、被 Git 忽略的目录中：

- `runs/local_validation/20261012/evaluation.json`：原始完整回归。
- `runs/local_validation/baseline_new_regressions.txt`：原始代码上第一组 14 个新增用例失败；
  其中也包括旧 API 缺少新参数/方法的合同失败，不应把 14 理解为 14 个独立算法问题。
- `runs/local_validation/baseline_vendored_network.txt`：原版网络的两项采样/重算失败。
- `runs/local_validation/20261012_final/`：最终完整回归、逐项测试事件、依赖版本快照。
- `runs/local_validation/dynamics_reset_reproduction.json`：修复前的环境重置复现。

## 首轮遗留发现（以下为修复前状态）

### P1：Dynamics 更新强制重置物理环境

`PhysicsSequenceCollector.on_dynamics_updated()` 将 `_ready` 置为 False，下一次 collect 调用
`reset_all()`；`HoloMotionPhysicsMJLabPPO.update_dynamics()` 的 finally 清空 `_observation`，
下一次 collect 同样触发 reset。此行为被部分现有测试明确写成预期，说明“测试全绿”不足以证明语义合理。

本次用无自然终止的 TensorTask 复现：采样 32 步后更新 Dynamics，再采样 1 步，环境 ticks 从
`[32,32]` 变成 `[1,1]`，reset 次数增加；保持同一回合应为 `[33,33]`。

不能只删除 reset 或继续使用旧 hidden。后续需要独立的历史重放组件，保存 deployable observation、
真实 episode_start、实际执行动作、必要的反馈输入；更新 online/EMA 参数后，重算预测误差和 GRU
记忆，再重建 Actor 历史。应覆盖异步局部 reset、跨 rollout、反馈开启/关闭、动作裁剪，以及历史
不足时的显式拒绝行为。当前改动不声称已完成该重构。

### P1：Physics 完整策略缺少独立冻结评测/部署闭环

训练 checkpoint 保存 residual、online Dynamics、EMA Dynamics，但基础 Actor safetensors
只包含原版 Actor；`evaluate_holomotion_k1_mjlab.py` 运行 ONNX 基础策略，不能验证这些额外分支。
下一步应将完整策略配置/状态的加载和初始化抽成同一个运行时入口，由训练恢复和冻结评测共同使用；
加入保存前后输出一致、冻结期间权重不变、缺失分支拒绝加载的测试，再执行真实参考轨迹基线。

### P2：checkpoint 加载失败可能已部分修改运行状态

`load_optimizer_checkpoint()` 在加载优化器和 Physics 模型之后才校验 physics_training_config
以及 iteration。无效配置会报错，但错误发生时部分权重/优化器可能已经改变。
首轮为静态发现。后续已提前校验 iteration、训练配置及 schema 4 的模型/任务合同，
并测试无效配置拒绝后状态不变；这不代表任意损坏的优化器状态加载都具有事务回滚能力。

## 复现

先安装兼容 Python 并在项目根目录创建单独的 CPU 环境；不要覆盖已有 GPU 训练环境。
本机可用 uv，下面的命令可创建新的审查环境：

```powershell
uv venv --python 3.12 runs/review-env
uv pip install --python runs/review-env/Scripts/python.exe "torch>=2.10,<2.11" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python runs/review-env/Scripts/python.exe -e ".[dev]"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:PYTHONUTF8 = "1"
.\runs\review-env\Scripts\python.exe tools/evaluate.py --output runs/review
```

本机最终测试使用的解释器是 `runs/local_validation/dev_env/Scripts/python.exe`。
依赖完整性另用 `uv pip check --python runs/local_validation/dev_env/Scripts/python.exe` 检查。
