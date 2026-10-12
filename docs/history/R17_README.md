# Unified Motion · 整理版 0.17.0

**本版以你提供的第 16 阶段源码包为基准修订，不是另起一个不兼容的新项目。**
主线现在是：同一套任务接口和独立动力学模型，服务 MLP / Dense Transformer / MoE Transformer，并使用统一时序采样与 PPO 更新。

## 先看结果，而不是文件编号

本次实际运行：**438 项回归，425 项通过，13 项 CUDA 跳过，0 失败、0 错误**。
随后 pytest 独立复跑：**425 passed, 13 skipped**，另有 409 个通过的子用例，子用例不重复计入438。
完整包重新解压后也复跑了同一套测试，结果相同。
环境：Python 3.13.5、PyTorch 2.10.0+cpu、NumPy 2.3.5。
报告在 `reports/evaluation.json`、`reports/cpu_regression.txt`、`reports/pytest.txt`。

**范围限制必须保留：** 当前环境没有 CUDA，也没有 mjlab、MuJoCo、Warp；下载依赖失败。
因此 CPU 网络/数据/训练接口套件已通过，但**实际机器人模型的原生 CPU 编译、mjlab 接触求解、GPU 训练均未验证**。
桥接测试用的是明确标记的假环境，不是物理引擎替代品。
不能把“CPU套件全过”扩大成“除了CUDA，其余任意运行都已通过”。

## 一、以后只用一个正式入口

```python
from unified_motion.api import (
    build_policy, DynamicsModel, PPOConfig,
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater,
)
```

`api.py` 是统一导入入口；实际训练用 `python -m unified_motion.train`。
不再分别找第14、15、16版本里的不同PPO更新器。

```text
当前物理观测 + 过去的实际经历
              ↓
       DynamicsModel（独立训练）
              ↓ context / latent
任务目标 ──→ MLP / Dense / MoE ──→ 原始动作
                                   ↓
                         环境真实执行一个控制步
                                   ↓
                    最终观测 / 奖励 / 结束原因
                                   ↓
           PhysicsSequenceCollector 保存独立快照
                    ├─ PPO：只训练策略/价值
                    └─ DynamicsUpdater：只训练物理预测器
```

## 二、目录按职责划分

```text
unified_motion/
  api.py                         # 推荐的统一导入和建模入口
  train.py                       # 唯一正式 K1 mjlab 训练命令
  robot_contract.py              # 关节名字、列顺序和机器人注册
  observations.py                # 当前身体状态
  reference_observations.py      # 当前/未来参考目标
  models/
    physics.py                   # 物理观测/目标、独立动力学、MLP基线
    physics_policy.py            # Dense / MoE 物理条件化策略（无v2后缀）
    attention.py                 # 时间注意力及KV缓存
    encoder.py                   # 多机器人输入编码
    reference_fusion.py           # 唯一正式参考融合实现
    moe.py / transformer.py      # 专家层和共享主干
    distributions.py             # 共用高斯动作输出头
  algorithms/
    common.py                    # 共用 PPOConfig / GAE
    sequence_training.py         # 新增：任务、采样、时序PPO、动力学优化
  data/motion_sampler.py         # 标准机器人NPZ读取、重采样、时钟
  rewards/motion_tracking.py     # 关节/根/部件跟踪；可选鞋底专项
  envs/mjlab_k1.py               # K1资产/位置PD/手动reset桥
  envs/mjlab_physics.py          # 脚地接触和已登记物体观测
  legacy/                        # 历史对照实现，不是新项目默认主线

tests/                           # 统一测试，不再散落在生产模块里
tools/                           # evaluate.py / check_static_model.py
 docs/                            # 阅读顺序、接口、迁移及能力边界
 reports/                         # 本次真实执行的结果
```

旧 `physics_dynamics.py`、`physics_policy_v2.py`、`actor_critic.py`、`algorithms/ppo.py`、`physics_training.py` 是**兼容转发文件**。
它们不维护第二套核心；不用继续编辑这些路径。
原参考跟踪 Actor/PPO 及旧单步 MLP 更新器移入 `legacy/`，保留供对照和旧测试。
所有新物理训练只走 `sequence_training.py`。

## 三、安装与CPU测评

解压到**新目录**，不要逐文件覆盖旧工程后留下混杂版本。
已有相容 PyTorch/NumPy 时：

```bash
cd unified_motion_release17
python -m pip install --no-build-isolation --no-deps -e .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/evaluate.py
```

安装核心依赖可以用 `pip install -e .`；有 NVIDIA GPU 时不要把现有 CUDA PyTorch换成CPU轮子。
`pyproject.toml` 的 torch 范围固定到2.10系列；Python>=3.10是代码语法目标，**本次只实测3.13.5**。

其他运行方式：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py' -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
python -m unified_motion.train --help
```

`evaluate.py --require-sim-imports` 会额外要求真实仿真依赖可导入；当前环境会退出2，而不是伪装通过。
普通测评返回0，只表示执行的核心套件成功，报告仍会显示实际物理检查未完成。

## 四、如何创建和训练三种策略

下面的 `task` 是真实任务适配器，不是可以省略的变量。已有K1接线使用 `MotionTrackingTask(physics_bridge, sampler, reward_fn)`。

```python
from unified_motion.api import *

robot = get_robot_contract("k1_22dof")
dynamics = DynamicsModel(robot)  # 在CPU构建；真实训练时与策略/环境一起 .to(device)
policy = build_policy({"k1_22dof": robot}, backbone="moe")  # 也可以 mlp / dense

# task 由真实环境、动作采样器和奖励函数构成；见 train.py 的完整构造。
collector = PhysicsSequenceCollector(task, dynamics, policy)
ppo = PhysicsSequencePPO(policy, PPOConfig(balance_coef=0.0, entropy_coef=0.0))
dyn_optimizer = DynamicsUpdater(dynamics)

rollout = collector.collect(horizon=32)
ppo_stats = ppo.update(rollout)       # context/latent保持采样时的值，只改策略
physics_stats = dyn_optimizer.update(rollout)  # 用真实 next_observation 改动力学
collector.on_dynamics_updated()      # 明确通知下一轮重建环境和记忆
```

顺序不能反过来。不要在一次 rollout 采到一半时修改任意模型权重。
真实生产入口将动力学更新间隔设为可配置，默认每8轮更新一次，减少反复清空环境。
这是一种明确的研究基线选择，**不是“权重更新后仍无缝保留GRU记忆”的实现**。

## 五、真实 K1 mjlab 训练命令（待GPU验收）

需要已有 NVIDIA CUDA 环境、mjlab，以及官方 `booster_assets` 和重定向后的项目 NPZ。
可选依赖候选固定为 `mjlab==1.6.0`（`pip install -e '.[sim]'`），**本次没有成功安装/验证该组合**；需要在你的GPU机核对依赖解算结果。
不要把人体BVH/SMPL、未说明列顺序的原生NPZ或旧HDF5直接当成项目标准NPZ。

先静态编译（依赖齐全时可用CPU）：

```bash
python tools/check_static_model.py --asset-root ~/booster_assets
```

再小规模接通：

```bash
python -m unified_motion.train \
  --asset-root ~/booster_assets \
  --motions /absolute/path/to/k1_project_npz \
  --device cuda:0 --backbone mlp \
  --num-envs 4 --horizon 16 --updates 2 \
  --minibatch-envs 4 --epochs 1 --save-every 1 \
  --output runs/k1_mlp_smoke
```

Dense / MoE 只改 `--backbone dense` 或 `--backbone moe`，不用换一套PPO。
命令接通不是控制成功。先看关节映射、数值、实际动作，再安排长期训练。
`--resume` 恢复权重、Adam状态及更新计数，但**不会恢复物理场景、接触状态、参考时钟和运行时记忆**；因此不是逐步完全一致的仿真续跑。

## 六、本版已接通和仍不支持的边界

| 内容 | 当前状态 |
|---|---|
| MLP / Dense / MoE 的动作、采样、PPO更新 | 统一代码路径，CPU实际更新通过 |
| 独立动力学预测、误差反馈、监督学习 | CPU因果/重置/梯度/保存加载通过 |
| K1 / G1-23 / G1-29 网络接口 | CPU通过；每份采样batch只能有一种身体 |
| MLP多机器人共用同一实例 | 未实现；每种机器人一个MLP实例 |
| 完整动作 / 纯接触 / 纯物体目标 | 网络与统一训练接口CPU通过；无目标会报错 |
| K1真实环境适配 | 已有源代码；GPU与真实资产编译待验收 |
| G1真实场景适配 | 尚未实现，不能把K1配置改个ID冒充 |
| 运动任务 | 标准NPZ参考跟踪；接触/物体任务须实现 PhysicsTask 的目标与奖励 |
| 通用多物体接触图、滑板机构和高层规划 | 未实现；两只脚与地面的接触不是通用接触图 |
| AMP | 未加入 |
| 零样本、控制精度、实机安全 | 没有实验结论 |

## 七、重要的算法约定

**历史与概率一致。** 有L层、每层C帧局部注意力，跨轮采样保留 `L*(C-1)` 个原始输入前缀。
PPO重算真实时间序列，前缀不计入主策略/价值损失。旧log_prob/mean/std/value在更新前必须复算一致。

**动作与执行不同。** 保存原始动作和原始动作概率；不要拿裁剪后的PD目标计算PPO概率。

**结束与截断不同。** 真正终止不bootstrap；时间截断读取reset前的final observation做bootstrap。两者均停止跨回合的GAE递推。

**两种记忆不同。** Transformer KV与动力学GRU分开管理；策略更新后重建KV；动力学更新后本版显式重新开始环境，不能把旧GRU直接当作新模型的记忆。

**MoE辅助项。** 当前有历史前缀时要求 `balance_coef=0`，否则明确报错，因为旧MoE辅助统计没有独立排除前缀。
没有前缀时支持该项。不要删掉这项保护来强行训练。
当前选中专家后再softmax归一化的实现，top_k=1会使选中权重恒为1，Router没有来自该组合权重的任务梯度；正式工厂固定top_k=2。

**Router并非完全不看状态。** 它不直接读取dyn latent，但目标token含接触/物体误差，运动参考也相对当前机身表达；因此不是与物理反馈无关的“纯目标路由”。

**传感器值不是实机保证。** 当前mjlab接触/物体状态是仿真真值。将来需要传感器或估计器，不能据此声称已经可部署。

## 八、阅读顺序与修改记录

- `docs/READING_GUIDE.md`：先看什么、每组文件解决什么问题。
- `docs/MIGRATION.md`：旧路径对应哪里，旧权重为什么可能不兼容。
- `CHANGELOG.md`：本版实际修复的缺陷与新功能。
- `reports/source_manifest.json`：输入包和新文件的SHA256。
- `reports/DELIVERY_STATUS.md`：本次通过/跳过/阻塞的准确范围。

**本项目是可执行、可回归的研究代码，不是已经完成通用机器人零样本控制的产品。**
