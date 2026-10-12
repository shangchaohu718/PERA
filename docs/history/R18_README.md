# Unified Motion · 整理版 0.18.0

本版沿用R17的唯一代码主线，**不增加新的神经网络、不再创建重复的PPO**。
新增重点是从“网络接口已测”走向“真实物理能否接通”的分层验收工具。

## 实际结果与阻塞

**474项回归：461通过、13 CUDA跳过、0失败、0错误。** pytest独立复跑、ZIP独立解压复跑同样通过。
Python3.13.5 / PyTorch2.10.0+cpu / NumPy2.3.5。

**真实物理没有通过结论：** mjlab、MuJoCo、MuJoCo-Warp、Warp仍未安装，外部下载失败；
本环境也没有可用CUDA和完整Booster mesh。因此真实编译、native CPU物理步进、mjlab GPU运行均未验收。
这与13项CUDA跳过是不同的范围，报告里分开记录。

## 一、R18增加/修复了什么？

1. `envs/physics_validation.py`：统一依赖、资产、编译、原生CPU探针、mjlab CUDA探针。
2. `tools/verify_physics.py`：唯一物理验收CLI，缺条件退出2，不返回假的PASS。
3. `envs/mjlab_k1.py`：用一张K1ControlSpec表同时生成训练PD和验收参数；严格检查角度单位、浮动根和限幅。
4. `train.py`：真正启动前检查版本/资产，保存XML及mesh指纹、PD和零动作裁剪差别。
5. `tools/check_static_model.py`：保留旧命令，但只转发到统一验收，不再维持另一份编译代码。

**特别发现：** 初始膝角0与零动作经过软限位后的0.11605rad不同。
本版记录该差异但不偷偷更改站姿。这个零动作探针不是“不施加力”也不是“保持初始关节角”。

## 二、代码结构

```text
unified_motion/
  api.py                       # 唯一正式算法导入入口
  train.py                     # 唯一正式K1训练入口
  robot_contract.py / observations.py / reference_observations.py
  models/                      # 独立动力学、MLP、Dense、MoE
  algorithms/sequence_training.py
  data/motion_sampler.py
  rewards/motion_tracking.py
  envs/
    mjlab_k1.py                 # K1控制配置及动作/状态桥
    mjlab_physics.py            # 接触和已登记物体观测
    physics_validation.py      # 新增：统一物理验收
  legacy/                      # 历史对照，不是默认训练路线

tests/                         # 集中CPU回归和可用时的CUDA测试
tools/verify_physics.py         # 不启动训练的物理验收
constraints/sim-r18.txt         # 源码核对的候选版本约束，不是实测完整lock
docs/PHYSICS_ACCEPTANCE.md      # 从依赖到物理步进的完整说明
reports/                       # 本次报告；r17_history下是明确归档的旧报告
```

旧兼容模块仍只转发，不维护第二份实现。算法使用方式和R17相同：

```python
from unified_motion.api import (
    DynamicsModel, build_policy, PPOConfig,
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater,
)
```

## 三、先运行CPU回归

解压到新目录，不把不同版本逐文件覆盖混用：

```bash
cd unified_motion_release18
python -m pip install --no-build-isolation --no-deps -e .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python tools/evaluate.py
python tools/verify_physics.py --help
python -m unified_motion.train --help
```

`--no-deps`要求已经装有相容的核心库。完整核心安装可用`pip install -e .`，注意不要意外替换训练机的CUDA PyTorch。
生产代码中没有新增`_self_test()`；所有测试统一在tests/。

## 四、真实物理的四步验收

完整安装、固定资产版本及说明：**`docs/PHYSICS_ACCEPTANCE.md`**。
依赖安装命令是待用户环境验证的候选，不伪装成本次成功安装。

```bash
# 1. 检查实际安装版本；不自动安装、不启动物理。
python tools/verify_physics.py --phase dependencies --output reports/dependencies.json

# 2. 检查真实资产、关节与执行参数；不编译、不步进。
python tools/verify_physics.py --phase assets \
  --asset-root ~/booster_assets_r18 --output reports/assets.json

# 3. 训练同源场景编译 + 原生CPU零策略动作物理探针。
python tools/verify_physics.py --phase native \
  --asset-root ~/booster_assets_r18 --steps 100 --output reports/native.json

# 4. 真正mjlab/Warp接通；不能拿native结果替代。
python tools/verify_physics.py --phase cuda \
  --asset-root ~/booster_assets_r18 --device cuda:0 --steps 20 \
  --output reports/mjlab_cuda.json
```

退出0=所请求范围通过；1=失败；2=阻塞。**即使0，也不代表策略已经学会运动。**

## 五、物理检查后再小规模训练

须有真实K1资产和已重定向且带列名元数据的项目NPZ。原始人体SMPL/BVH不能直接用。

```bash
python -m unified_motion.train \
  --asset-root ~/booster_assets_r18 --motions /absolute/path/to/k1_project_npz \
  --device cuda:0 --backbone mlp --num-envs 4 --horizon 16 --updates 2 \
  --minibatch-envs 4 --epochs 1 --save-every 1 --output runs/k1_mlp_smoke
```

Dense/MoE只改`--backbone dense`或`--backbone moe`，不换训练器。
R18训练启动会保存`asset_manifest.json`，包含文件SHA256和实际PD控制表。
候选依赖目标：mjlab1.6.0、MuJoCo/MuJoCo-Warp3.11系列、Warp>=1.14；仅核对源码，物理兼容待执行验收。

## 六、算法约定仍然保留

- PPO保存原始动作和log_prob，不拿裁剪后的关节目标冒充动作。
- L层、每层C帧Attention保留L*(C-1)帧前缀，前缀不参加主PPO损失。
- 策略更新后重建KV；独立动力学更新后显式重置环境和GRU记忆。
- 真正终止不bootstrap；时间截断使用reset前状态，均停止跨回合GAE。
- 有历史前缀时，MoE的balance_coef仍要求0；没有前缀时才支持现有辅助统计。
- top_k=1在当前选中logit再softmax的实现中没有来自组合权重的路由任务梯度；正式工厂用2。
- Router不直接读取latent，但相对目标包含状态信息，不能称为绝对的“与物理无关路由”。

## 七、当前能力边界

MLP/Dense/MoE网络、PPO、动力学监督的CPU逻辑已经测评；真实K1训练还没有实验结论。
G1仅有网络/数据接口，没有新增G1物理场景。AMP、高层接触规划、通用多物体接触图未实现。
当前maxforce脚接触只代表每脚选出的一个接触点，不是足底总力或CoP。
仿真真值也不是实机传感器保证。零样本、精细运动、长训稳定性仍须另行研究验收。

## 阅读与证据

- `CHANGELOG.md`：本次改了哪里。
- `docs/READING_GUIDE.md`：原有算法学习顺序。
- `docs/PHYSICS_ACCEPTANCE.md`：新物理检查逐步说明。
- `docs/MIGRATION.md`：旧目录和R17迁移。
- `reports/evaluation.json / cpu_regression.txt / pytest.txt`：本次真实测试。
- `reports/native_physics_gate.json`：本次真实BLOCKED结果，不是通过证明。
- `reports/DELIVERY_STATUS.md`：实际完成和未执行范围。
