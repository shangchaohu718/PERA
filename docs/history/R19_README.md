# Unified Motion R19 · 可下载的源码交付

本包以R18完整工程为基础，实际新增了YAML实验入口、课程/接触计划、关系观测、冻结评测，并接通接触/物体预测误差回灌。
**不是只提供接口示意；CPU解析诊断下，MLP、Dense和MoE均实际完成训练、保存、加载及冻结评测。**

## 先说明已验证范围

- 本轮全量回归：486通过、13 CUDA跳过、0失败（含原有回归）。最终独立解压结果另见reports/current。
- 三种策略各训练3轮，执行独立动力学更新，随后每种策略冻结评测4个诊断回合，参数/缓冲区指纹不变。
- **这些短程评测成功率全部为0，说明连接可运行，不说明任务已经学会。**
- `diagnostic`是质点/运动平面的解析测试，不是K1，不是MuJoCo，不用于论文机器人成功率。
- 当前无mjlab/MuJoCo/Warp/CUDA，真实K1场景没有运行结论。`mjlab_contact`不会退回诊断后端。
- 没有预训练的机器人技能。包内`runs/*/checkpoint.pt`只属于诊断场景，禁止用于实体机器人。
- 没有完成自主接触规划器、滑板机构或任意器具零样本能力。任务执行器使用外部给定计划。

## 1. 直接运行已有的CPU闭环

在含相容PyTorch和NumPy的Python环境中：

```bash
cd unified_motion_release19
# 只补本轮配置/测评依赖，避免替换GPU机现有的PyTorch。
python -m pip install PyYAML pytest
python -m pip install --no-build-isolation --no-deps -e .

# 检查YAML及预先划分的物理域。
python -m unified_motion.experiment --config configs/diagnostic.yaml --mode validate

# 真正运行CPU诊断的采样、PPO和独立动力学更新。
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m unified_motion.experiment --config configs/diagnostic.yaml --mode train

# 从刚保存的权重评测，不创建优化器；每步验证state_dict未变。
python -m unified_motion.experiment --config configs/diagnostic.yaml --mode evaluate \
  --checkpoint runs/diagnostic/checkpoint.pt

# 集中测评。新增测试是pytest格式，应使用pytest运行全套，不只unittest discover。
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

另外两种策略直接使用`configs/diagnostic_dense.yaml`和`configs/diagnostic_moe.yaml`。
依赖缺失会报告错误，不能把所有用户环境都称为已经验证。实测版本写在`reports/current/delivery.json`。

## 2. YAML改哪里

| 配置 | 含义 |
|---|---|
| `model.backbone` | mlp / dense / moe，复用同一PPO |
| `dynamics.interaction_feedback` | true：把接触与物体预测误差接回GRU；false：R18身体误差基线 |
| `run.updates` / `run.horizon` | 更新轮数与每轮新控制步数 |
| `domains` | 摩擦、根部件质量比例、动作延迟、外力、支撑面速度区间 |
| `train_domains` / `evaluation_domains` | 提前固定的训练/评测域，名字不能重合 |
| `curriculum` | 初始难度、成功率升级阈值、按完成回合更新的窗口 |
| `plan` | 外部给定的接触阶段，位置和速度以对象坐标系表示 |
| `evaluation_plan` | 可选保留计划；不提供时只验证新物理参数，而非新任务组合 |
| `contacts` | 固定的机器人body—对象body关系、body局部测量点 |

范围分离报告不仅检查域名，还检查物理参数区间；重叠时明确报告，不能只换域名就宣称零样本。
`evaluation_plan`不自动检测语义相似性：任务组合是否训练过仍需审计配置和数据，程序不代替科研判断。

## 3. 本轮核心代码

```text
unified_motion/
  experiment.py                  YAML→建模→训练→checkpoint→冻结评测
  tasks/interaction.py           固定关系、对象局部目标、达标换阶段、课程、延迟和外力调度
  envs/diagnostic_interaction.py  可运行的CPU解析诊断（绝不是人形机器人）
  envs/mjlab_interaction.py       真实mjlab关系传感器/参数写回适配（待GPU验收）
  models/physics.py              升级现有动力学，不另建第二个DynamicsModel
  algorithms/sequence_training.py 保存/重放接触物体预测，旧PPO主公式不重写
configs/
  diagnostic.yaml                MLP、CPU、开箱即可检查的诊断配置
  diagnostic_dense.yaml
  diagnostic_moe.yaml
  k1_contact.yaml                K1真实接触任务配置，需资产/CUDA/依赖
```

其它R18模块、传统动作跟踪入口`unified_motion.train`、历史兼容转发文件都保留。
R18报告集中到`reports/r18_history/`，本轮报告只看`reports/current/`。

## 4. 真实K1入口

```bash
# 修改asset_root为本地官方资产；先跑R18保留的物理验收。
python tools/verify_physics.py --phase native --asset-root ~/booster_assets_r18
python tools/verify_physics.py --phase cuda --asset-root ~/booster_assets_r18 --device cuda:0

# 新的外部接触计划训练（与完整MoCap参考跟踪是两种任务入口）。
python -m unified_motion.experiment --config configs/k1_contact.yaml --mode train
```

当前真实工厂只创建K1和地面。通用关系适配类允许读取**调用者显式加入scene.entities的外部对象**，但YAML工厂没有自动生成滑板、台阶、受驱动平台。
非零`support_speed_m_s`在默认K1平面后端会明确抛异常；不能往观测填假速度冒充移动平台。
真实场景的摩擦、根部件质量/惯量设置、控制步延迟和外力代码已写，尚未运行实际后端验收。
因此本包**不宣称把上一需求的所有真实物理任务全部完成**。

## 5. 核心思想具体落在哪里

1. 当前物理观测和上次预测形成身体误差、接触概率误差及物体位置/速度/姿态误差。
2. 对回合边界、前后有效槽位和固定对象身份进行检查，错误只在时间对应时回灌。
3. 14维接触/物体误差经小网络进入当前物理context和GRU，进而影响Actor。
4. Collector保留之前预测的独立快照；动力学监督复放同样的反馈，不在训练时丢掉这条支路。
5. PPO与Dynamics使用独立优化器；冻结评测允许GRU/KV改变，但禁止模型参数或buffer改变。

目前误差使用单位尺度/限幅/有效性筛选，不是训练好的不确定性估计器。不会把mask称为校准置信度。
接触传感器是`maxforce`代表点，不是总足底支撑力或CoP；固定测量点与碰撞点含义在代码中分开。
独立模型能学到有用适应仍需实验，本包提供可训练连接而非已经证明的科研成果。

## 6. 权重与续跑

新增交互反馈网络改变Dynamics权重结构和元数据，建议新开实验；本轮配置禁止旧权重静默载入。
`--checkpoint`在train模式恢复同配置的网络、Adam和课程状态，物理场景重新reset，不保证逐步精确续跑。
在evaluate模式，配置必须与checkpoint中预先登记的配置完全相同，不能临时把测试改容易。
CUDA/RNG完整环境状态恢复、噪声置信度学习、多物体身份自动追踪、真实移动平台控制均非本轮完成项。

完整结果和每种策略的短程日志在`reports/current/`，没有把诊断权重包装为机器人checkpoint。
