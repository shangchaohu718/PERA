# Unified Motion R21 · LRU与无MoE Transformer

本版本在用户提供的R20工程上继续修改，**不另起项目、不重复写PPO或动力学模型**。
新增LRU（Linear Recurrent Unit）Actor插口；`transformer`是原`dense`的易读别名，真实结构没有MoE/Router。

## 1. 现在可以怎样组合

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
python -m pip install PyYAML pytest
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
并为 mjlab 生成与训练时一致的 PD/action-scale 配置。真实评测入口是：

```bash
python tools/evaluate_holomotion_k1_mjlab.py --help
```

参考轨迹转换、四元数顺序以及当前运行门禁见 `docs/HOLOMOTION_MJLAB.md`。现有
`model_9999.onnx` 已完成 CPU ONNX 前向合同验证；本机当前没有同时包含 mjlab 与
HoloMotion Runtime 的可用环境，且 CUDA 驱动不可见，因此不能把接口验证描述成
已完成的 mjlab 策略 rollout。
