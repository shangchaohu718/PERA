# K1 Foot-Ground Dynamics Teacher V1 — 最终报告

## 结论

Teacher V1 的真实数据采集、模型、三权重×三 seed 训练及 checkpoint-exclusive
独立评估均已完成，但**独立质量门失败**。禁止进入 Student 蒸馏、Actor 集成或
PPO；本次对应计数均为 0。

安全的 RSL-RL 适配已解决原始加载风险：native checkpoint 只作为不可变来源，
推理使用 `weights_only=True` 的 tensor-only bundle。采集前后均验证唯一
checkpoint ID、source SHA-256、冻结参数和 eval 模式；6/6 通过，旧权重未修改。

## 数据与物理审计

- 6 个行为阶段不同的旧策略；3 rollout seeds；动态摩擦系数 0.3/0.5/0.7。
- 54 条完整轨迹，14,063 个真实 50 Hz transition；起始相位随 seed 覆盖约
  0.032 / 0.497 / 0.810。
- 来源隔离分割：36 train（4 checkpoints）/ 9 validation（1 checkpoint）/
  9 test（1 完全未参与训练的 checkpoint）。smoke 数据未进入清单。
- 每控制步保留 4 个 PhysX 子步的原始 normal/friction point buffers；54/54
  通过最终子步力求和、20 ms 冲量积分、法向朝向、滑速恒等式、mask 与来源哈希审计。
- train 覆盖：左右锚定/对侧滑动 685/450，双支撑 2,460，接触切换 3,796，
  active-slide contact 2,223，意外滑移 4,906，impact 1,984。
- held-out 分布明显漂移：stable 仅 27，separation 3,083，impact 1,093；这是
  泛化失败的重要诊断，而不是重新划分测试集的理由。

## 模型与训练协议

- 94-D Student 可见历史状态 + 22-D 当前执行动作 + 12-D 当前特权全局状态 +
  左右脚各 15-D 当前物理状态。
- 左右脚共享 foot encoder/contact head，并使用 side ID；3 层、宽 128 的 causal
  Transformer；Contact 与 Dynamics 输出层独立。目标严格对齐到 `t+1`。
- 训练集统计量标准化回归目标；所有几何、运动、力、冲量、动力学 loss 独立计算并
  mask；Replay 固定为 1。每 run 100 epochs、3,600 次真实 optimizer updates。
- 固定轨迹顺序基线出现 mode 全 separation；仅加入 seeded whole-trajectory
  epoch shuffle 后修复概率校准；再单独加入 train-only inverse-frequency mode
  weights；20→100 epochs 的时长改动另有 validation 对照。
- 正式 contact loss 权重消融为 1.0/0.1/0.03 × seeds 730201/730202/730203。
  validation pooled contact F1 分别为 0.8973/0.8865/0.8864，故在 test 访问前
  冻结选择 1.0。三种权重均未通过全部 validation gate。
- 共享主干梯度审计的三 seed 均值（contact norm / dynamics norm / cosine）：
  1.0 = 0.3831 / 0.6326 / -0.0092；0.1 = 0.03385 / 0.6344 / 0.0109；
  0.03 = 0.00974 / 0.6631 / 0.0128。逐次审计值保存在各 run 的 `metrics.jsonl`。

## 独立指标（选中权重 1.0，三 Teacher seeds 池化）

### 每只脚

| 指标 | Validation 左/右 | Held-out test 左/右 |
|---|---:|---:|
| Contact BCE | 0.407 / 0.451 | 0.651 / 0.659 |
| Contact Brier | 0.123 / 0.140 | 0.228 / 0.230 |
| Precision | 0.890 / 0.870 | 0.465 / 0.436 |
| Recall | 0.921 / 0.910 | 0.478 / 0.447 |
| F1 | 0.905 / 0.889 | 0.471 / 0.441 |
| Contact-mode macro-F1 | 0.481 / 0.476 | 0.306 / 0.243 |
| Switch MAE (frames) | 1.369 / 2.608 | 0.724 / 0.679 |
| Contact point RMSE (m) | 0.0330 / 0.0297 | 0.0435 / 0.0520 |
| Normal mean error (deg) | 5.82 / 11.26 | 8.81 / 7.16 |
| Normal velocity RMSE (m/s) | 0.0122 / 0.0114 | 0.0650 / 0.0656 |
| Tangent velocity RMSE (m/s) | 0.1518 / 0.1106 | 0.4315 / 0.4708 |
| Slip speed RMSE (m/s) | 0.1841 / 0.1289 | 0.4908 / 0.5701 |
| Normal force RMSE (N) | 129.1 / 126.7 | 358.6 / 362.2 |
| Tangential force RMSE (N) | 25.16 / 28.91 | 94.69 / 87.33 |
| Contact impulse RMSE (N·s) | 1.761 / 1.800 | 2.968 / 2.768 |
| Foot load-ratio RMSE | 0.293 / 0.295 | 0.533 / 0.491 |

### 滑动、冲击与摩擦分层

| 指标 | Validation | Held-out test |
|---|---:|---:|
| Active-slide tangent/slip RMSE (m/s) | 0.222 / 0.265 | 0.554 / 0.675 |
| Unexpected-slip tangent/slip RMSE (m/s) | 0.138 / 0.139 | 0.420 / 0.481 |
| Impact normal-force RMSE (N) | 254.4 | 460.8 |
| μ=0.3 normal/tangent-force RMSE (N) | 127.1 / 20.8 | 341.0 / 59.7 |
| μ=0.5 normal/tangent-force RMSE (N) | 120.7 / 25.5 | 380.1 / 100.6 |
| μ=0.7 normal/tangent-force RMSE (N) | 133.0 / 31.2 | 359.0 / 106.7 |

### 动力学

| 指标 | Validation | Held-out test |
|---|---:|---:|
| Δ joint velocity RMSE | 1.409 | 13.736 |
| 相对 Persistence 的 RMSE 改善 | **−9.10%** | +3.14% |
| Δ root angular velocity RMSE (rad/s) | 0.308 | 1.549 |
| Joint residual RMSE | 59.98 | 392.23 |
| 相对 simple-physics residual baseline 的 RMSE 改善 | −0.66% | +0.08% |

Validation 仅有 17/18 项通过，已在看 test 前判失败；held-out 最终只有 11/25
项通过。不能因 held-out 的 joint-delta 指标偶然优于 Persistence 而推翻 validation
门禁，也不能用总 loss 下降宣布成功。

## 可辨识性与下一步

当前 Teacher 使用当前真实接触物理、摩擦与根状态；其精度不等价于未来 Student
仅凭 `q/dq/IMU/tau_est/action history` 的可部署精度。接触力、接触点、摩擦系数、
外部扰动尤其可能不可辨识。本轮没有训练 Student。

下一轮应先处理 checkpoint/domain shift 与 dynamics 泛化：可考虑 checkpoint-balanced
sequence sampling、只基于 validation 的 domain-balanced objective、Dynamics/Contact
共享层冲突方法，以及更强但仍轻量的物理 baseline。不得复用 held-out checkpoint
调参；需要新的最终测试来源。
