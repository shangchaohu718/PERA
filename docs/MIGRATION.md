# R21迁移

`dense`与`transformer`是同一个主干；不必迁移权重，只是新配置可用更直观的名字。
旧R20参数树保持，LRU字段只进入新LRU模型元数据。
从checkpoint继续完整实验仍用训练时保存的原始配置（包括dense/transformer字符串），不要改配置后绕过一致性检查。
LRU、GRU、LSTM记忆不可互用；裸LRU为[L,B,N,2]实部/虚部，正式策略已有缓存身份检查。
本轮不替换独立DynamicsModel的GRU。新配置建议新输出目录。

---

# R17 → R18

请使用新目录。算法入口api.py、build_policy、sequence_training未更换。
所有网络/动力学权重结构沿用R17。不要再次寻找旧版reference_fusion。

新增的正式物理入口是tools/verify_physics.py；check_static_model.py只做兼容转发。
训练机首次运行前先看docs/PHYSICS_ACCEPTANCE.md。BLOCKED不是PASS。

K1 XML必须显式标记radian，关节和电机不能关闭相应限幅；include/class不自动猜测。
此约束可能拒绝旧自制XML，应显式展开和审计，而不是关闭检查。

默认站姿和名义PD来源未改变。执行边界现在保证包含合法重置角，因而action=0
保持重置目标；旧版膝关节0→0.11605rad的首步跳变不再保留。
这只修正控制语义，不代表长期站立或接触已经通过；更换初始姿态仍应另建实验配置。

---

# 从阶段16迁移到0.17.0

## 请解压到新目录，不要叠加覆盖

保留原来的源码包和日志。新目录作为唯一正式工作目录，用Git单独提交。
这样旧版本可回退，也不会遗留同名 `reference_fusion` 或“v1/v2”并存。

## 路径对应

| 旧导入 | 当前唯一实现 | 旧路径 |
|---|---|---|
| models.physics_dynamics | models.physics | 转发兼容 |
| models.physics_policy_v2 | models.physics_policy | 转发兼容 |
| PhysicsPolicyV2Config | PhysicsPolicyConfig | 同一类的别名 |
| models.actor_critic | legacy.reference_actor_critic | 转发兼容；只给旧参考基线 |
| algorithms.ppo.PPOConfig / compute_gae | algorithms.common | 同一个对象，不复制公式 |
| algorithms.ppo.PPOTrainer / RolloutBuffer | legacy.reference_ppo | 旧参考基线 |
| algorithms.physics_training | legacy.mlp_physics_training | 历史单帧MLP，不接时序策略 |
| 新物理训练 | algorithms.sequence_training | 推荐从api.py导入 |

原Attention、MoE、ReferenceFusion只有一份正式实现，不增加FiLM/AMP等额外研究变量。

## 权重兼容不是文件路径兼容

推荐继续保存 `state_dict()`，不使用 `torch.save(model_object)`。

本版为 DynamicsModel 与 PhysicsMLPActorCritic 加了输入语义元数据，修复了物体四元数符号处理。
**旧阶段14/15权重不能假定直接继续训练等价。** 缺少新元数据时strict加载拒绝是故意的。
不要用 `strict=False` 或删除元数据绕过检查。需要导入旧实验时，应明确记录旧机器人关节顺序、物理观测定义、架构宽度，并单独写迁移记录与重新验收。

Dense/MoE基础权重形状多数保持；混合目标packing的运行行为已改变，旧cache绝不可复用。
凡模型或上游编码器权重更新，重新建立运行时记忆。

## 训练调用的统一替换

旧：MLP用PhysicsPPOUpdater，时序策略没有正式Collector。

新：三种模型都用下面的组合：

```python
from unified_motion.api import (
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater, PPOConfig,
)
collector = PhysicsSequenceCollector(task, dynamics, policy)
ppo = PhysicsSequencePPO(policy, PPOConfig(balance_coef=0.0))
dynamics_updater = DynamicsUpdater(dynamics)

batch = collector.collect(32)
ppo.update(batch)
dynamics_updater.update(batch)
collector.on_dynamics_updated()
```

`task`不是只有env.step的对象。它还必须给出当前目标、实际物理观测、奖励和严格的手动reset接口。
见 `PhysicsTask` Protocol 和已有 `MotionTrackingTask`。

resume恢复权重/优化器，不恢复场景；日志必须区分它和无中断仿真训练。
