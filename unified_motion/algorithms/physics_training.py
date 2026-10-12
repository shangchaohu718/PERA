"""历史单步 MLP 训练兼容入口。新代码使用统一的 PhysicsSequenceCollector/PPO。

不要把 Dense/MoE 传给这里；保留原类只是为了旧实验可以复查。
"""
from ..legacy.mlp_physics_training import (
    PhysicsRollout, PhysicsRolloutCollector, PhysicsPPOStats, PhysicsPPOUpdater,
    DynamicsTrainStats, DynamicsSupervisedUpdater,
)
