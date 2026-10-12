"""旧的纯参考 Actor/Critic 兼容入口；物理策略请从 unified_motion.api 创建。

这里只转发类型，实际历史实现放在 legacy/，避免两套 Actor 混用。
"""
from .distributions import GaussianActionHead
from ..legacy.reference_actor_critic import (
    PolicyValueOutput, ActionEvaluation, ActionStep, MultiRobotValueCritic, MultiRobotActorCritic,
)
