"""旧参考跟踪 PPO 的兼容入口；新物理训练统一用 algorithms.sequence_training。

此文件只转发，不维护第二套 PPO 公式。旧测试和调用仍可导入这里。
"""
from .common import PPOConfig, compute_gae, _require_number
from ..legacy.reference_ppo import HistoryPrefix, PPOBatch, RolloutBuffer, PPOStats, PPOTrainer
