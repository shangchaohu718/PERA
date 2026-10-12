"""统一训练入口的确定性学习率衰减回归。"""

import pytest

from unified_motion.algorithms.common import learning_rate_scale


def test_cosine_schedule_reaches_configured_floor():
    assert learning_rate_scale(0.0, schedule="cosine", min_ratio=0.1) == 1.0
    assert learning_rate_scale(1.0, schedule="cosine", min_ratio=0.1) == pytest.approx(0.1)
    assert learning_rate_scale(0.5, schedule="cosine", min_ratio=0.1) == pytest.approx(0.55)


def test_linear_and_constant_schedules_are_explicit():
    assert learning_rate_scale(0.5, schedule="linear", min_ratio=0.2) == pytest.approx(0.6)
    assert learning_rate_scale(1.0, schedule="constant", min_ratio=0.2) == 1.0
    with pytest.raises(ValueError, match="lr_schedule"):
        learning_rate_scale(0.5, schedule="mystery", min_ratio=0.1)
