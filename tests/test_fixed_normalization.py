"""固定物理量归一化回归：训练/部署共用常数，不依赖运行统计。"""

import math

import torch

from unified_motion.models.fixed_normalization import (
    ANGULAR_VELOCITY_RAD_S,
    JOINT_VELOCITY_RAD_S,
    body_observation,
    body_weight_n,
    contact_observation,
)
from unified_motion.robot_contract import get_robot_contract


def test_body_and_force_use_fixed_physical_scales():
    robot = get_robot_contract("k1_22dof")
    n = robot.dof
    body = torch.zeros(1, 6 + 3*n)
    body[:, :3] = ANGULAR_VELOCITY_RAD_S
    body[:, 6:6+n] = math.pi
    body[:, 6+n:6+2*n] = JOINT_VELOCITY_RAD_S
    body[:, 6+2*n:] = 1.0
    normalized = body_observation(body, robot)
    torch.testing.assert_close(normalized[:, :3], torch.ones(1, 3))
    torch.testing.assert_close(normalized[:, 6:], torch.ones(1, 3*n))

    contact = torch.zeros(1, 1, 12)
    contact[..., 6:9] = torch.tensor([0.0, 0.0, 1.0])
    contact[..., 9] = body_weight_n(robot)
    contact[..., 10:] = 1.0
    normalized_contact = contact_observation(contact, robot)
    torch.testing.assert_close(normalized_contact[..., 9], torch.ones(1, 1))
    # 方向、接触布尔量和“力可用”标志不改变。
    torch.testing.assert_close(normalized_contact[..., 6:9], contact[..., 6:9])
    torch.testing.assert_close(normalized_contact[..., 10:], contact[..., 10:])


def test_normalization_has_no_running_state():
    robot = get_robot_contract("k1_22dof")
    sample = torch.randn(2, 6 + 3*robot.dof)
    first = body_observation(sample, robot)
    body_observation(torch.randn(100, 6 + 3*robot.dof) * 1000, robot)
    torch.testing.assert_close(body_observation(sample, robot), first, rtol=0, atol=0)
