"""Deterministic physical-unit normalization shared by training and deployment."""

from __future__ import annotations

import math

import torch

from ..robot_contract import RobotContract


SCHEMA = "fixed_physical_v1"
ANGULAR_VELOCITY_RAD_S = 10.0
LINEAR_VELOCITY_M_S = 5.0
POSITION_M = 2.0
JOINT_POSITION_RAD = math.pi
JOINT_VELOCITY_RAD_S = 10.0
TIME_S = 2.0
_BODY_WEIGHT_N = {
    "Booster K1 22DoF": 194.88546,
    "Unitree G1 23DoF": 350.0,
    "Unitree G1 29DoF": 350.0,
}


def body_weight_n(robot: RobotContract) -> float:
    """Return a fixed nominal body weight; never read randomized simulator mass."""
    try:
        return _BODY_WEIGHT_N[robot.name]
    except KeyError as error:
        raise ValueError(f"robot {robot.name!r} lacks a fixed body-weight scale") from error


def body_observation(value: torch.Tensor, robot: RobotContract) -> torch.Tensor:
    n = robot.dof
    return torch.cat((
        value[..., :3] / ANGULAR_VELOCITY_RAD_S,
        value[..., 3:6],
        value[..., 6:6+n] / JOINT_POSITION_RAD,
        value[..., 6+n:6+2*n] / JOINT_VELOCITY_RAD_S,
        value[..., 6+2*n:6+3*n],
    ), dim=-1)


def reference_observation(value: torch.Tensor, robot: RobotContract) -> torch.Tensor:
    n = robot.dof
    return torch.cat((
        value[..., 0:1] / TIME_S,
        value[..., 1:4] / POSITION_M,
        value[..., 4:10],
        value[..., 10:13] / LINEAR_VELOCITY_M_S,
        value[..., 13:16] / ANGULAR_VELOCITY_RAD_S,
        value[..., 16:16+n] / JOINT_POSITION_RAD,
        value[..., 16+n:16+2*n] / JOINT_VELOCITY_RAD_S,
    ), dim=-1)


def contact_observation(value: torch.Tensor, robot: RobotContract) -> torch.Tensor:
    return torch.cat((
        value[..., :3] / POSITION_M,
        value[..., 3:6] / LINEAR_VELOCITY_M_S,
        value[..., 6:9],
        value[..., 9:10] / body_weight_n(robot),
        value[..., 10:12],
    ), dim=-1)


def object_observation(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((
        value[..., :3] / POSITION_M,
        value[..., 3:7],
        value[..., 7:10] / LINEAR_VELOCITY_M_S,
        value[..., 10:13] / ANGULAR_VELOCITY_RAD_S,
        value[..., 13:16] / POSITION_M,
    ), dim=-1)


def contact_goal(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :3] / POSITION_M,
                      value[..., 3:6] / LINEAR_VELOCITY_M_S,
                      value[..., 6:7]), dim=-1)


def object_goal(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :3] / POSITION_M,
                      value[..., 3:6] / LINEAR_VELOCITY_M_S), dim=-1)
