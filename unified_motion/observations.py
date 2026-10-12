"""身体观测：按合同重排关节，整理当前状态与上一步原始动作。

统一测评：python tools/evaluate.py
依赖：同目录的 robot_contract.py、PyTorch、Python >= 3.10。

本文件是我们项目的基础观测接口，不是 HoloMotion 官方观测的完整复刻。
只处理身体状态，不处理参考动作、奖励、历史缓存或物理仿真。
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

# 兼容“直接运行文件”和“作为项目模块导入”两种方式。
if __package__:
    from .robot_contract import RobotContract, get_robot_contract
else:
    from robot_contract import RobotContract, get_robot_contract


@dataclass
class RobotState:
    """存放当前状态；各张量来自同一时刻、同一批机器人。

    joint_pos / joint_vel 按 joint_names 排列，单位为 rad / rad/s。
    root_ang_vel_b：身体坐标系的角速度，单位 rad/s。
    gravity_b：身体坐标系的重力单位方向，不是 m/s² 加速度。
    身体坐标约定：x 向前、y 向左、z 向上；适配器负责统一。
    关节张量支持 [B, N] 或 [B, T, N]，身体向量对应 [..., 3]。
    """

    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_ang_vel_b: torch.Tensor
    gravity_b: torch.Tensor


class ObservationBuilder(nn.Module):
    """检查、重排、拼接观测；没有需要学习的网络权重。

    使用 nn.Module 只是为了让缓存跟随 .to(device) 移到 GPU。
    default_joint_pos 必须是当前机器人的固定默认姿态，按策略顺序排列。
    它不是当前时刻的动作参考，也不是机器人统一使用的全零站姿。
    """

    def __init__(
        self,
        robot: RobotContract,
        simulator_joint_names: Sequence[str],
        default_joint_pos: torch.Tensor,
        *,
        check_values: bool = True,
    ) -> None:
        """建立身体观测整理器，先核对机器人和默认关节角。

        输入：robot 关节合同；simulator_joint_names 仿真列顺序；default_joint_pos [N] 策略顺序默认角；check_values 是否检查非有限值。
        输出：无返回，保存默认姿态与重排索引。N 是关节数；不创建需要训练的权重。"""
        super().__init__()

        # 先确认仿真器给出的确实是这一款机器人的受控关节。
        robot.check_joint_set(simulator_joint_names, source="观测输入")
        self.robot = robot
        self.simulator_joint_names = tuple(simulator_joint_names)
        self.check_values = check_values

        # 不接受缺关节、整数角度、NaN 或无穷大的默认姿态。
        if not isinstance(default_joint_pos, torch.Tensor):
            raise TypeError("default_joint_pos 必须是 Tensor")
        if default_joint_pos.shape != (robot.dof,):
            raise ValueError(f"default_joint_pos 必须是 [{robot.dof}]")
        if not default_joint_pos.is_floating_point():
            raise TypeError("default_joint_pos 必须使用浮点数")
        if not torch.isfinite(default_joint_pos).all().item():
            raise ValueError("default_joint_pos 包含 NaN 或 Inf")

        # buffer 是随模块保存、移动的常量，不参与 PPO 参数优化。
        self.register_buffer("default_joint_pos", default_joint_pos.detach().clone())

        # 复用上一份代码，只在初始化时生成重排索引，避免每步重复建表。
        indices = robot.to_policy_order(
            torch.arange(robot.dof, device=default_joint_pos.device),
            self.simulator_joint_names,
        )
        self.register_buffer("policy_indices", indices, persistent=False)

    @property
    def obs_dim(self) -> int:
        # 角速度3 + 重力方向3 + 关节角N + 关节速度N + 上一步动作N。
        """输出当前机器人的身体观测宽度，等于 6+3N。

        输入：无。输出：整数；N 是受控关节数，K1 的结果为72。"""
        return 6 + 3 * self.robot.dof

    def forward(
        self,
        state: RobotState,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        """生成 [B, obs_dim] 或 [B, T, obs_dim]。

        previous_action 已经是策略顺序，不能再当成仿真器顺序重排。
        它是上一步送给环境的原始动作，不是角度目标或实际力矩。
        回合重置时，环境负责清零对应机器人的 previous_action。
        """
        if not isinstance(state, RobotState):
            raise TypeError("state 必须是 RobotState")
        if tuple(state.joint_names) != self.simulator_joint_names:
            raise ValueError("输入关节顺序发生变化，请检查适配器或重建 builder")
        if not isinstance(state.joint_pos, torch.Tensor):
            raise TypeError("joint_pos 必须是 Tensor")
        if state.joint_pos.ndim not in (2, 3):
            raise ValueError("joint_pos 必须是 [B,N] 或 [B,T,N]")

        # B 是环境数，T 是时间长度；不允许偷偷广播不同批次的数据。
        leading_shape = state.joint_pos.shape[:-1]
        if any(size == 0 for size in leading_shape):
            raise ValueError("环境数和时间长度都必须大于 0")
        n = self.robot.dof
        fields = (
            ("joint_pos", state.joint_pos, n),
            ("joint_vel", state.joint_vel, n),
            ("root_ang_vel_b", state.root_ang_vel_b, 3),
            ("gravity_b", state.gravity_b, 3),
            ("previous_action", previous_action, n),
        )

        for name, value, width in fields:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} 必须是 Tensor")
            if value.shape != (*leading_shape, width):
                raise ValueError(f"{name} 形状错误，应为 {(*leading_shape, width)}")
            if value.device != self.default_joint_pos.device:
                raise ValueError(f"{name} 和 builder 必须在同一设备")
            if value.dtype != self.default_joint_pos.dtype:
                raise TypeError(f"{name} 和默认姿态必须使用相同浮点类型")
            if self.check_values and not torch.isfinite(value).all().item():
                raise ValueError(f"{name} 包含 NaN 或 Inf")

        if self.check_values:
            # 防止误把 [0,0,-9.81] 或 [0,0,0] 当成重力单位方向。
            lengths = torch.linalg.vector_norm(state.gravity_b, dim=-1)
            if ((lengths - 1.0).abs() > 1e-3).any().item():
                raise ValueError("gravity_b 必须是单位方向，长度应接近 1")

        # 关节状态从仿真器顺序，转换成本项目约定的策略顺序。
        joint_pos = state.joint_pos.index_select(-1, self.policy_indices)
        joint_vel = state.joint_vel.index_select(-1, self.policy_indices)

        # 表示“偏离默认姿态多少”，不是“偏离当前参考动作多少”。
        joint_pos_rel = joint_pos - self.default_joint_pos

        # 只沿最后一维拼接，环境维 B、时间维 T 都原样保留。
        observation = torch.cat(
            (
                state.root_ang_vel_b,
                state.gravity_b,
                joint_pos_rel,
                joint_vel,
                previous_action,
            ),
            dim=-1,
        )
        if self.check_values and not torch.isfinite(observation).all().item():
            raise ValueError("拼接后的观测出现 NaN 或 Inf，请检查数值范围")
        return observation




