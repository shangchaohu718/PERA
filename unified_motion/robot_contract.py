"""机器人关节合同：统一策略与仿真器之间的名字和列顺序。

第 01 个文件（多机器人版本）：
- 给 K1、G1 及未来机器人分别保存“关节身份证”；
- 按名字重排观测/动作张量，避免把膝盖数据误当成脚踝数据；
- 发现关节缺失、名字重复、维度错误时，立即报错；
- 不依赖 Isaac Lab、mjlab 或 MuJoCo，便于独立测试。

注意：
1. joint_names 是“本项目选定的策略顺序”，不保证等于 URDF/MJCF 的原始顺序。
   接上仿真器时，必须读取实际受控关节名，用 to_policy_order / to_simulator_order 对齐。
2. 目前只针对“每个受控关节对应一个位置数值”的机器人。
   root_qpos_dim=7 表示一个 MuJoCo 风格的浮动根（位置 3 + 四元数 4）。
   如有被动关节、球关节等，需要另外验证真实的 qpos 维数，不能直接套用。
3. K1 与 G1 的动作维数不同。本文件负责数据接口，不代表单个 Actor 已经能同时控制两种机器人。

统一测评：python tools/evaluate.py
依赖：Python >= 3.10、PyTorch
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# ========== 一、各机器人关节名单 ==========

# K1：项目所选的 22 个受控关节顺序。
# 这里的 aa 前缀遵循我们当前 K1 资产的命名；换资产时还要读取真实关节名进行核对。
K1_JOINT_NAMES: tuple[str, ...] = (
    "aahead_yaw_joint",
    "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# G1：将两种官方常见配置的共同关节只写一次，减少复制粘贴带来的错误。
# 这只是本项目的“策略顺序”，与实际仿真器顺序不同时，由下面的函数负责重排。
_G1_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

_G1_LEFT_ARM_23 = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
)
_G1_RIGHT_ARM_23 = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

_G1_LEFT_ARM_29 = _G1_LEFT_ARM_23 + (
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
_G1_RIGHT_ARM_29 = _G1_RIGHT_ARM_23 + (
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Candidate G1 dexterous-hand asset used by the project: seven actuated joints
# per hand (thumb 3, middle 2, index 2).  These are real action dimensions, not
# aliases for the wrist joints already present in the 29DoF body contract.
_G1_LEFT_HAND_7 = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
_G1_RIGHT_HAND_7 = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)

# Isaac Lab's bundled Unitree G1 USD uses its own 37-DoF articulation contract:
# 23 body/arm joints plus seven named finger joints on each hand.  It does not
# expose the 29-DoF MJCF wrist naming, so it must have a distinct robot id.
_ISAAC_G1_LEFT_ARM_HAND = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_six_joint",
    "left_four_joint",
    "left_zero_joint",
    "left_one_joint",
    "left_two_joint",
)
_ISAAC_G1_RIGHT_ARM_HAND = tuple(
    name.replace("left_", "right_", 1) for name in _ISAAC_G1_LEFT_ARM_HAND
)

# G1 23DoF：双腿 12 + 腰部 1 + 双臂 10 = 23。
G1_23DOF_JOINT_NAMES: tuple[str, ...] = (
    _G1_LEG_JOINT_NAMES
    + ("waist_yaw_joint",)
    + _G1_LEFT_ARM_23
    + _G1_RIGHT_ARM_23
)

# G1 29DoF：在 23DoF 基础上，增加腰部 roll/pitch 和双手腕 pitch/yaw。
# 注意要把新增的关节放回对应部位，而不是简单追加在最后。
G1_29DOF_JOINT_NAMES: tuple[str, ...] = (
    _G1_LEG_JOINT_NAMES
    + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    + _G1_LEFT_ARM_29
    + _G1_RIGHT_ARM_29
)

# G1 43DoF：29DoF 全身 + 左右手各 7 个主动关节。顺序与选定的
# g1_29dof_with_hand MJCF 一致，手指跟在各自手臂之后。
G1_43DOF_JOINT_NAMES: tuple[str, ...] = (
    _G1_LEG_JOINT_NAMES
    + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    + _G1_LEFT_ARM_29
    + _G1_LEFT_HAND_7
    + _G1_RIGHT_ARM_29
    + _G1_RIGHT_HAND_7
)

ISAAC_G1_37DOF_JOINT_NAMES: tuple[str, ...] = (
    _G1_LEG_JOINT_NAMES
    + ("torso_joint",)
    + _ISAAC_G1_LEFT_ARM_HAND
    + _ISAAC_G1_RIGHT_ARM_HAND
)


# ========== 二、通用张量重排工具 ==========

def _checked_names(names: Sequence[str], *, label: str) -> tuple[str, ...]:
    """检查关节名单，防止一个名字对应两列数字。

    输入：names 名字序列；label 用于错误提示。输出：无重复、非空的名字 tuple；不满足要求就报错。"""
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{label} 必须是关节名字序列，不能是一个字符串")
    result = tuple(names)
    if not result:
        raise ValueError(f"{label} 不能为空")
    if any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError(f"{label} 包含空名字或非字符串")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} 存在重复关节名")
    return result


def reorder_joint_tensor(
    values: torch.Tensor,
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> torch.Tensor:
    """按照关节名字重排最后一维；支持 [N]、[B,N]、[B,T,N]。"""
    source = _checked_names(source_names, label="source_names")
    target = _checked_names(target_names, label="target_names")

    # 顺序可以不同，但必须是同一批关节：不能把 K1 动作硬塞给 G1。
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise ValueError(f"关节名单不匹配：缺少={missing}，多出={extra}")

    if not isinstance(values, torch.Tensor):
        raise TypeError("values 必须是 PyTorch Tensor")
    if values.ndim == 0 or values.shape[-1] != len(source):
        raise ValueError(
            f"关节张量最后一维应为 {len(source)}，实际形状={tuple(values.shape)}"
        )

    # 例：原顺序 [膝, 髋, 踝]，新顺序 [髋, 膝, 踝] -> 取第 [1, 0, 2] 列。
    source_index = {name: i for i, name in enumerate(source)}
    indices = torch.tensor(
        [source_index[name] for name in target],
        dtype=torch.long,
        device=values.device,  # 输入在 GPU，索引也放在同一块 GPU 上。
    )
    return values.index_select(dim=-1, index=indices)


# ========== 三、任何机器人都可以使用的数据合同 ==========

@dataclass(frozen=True)
class RobotContract:
    """描述某一款机器人：它叫什么、根节点是谁、受控关节按什么顺序排列。"""

    name: str
    root_name: str
    joint_names: tuple[str, ...]
    root_qpos_dim: int = 7  # 默认是浮动根：x,y,z + qw,qx,qy,qz。

    def __post_init__(self) -> None:
        """创建合同后核对机器人名称、根名称及关节列表。

        输入：对象初始化的字段。输出：无；将 joint_names 固定为 tuple，只接受固定根0或浮动根7的简化位置定义。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("机器人名称不能为空")
        if not isinstance(self.root_name, str) or not self.root_name.strip():
            raise ValueError("根节点名称不能为空")
        names = _checked_names(self.joint_names, label="joint_names")
        # frozen=True 意味着之后不允许改；初始化时统一转成不可修改的 tuple。
        object.__setattr__(self, "joint_names", names)
        if self.root_qpos_dim not in (0, 7):
            raise ValueError("目前仅支持固定根(0)或浮动根(7)的数据合同")

    @property
    def dof(self) -> int:
        """获取该机器人的受控关节数。

        输入：无。输出：joint_names 的长度；不包含整个机器人在空间中的浮动根。"""
        return len(self.joint_names)

    @property
    def qpos_dim(self) -> int:
        """计算当前简化合同需要多少个位置数值。

        输入：无。输出：root_qpos_dim + dof。仅适用于每个受控关节一个角度；并不验证真实资产是否含被动/球关节。"""
        return self.root_qpos_dim + self.dof

    def check_joint_set(self, names: Sequence[str], *, source: str) -> None:
        """检查名单是否包含同一批关节，顺序可以不同。

        输入：names 实际名单；source 表示数据来源。输出：无；缺关节、重复或多关节立即报错。"""
        actual = _checked_names(names, label=f"{source}关节名单")
        if set(actual) != set(self.joint_names):
            missing = sorted(set(self.joint_names) - set(actual))
            extra = sorted(set(actual) - set(self.joint_names))
            raise ValueError(f"{source} 关节不匹配：缺少={missing}，多出={extra}")

    def check_exact_order(self, names: Sequence[str], *, source: str) -> None:
        """检查 HDF5 等是否已经按策略顺序保存，顺序不对立即报错。"""
        actual = _checked_names(names, label=f"{source}关节名单")
        self.check_joint_set(actual, source=source)
        if actual != self.joint_names:
            i = next(i for i, (a, b) in enumerate(zip(actual, self.joint_names)) if a != b)
            raise ValueError(
                f"{source} 关节顺序错误，第 {i} 列："
                f"应为 {self.joint_names[i]!r}，实际是 {actual[i]!r}"
            )

    def to_policy_order(
        self, joint_values: torch.Tensor, simulator_joint_names: Sequence[str]
    ) -> torch.Tensor:
        """把仿真器关节列顺序换成策略认识的顺序。

        输入：joint_values [...,N]、simulator_joint_names 的N个名字。输出：同形状张量；只改变最后一维列顺序，保留环境/时间轴与梯度。"""
        return reorder_joint_tensor(joint_values, simulator_joint_names, self.joint_names)

    def to_simulator_order(
        self, actions: torch.Tensor, simulator_joint_names: Sequence[str]
    ) -> torch.Tensor:
        """把策略输出换成仿真器要求的列顺序。

        输入：actions [...,N] 策略顺序动作；simulator_joint_names 目标名单。输出：同形状重排结果，不缩放、不裁剪动作。"""
        return reorder_joint_tensor(actions, self.joint_names, simulator_joint_names)


# ========== 四、机器人注册表：以后加新机器人不用改上面的通用函数 ==========

K1_CONTRACT = RobotContract("Booster K1 22DoF", "trunk", K1_JOINT_NAMES)
G1_23DOF_CONTRACT = RobotContract("Unitree G1 23DoF", "pelvis", G1_23DOF_JOINT_NAMES)
G1_29DOF_CONTRACT = RobotContract("Unitree G1 29DoF", "pelvis", G1_29DOF_JOINT_NAMES)
G1_43DOF_CONTRACT = RobotContract(
    "Unitree G1 29DoF + dual 7DoF hands", "pelvis", G1_43DOF_JOINT_NAMES
)
ISAAC_G1_37DOF_CONTRACT = RobotContract(
    "Isaac Lab Unitree G1 with dual 7DoF hands",
    "pelvis",
    ISAAC_G1_37DOF_JOINT_NAMES,
)

_ROBOT_REGISTRY: dict[str, RobotContract] = {}


def register_robot(robot_id: str, contract: RobotContract) -> None:
    """把一款机器人加入名单；robot_id 是配置文件中使用的唯一名字。"""
    if not isinstance(robot_id, str) or not robot_id.strip():
        raise ValueError("robot_id 不能为空")
    if not isinstance(contract, RobotContract):
        raise TypeError("contract 必须是 RobotContract")
    if robot_id in _ROBOT_REGISTRY:
        raise ValueError(f"机器人 {robot_id!r} 已注册，不能悄悄覆盖")
    _ROBOT_REGISTRY[robot_id] = contract


def get_robot_contract(robot_id: str) -> RobotContract:
    """根据配置中的机器人 ID 取出对应合同。

    输入：robot_id，如 k1_22dof、g1_29dof。输出：RobotContract；不存在就报错，不把含糊的 g1 自动猜成29关节。"""
    try:
        return _ROBOT_REGISTRY[robot_id]
    except (KeyError, TypeError):
        raise KeyError(
            f"未知机器人 {robot_id!r}，可选：{list_robot_contracts()}"
        ) from None


def list_robot_contracts() -> tuple[str, ...]:
    """查看目前已注册的机器人 ID。

    输入：无。输出：按名字排序的 tuple；注册只是数据接口，不代表物理环境已经配置。"""
    return tuple(sorted(_ROBOT_REGISTRY))


# G1 不使用含糊的 'g1'，因为 23DoF 和 29DoF 不能混用。
register_robot("k1_22dof", K1_CONTRACT)
register_robot("g1_23dof", G1_23DOF_CONTRACT)
register_robot("g1_29dof", G1_29DOF_CONTRACT)
register_robot("g1_43dof", G1_43DOF_CONTRACT)
register_robot("g1_37dof_isaac", ISAAC_G1_37DOF_CONTRACT)


# ========== 五、不依赖仿真器的最小自测 ==========
