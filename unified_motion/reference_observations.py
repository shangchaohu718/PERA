"""参考观测：将重定向后的机器人目标转成相对于当前身体的特征。

统一测评：python tools/evaluate.py
依赖：Python >= 3.10、PyTorch、同目录的 robot_contract.py。

接口版本 1。不是 HoloMotion 官方观测的逐项复刻，也不能直接加载其权重。
输入必须已重定向为目标机器人动作；不在此读取 HDF5、插值、计算 FK 或训练。
坐标：世界系 z 向上；机身系 x 前/y 左/z 上；四元数 wxyz 表示机身到世界。
单位：米、秒、弧度、米/秒、弧度/秒。当前状态与参考必须使用同一世界系。
支持 [B,H,...] 和 [B,T,H,...]：B 是环境，T 是历史时间，H 是参考时间点。
H 可以含当前时刻；每个窗口至少一个有效点，补位只能在末尾，不跨 clip。
这是完整的“参考特征构建模块”，不是完整的机器人训练系统。
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

if __package__:
    from .robot_contract import RobotContract, get_robot_contract
else:
    from robot_contract import RobotContract, get_robot_contract


@dataclass
class ReferenceWindow:
    """一个已由数据采样器选好的参考窗口，不是整个动作数据集。

    joint_names 对应 joint_pos / joint_vel 的最后一维。
    root_name 必须是合同中的根部件，不能把躯干、骨盆等原点混用。
    速度是参考的目标速度，不是当前机器人的速度，也不是速度误差。
    无效位置可以填有限的零；不能用 NaN/Inf。有效四元数必须近似单位长度。
    """

    root_name: str
    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor          # [..., H, N]：机器人参考关节角
    joint_vel: torch.Tensor          # [..., H, N]：机器人参考关节速度
    root_pos_w: torch.Tensor         # [..., H, 3]：参考根部件世界位置
    root_quat_wxyz: torch.Tensor     # [..., H, 4]：参考根部件世界朝向
    root_lin_vel_w: torch.Tensor     # [..., H, 3]：根部件原点的世界线速度
    root_ang_vel_w: torch.Tensor     # [..., H, 3]：参考根部件世界角速度
    time_offsets_s: torch.Tensor     # [..., H]：距当前状态的时间，单位秒
    valid_mask: torch.Tensor         # [..., H]：bool，True 表示有真实参考


@dataclass
class ReferenceObservation:
    """features 是目标特征，还不是神经网络编码后的 token。

    mask 约定 True=有效。后续 Attention 必须按其 API 转换 mask 的含义。
    """

    features: torch.Tensor           # [..., H, 16 + 2N]
    valid_mask: torch.Tensor         # [..., H]

    def flatten_for_mlp(self) -> torch.Tensor:
        """每个时间点补上有效标志，再展平；MLP 使用时需要固定 H。"""
        with_mask = torch.cat(
            (self.features, self.valid_mask.unsqueeze(-1).to(self.features.dtype)),
            dim=-1,
        )
        return with_mask.flatten(start_dim=-2)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """wxyz -> 3x3 旋转矩阵。调用前已检查单位长度；此处消除小数误差。

    矩阵把机身向量变到世界系；转置后才是世界系变到机身系。
    q 与 -q 得到同一个矩阵，避免同一朝向仅因四元数变号而改变特征。
    """
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ), dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


class ReferenceObservationBuilder(nn.Module):
    """只有确定性的数据转换，没有可训练参数，也不改变机器人动作。

    default_joint_pos 必须与 ObservationBuilder 使用同一份策略顺序默认姿态。
    物理数据使用 float32/float64；网络混合精度在后面的 Encoder 内处理。
    check_values=True 做严格数值检查，会在 GPU 上同步；False 仅适用于
    已验证的数据管线，关闭时仍检查形状、类型、设备、关节和根名称。
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        robot: RobotContract,
        reference_joint_names: Sequence[str],
        default_joint_pos: torch.Tensor,
        *,
        check_values: bool = True,
    ) -> None:
        """准备把机器人动作窗口转成参考特征。

        输入：robot 合同、reference_joint_names 动作文件列顺序、default_joint_pos [N] 默认角；check_values 控制数值检查。
        输出：无；缓存重排索引，必须与身体 ObservationBuilder 使用同一默认姿态。"""
        super().__init__()
        if not isinstance(robot, RobotContract):
            raise TypeError("robot 必须是 RobotContract")
        if not isinstance(check_values, bool):
            raise TypeError("check_values 必须是 bool")
        robot.check_joint_set(reference_joint_names, source="参考动作")
        self.robot = robot
        self.reference_joint_names = tuple(reference_joint_names)
        self.check_values = check_values
        if not isinstance(default_joint_pos, torch.Tensor):
            raise TypeError("default_joint_pos 必须是 Tensor")
        if default_joint_pos.shape != (robot.dof,):
            raise ValueError(f"默认关节角必须是 [{robot.dof}]")
        if default_joint_pos.dtype not in (torch.float32, torch.float64):
            raise TypeError("默认关节角必须是 float32 或 float64")
        if not torch.isfinite(default_joint_pos).all().item():
            raise ValueError("默认关节角包含 NaN 或 Inf")
        self.register_buffer("default_joint_pos", default_joint_pos.detach().clone())
        # 只在创建时计算重排表；每一步不再重新按名字查找。
        indices = robot.to_policy_order(
            torch.arange(robot.dof, device=default_joint_pos.device),
            self.reference_joint_names,
        )
        self.register_buffer("policy_indices", indices, persistent=False)

    @property
    def feature_dim(self) -> int:
        """时间1 + 根位置3 + 朝向6 + 根线/角速度6 + 关节位置/速度2N。"""
        return 16 + 2 * self.robot.dof

    @property
    def feature_slices(self) -> dict[str, slice]:
        """给每段参考特征标上名字，避免调用处到处写固定下标。

        输入：无。输出：{特征名称: slice}；例如 root_pos_b 取出相对于当前机身的位置三维。"""
        n = self.robot.dof
        return {
            "time_offsets_s": slice(0, 1),
            "root_pos_b": slice(1, 4),
            "root_rot6d_b": slice(4, 10),
            "root_lin_vel_b": slice(10, 13),
            "root_ang_vel_b": slice(13, 16),
            "joint_pos_rel": slice(16, 16+n),
            "joint_vel": slice(16+n, 16+2*n),
        }

    def get_extra_state(self) -> dict:
        """随 state_dict 保存接口信息，拒绝默默加载另一套数据定义。"""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "root_name": self.robot.root_name,
            "joint_names": self.robot.joint_names,
            "reference_joint_names": self.reference_joint_names,
            "frame": "current_root_full_rotation",
            "rotation": "matrix_first_two_columns",
        }

    def set_extra_state(self, state: dict) -> None:
        """读取 checkpoint 时检查数据定义有没有变化。

        输入：保存的结构字典。输出：无；机器人、关节顺序、坐标或版本不同就拒绝，不按相同数组长度猜兼容。"""
        if state != self.get_extra_state():
            raise ValueError("参考观测 checkpoint 的机器人或接口版本不匹配")

    def _check_tensor(
        self, name: str, value: torch.Tensor, shape: tuple[int, ...],
        *, boolean: bool = False,
    ) -> None:
        """核对一个输入是否符合本模块的数据合同。

        输入：name 字段名、value 张量、shape 期望形状、boolean 是否要求bool。输出：无；同时检查设备、精度和可选的有限值。"""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} 必须是 Tensor")
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} 应为 {tuple(shape)}，实际 {tuple(value.shape)}")
        if value.device != self.default_joint_pos.device:
            raise ValueError(f"{name} 与 builder 不在同一设备")
        expected_dtype = torch.bool if boolean else self.default_joint_pos.dtype
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} 应使用 {expected_dtype}")
        if self.check_values and not boolean and not torch.isfinite(value).all().item():
            raise ValueError(f"{name} 包含 NaN 或 Inf")

    def forward(
        self,
        reference: ReferenceWindow,
        *,
        current_root_pos_w: torch.Tensor,
        current_root_quat_wxyz: torch.Tensor,
    ) -> ReferenceObservation:
        """当前根姿态是 [...,3/4]，不带 H；每个窗口都相对其当前状态。

        current_root_* 必须对应 robot.root_name，且与身体观测同一时刻。
        本模块不检查动力学可行性、不产生接触力标签、不补造缺失参考。
        """
        if not isinstance(reference, ReferenceWindow):
            raise TypeError("reference 必须是 ReferenceWindow")
        if reference.root_name != self.robot.root_name:
            raise ValueError("参考根部件与机器人合同不一致")
        if tuple(reference.joint_names) != self.reference_joint_names:
            raise ValueError("参考关节顺序变化，请检查数据或重新创建 builder")
        q = reference.joint_pos
        if not isinstance(q, torch.Tensor):
            raise TypeError("joint_pos 必须是 Tensor")
        if q.ndim not in (3, 4) or any(size == 0 for size in q.shape):
            raise ValueError("参考关节角必须是非空 [B,H,N] 或 [B,T,H,N]")
        if self.default_joint_pos.dtype not in (torch.float32, torch.float64):
            raise TypeError("参考几何计算必须保持 float32/float64")
        leading, h = q.shape[:-2], q.shape[-2]
        window_shape = (*leading, h)
        fields = (
            ("joint_pos", self.robot.dof), ("joint_vel", self.robot.dof),
            ("root_pos_w", 3), ("root_quat_wxyz", 4),
            ("root_lin_vel_w", 3), ("root_ang_vel_w", 3),
        )
        for name, width in fields:
            self._check_tensor(name, getattr(reference, name), (*window_shape, width))
        self._check_tensor("time_offsets_s", reference.time_offsets_s, window_shape)
        self._check_tensor("valid_mask", reference.valid_mask, window_shape, boolean=True)
        self._check_tensor("current_root_pos_w", current_root_pos_w, (*leading, 3))
        self._check_tensor("current_root_quat_wxyz", current_root_quat_wxyz, (*leading, 4))

        valid = reference.valid_mask
        dt = reference.time_offsets_s
        if self.check_values:
            if not valid.any(dim=-1).all().item():
                raise ValueError("每个窗口至少需要一个有效参考；无参考任务用独立目标接口")
            if (valid[..., 1:] & ~valid[..., :-1]).any().item():
                raise ValueError("无效补位只能在窗口末尾，不能中间挖洞")
            if ((dt < 0) & valid).any().item():
                raise ValueError("参考时间必须相对当前时刻且非负")
            if ((dt[..., 1:] <= dt[..., :-1]) & valid[..., 1:]).any().item():
                raise ValueError("有效参考时间必须严格递增")
            for name, quat, mask in (
                ("current_root_quat_wxyz", current_root_quat_wxyz, None),
                ("root_quat_wxyz", reference.root_quat_wxyz, valid),
            ):
                error = (torch.linalg.vector_norm(quat, dim=-1) - 1).abs() > 1e-3
                if (error if mask is None else error & mask).any().item():
                    raise ValueError(f"{name} 的有效四元数必须近似单位长度")

        # 不让外部 autocast 降低几何计算精度；本模块没有神经网络权重。
        with torch.autocast(device_type=q.device.type, enabled=False):
            mask = valid.unsqueeze(-1)
            root_origin = current_root_pos_w.unsqueeze(-2)
            # 先清理补位，避免无效的大数进入减法、旋转运算。
            ref_pos = torch.where(mask, reference.root_pos_w, root_origin)
            identity = q.new_tensor([1.0, 0.0, 0.0, 0.0])
            ref_quat = torch.where(mask, reference.root_quat_wxyz, identity)
            world_to_body = _quat_to_matrix(current_root_quat_wxyz).transpose(-1, -2)
            world_to_body = world_to_body.unsqueeze(-3)  # 增加 H 轴，用于批量运算。

            def rotate(vector: torch.Tensor) -> torch.Tensor:
                return (world_to_body @ vector.unsqueeze(-1)).squeeze(-1)

            relative_rotation = world_to_body @ _quat_to_matrix(ref_quat)
            # 取旋转矩阵前两列，每列3个数字；这是朝向，不是6个关节角。
            rotation6d = relative_rotation[..., :, :2].transpose(-1, -2).flatten(-2)
            ref_q = reference.joint_pos.index_select(-1, self.policy_indices)
            ref_v = reference.joint_vel.index_select(-1, self.policy_indices)
            features = torch.cat(
                (
                    torch.where(mask, dt.unsqueeze(-1), 0.0),
                    rotate(ref_pos - root_origin),
                    rotation6d,
                    rotate(torch.where(mask, reference.root_lin_vel_w, 0.0)),
                    rotate(torch.where(mask, reference.root_ang_vel_w, 0.0)),
                    torch.where(mask, ref_q, self.default_joint_pos) - self.default_joint_pos,
                    torch.where(mask, ref_v, 0.0),
                ), dim=-1,
            )
            features = torch.where(mask, features, 0.0)
        if self.check_values and not torch.isfinite(features).all().item():
            raise ValueError("构建后的参考特征出现 NaN 或 Inf")
        return ReferenceObservation(features, valid.clone())




