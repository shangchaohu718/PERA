"""多机器人身体/参考编码器：把不同宽度的输入转成同一隐藏宽度。

统一测评（在项目外层）：python tools/evaluate.py
输入由第 02/03 个文件生成；前三个文件不需要修改。
本模块是自己的实现，不是 HoloMotion 官方源码或预训练权重兼容层。

约定：
- 每次调用只编码一种机器人；不同机器人分组调用，再交给共享主干。
- 每种机器人各有身体/参考两个小 MLP，输出相同宽度 D。
- 保留时间轴 T 和参考轴 H；不进行 Attention、MoE、时间平均或动作输出。
- 输入必须已经按策略关节顺序、正确坐标系/单位整理；同尺寸语义错误无法靠张量识别。
- 每个窗口至少一帧有效参考；无参考的稀疏目标任务不在本模块范围内。
- 模型/输入保持 float32 或 float64；混合精度用外部 torch.autocast。
- 不在线更新观测统计，不使用 Dropout/BatchNorm；LayerNorm 不是物理量标准化器。
- 所有机器人分支在创建优化器前建立；新增机器人需要新的分支训练和显式权重迁移。
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from ..robot_contract import RobotContract, get_robot_contract
from ..reference_observations import ReferenceObservation, ReferenceObservationBuilder


@dataclass
class EncodedMotion:
    """编码结果；token 就是一组通过训练学到的特征数字。"""

    robot_id: str
    state_token: torch.Tensor          # [B,D] 或 [B,T,D]
    reference_tokens: torch.Tensor     # [B,H,D] 或 [B,T,H,D]
    reference_mask: torch.Tensor       # [B,H] 或 [B,T,H]，True=有效

    def flatten_for_mlp(self) -> torch.Tensor:
        """保留所有参考点，拼成一排供后续 MLP 使用；下游 MLP 必须固定 H。

        输出宽度 D + H*(D+1)，其中每个 +1 是参考有效标记。
        这不是完整 Actor，也不把 T 历史轴展平。
        """
        refs = torch.cat(
            (self.reference_tokens,
             self.reference_mask.unsqueeze(-1).to(self.reference_tokens.dtype)),
            dim=-1,
        )
        return torch.cat((self.state_token, refs.flatten(start_dim=-2)), dim=-1)


def _make_mlp(input_dim: int, hidden_dim: int, d_model: int) -> nn.Sequential:
    """同一网络逐个处理时间点，不混合环境 B、历史 T、参考 H。"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),  # 学习怎样组合原始数字。
        nn.SiLU(),                        # 非线性，不是只做线性缩放。
        nn.Linear(hidden_dim, d_model),   # 统一到后续主干要求的宽度 D。
        nn.LayerNorm(d_model),            # 只规范当前 token 的隐藏特征。
    )


class MultiRobotEncoder(nn.Module):
    """不同身体使用小型输入分支，后续可以连接同一个 Transformer/MoE。

    宽度相同只解决接口，不保证 K1/G1 语义自动对齐，更不代表已经学会运动。
    check_values=True 用于调试；其中 .item() 在 GPU 上会造成同步。
    关闭数值检查只适用于已验证的数据管线，形状/设备/类型检查仍保留。
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        robots: Mapping[str, RobotContract],
        *,
        d_model: int = 256,
        hidden_dim: int = 512,
        check_values: bool = True,
    ) -> None:
        """为每款机器人创建身体和参考两个小 MLP。

        输入：robots={ID:合同}；d_model 是统一输出宽度D；hidden_dim 是中间层宽度；check_values 控制输入检查。
        输出：无；不同机器人的输入层独立，后面的 Transformer 可以共享。须在创建优化器前注册全部分支。"""
        super().__init__()
        if not isinstance(robots, Mapping) or not robots:
            raise ValueError("robots 必须是非空的 {机器人ID: RobotContract} 映射")
        for robot_id, robot in robots.items():
            if not isinstance(robot_id, str) or not robot_id.strip():
                raise ValueError("机器人 ID 必须是非空字符串")
            if not isinstance(robot, RobotContract):
                raise TypeError(f"{robot_id} 对应的配置必须是 RobotContract")
        for name, value in (("d_model", d_model), ("hidden_dim", hidden_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数，不能是 bool")
        if d_model < 2:
            raise ValueError("d_model 至少为2，避免一维 LayerNorm 丢失输入信息")
        if not isinstance(check_values, bool):
            raise TypeError("check_values 必须是 bool")

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.check_values = check_values
        # 固定排序，让字典传入顺序不同也得到相同 checkpoint 结构。
        self._robots = {key: robots[key] for key in sorted(robots)}
        self.robot_ids = tuple(self._robots)
        # 使用内部安全键，避免机器人 ID 中的点号与 PyTorch 模块路径冲突。
        self._keys = {key: f"robot_{i}" for i, key in enumerate(self.robot_ids)}
        self.state_encoders = nn.ModuleDict()
        self.reference_encoders = nn.ModuleDict()
        for robot_id in self.robot_ids:
            obs_dim, ref_dim = self.input_dims(robot_id)
            key = self._keys[robot_id]
            self.state_encoders[key] = _make_mlp(obs_dim, hidden_dim, d_model)
            self.reference_encoders[key] = _make_mlp(ref_dim, hidden_dim, d_model)

    def input_dims(self, robot_id: str) -> tuple[int, int]:
        """与第 02/03 个模块的 v1 特征排列保持一致；测试会交叉检查。"""
        if not isinstance(robot_id, str) or robot_id not in self._robots:
            raise KeyError(f"未知机器人 {robot_id!r}，可选 {self.robot_ids}")
        n = self._robots[robot_id].dof
        return 6 + 3*n, 16 + 2*n

    def get_extra_state(self) -> dict:
        """保存编码器的输入含义，避免同样维数却关节顺序不同。

        输入：无。输出：包含机器人合同、D和接口版本的字典，随权重一起保存。"""
        return {
            "encoder_schema": self.SCHEMA_VERSION,
            "observation_schema": "body_v1_angular_gravity_q_qd_action",
            "reference_schema": ReferenceObservationBuilder.SCHEMA_VERSION,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "robots": {
                key: (robot.root_name, robot.root_qpos_dim, robot.joint_names)
                for key, robot in self._robots.items()
            },
        }

    def set_extra_state(self, state: dict) -> None:
        """加载模型时核对机器人与编码方式。

        输入：checkpoint 的定义字典。输出：无；与当前配置不一致就报错，不暗中丢弃不兼容参数。"""
        if state != self.get_extra_state():
            raise ValueError("编码器 checkpoint 的机器人、关节顺序或接口版本不匹配")

    def _check_tensor(
        self, name: str, value: torch.Tensor, shape: tuple[int, ...],
        parameter: torch.Tensor, *, boolean: bool = False,
    ) -> None:
        """检查张量形状、设备、类型和有限值。

        输入：name字段名、value数据、shape目标大小、parameter参照网络权重、boolean类型标志。输出：无；用于阻止不同环境/时间的数据被广播拼错。"""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} 必须是 Tensor")
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} 应为 {tuple(shape)}，实际 {tuple(value.shape)}")
        if value.device != parameter.device:
            raise ValueError(f"{name} 与编码器必须在同一设备")
        expected_dtype = torch.bool if boolean else parameter.dtype
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} 应使用 {expected_dtype}")
        if self.check_values and not boolean and not torch.isfinite(value).all().item():
            raise ValueError(f"{name} 包含 NaN 或 Inf")

    def forward(
        self,
        robot_id: str,
        observation: torch.Tensor,
        reference: ReferenceObservation,
    ) -> EncodedMotion:
        """接收已整理的观测，返回身体 token 与逐参考点 token。

        本模块不读取未来真实机器人状态；未来目标仅来自已知参考轨迹。
        T 轴不在这里计算注意力，因此不是已经实现了时序记忆。
        """
        obs_dim, ref_dim = self.input_dims(robot_id)
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation 必须是 Tensor")
        if observation.ndim not in (2, 3) or any(s == 0 for s in observation.shape):
            raise ValueError("observation 必须是非空 [B,O] 或 [B,T,O]")
        if not isinstance(reference, ReferenceObservation):
            raise TypeError("reference 必须是 ReferenceObservation")
        features = reference.features
        if not isinstance(features, torch.Tensor):
            raise TypeError("reference.features 必须是 Tensor")
        if features.ndim != observation.ndim + 1 or features.shape[-2] <= 0:
            raise ValueError("参考应比身体观测多一个非空的 H 参考轴")

        leading, h = observation.shape[:-1], features.shape[-2]
        key = self._keys[robot_id]
        state_encoder = self.state_encoders[key]
        reference_encoder = self.reference_encoders[key]
        parameter = state_encoder[0].weight
        if parameter.dtype not in (torch.float32, torch.float64):
            raise TypeError("模型权重保持 float32/64；混合精度请用 torch.autocast")
        self._check_tensor("observation", observation, (*leading, obs_dim), parameter)
        self._check_tensor("reference.features", features, (*leading, h, ref_dim), parameter)
        valid = reference.valid_mask
        self._check_tensor("reference.valid_mask", valid, (*leading, h), parameter, boolean=True)
        if self.check_values:
            if not valid.any(dim=-1).all().item():
                raise ValueError("每个窗口至少需要一个有效参考点")
            if (valid[..., 1:] & ~valid[..., :-1]).any().item():
                raise ValueError("无效参考只能在末尾补位，不能中间挖洞")

        # 身体特征只经过自己的小网络；不与其他机器人或历史帧混在一起。
        state_token = state_encoder(observation)
        mask = valid.unsqueeze(-1)
        # 编码前清零：让补位中的任意有限垃圾数值无法影响计算和梯度。
        safe_features = torch.where(mask, features, 0.0)
        reference_tokens = reference_encoder(safe_features)
        # 编码后再清零：Linear 的 bias 等可能让“零输入”产生非零输出。
        reference_tokens = torch.where(mask, reference_tokens, 0.0)
        if self.check_values:
            for name, value in (("state_token", state_token),
                                ("reference_tokens", reference_tokens)):
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"{name} 出现 NaN 或 Inf")
        return EncodedMotion(robot_id, state_token, reference_tokens, valid.clone())




