"""共用的关节动作分布输出头。输入隐藏特征，输出原始均值和标准差。

动作缩放、关节目标裁剪、PD 与力矩限制属于环境层，不在这里修改随机变量。
"""
import math
import torch
from torch import nn

class GaussianActionHead(nn.Module):
    """机器人专用的 Actor 最后一层。log_std 是 PPO 会学习的可训练参数。"""

    def __init__(
        self, d_model: int, dof: int, init_std: float, *, check_values: bool = True
    ) -> None:
        """创建机器人专用的动作输出头。

        输入：d_model隐藏宽度、dof关节数N、init_std初始探索标准差、check_values检查开关。
        输出：无；创建均值线性层和N个可训练log_std。它们还不是PD角度目标。"""
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model 必须是正整数")
        if type(dof) is not int or dof <= 0:
            raise ValueError("dof 必须是正整数")
        if not isinstance(init_std, (float, int)) or isinstance(init_std, bool):
            raise TypeError("init_std 必须是正实数")
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std 必须是有限正数")
        if type(check_values) is not bool:
            raise TypeError("check_values 必须是 bool")
        self.check_values = check_values
        self.mean_head = nn.Linear(d_model, dof)
        # 初始动作均值接近 0，避免训练刚开始就发出很大的关节目标。
        nn.init.normal_(self.mean_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mean_head.bias)
        # 不设置人工 std cap：监测它是否漂移，后续在 PPO 算法层验证是否需要约束。
        self.log_std = nn.Parameter(torch.full((dof,), math.log(init_std)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 混合精度下，也要在 float32/64 中计算高斯分布和 PPO log_prob。
        """把隐藏特征转换为原始动作高斯分布参数。

        输入：x [...,D]；输出：(mean,std)，均为 [...,N]，标准差由exp(log_std)生成。
        不采样、不裁剪，PPO必须对之后采样的原始动作计算概率。"""
        mean = self.mean_head(x).to(self.log_std.dtype)
        std = self.log_std.exp().expand_as(mean)
        # 在计算概率之前拒绝非法 std/mean，而不是悄悄设为 0 或替换成常数。
        if self.check_values and (not torch.isfinite(std).all().item() or (std <= 0).any().item()):
            raise ValueError("policy std 非法：检查训练中 log_std 是否发生了膨胀/下溢")
        if self.check_values and not torch.isfinite(mean).all().item():
            raise ValueError("policy mean 出现 NaN/Inf")
        return mean, std

