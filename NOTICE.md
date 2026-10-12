# 数据、依赖和研究声明

本包基于本次对话提供的自研代码修订，未附带HoloMotion官方权重、LAFAN/AMASS数据、Booster/Unitree模型、字体或第三方二进制。
外部库、机器人模型和动作数据需要分别遵守各自许可；免费公开不自动允许任意再分发。

这不是HoloMotion官方实现或兼容权重发布，也没有实际机器人训练结果。
代码中的TensorTask/FakeEnv只用于接口与数值测试，严禁用它们的通过结果作为物理性能证据。

实现参考的公开API资料：
- PyTorch SDPA / Module / distributions 官方文档：https://docs.pytorch.org/
- Gymnasium 时间限制与bootstrap说明：https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- mjlab 官方项目：https://github.com/mujocolab/mjlab
- Booster官方资产：https://github.com/BoosterRobotics/booster_assets

CPU依赖按报告实测；可选mjlab1.6.0依赖是待验证候选，不是当前环境成功安装的锁文件。


## R21数学来源
LRU线性递推、稳定参数化与初始化依据 Orvieto et al., ICML 2023,
Resurrecting Recurrent Neural Networks for Long Sequences, Appendix A.
https://proceedings.mlr.press/v202/orvieto23a.html
本轮PyTorch实现为本项目编写，没有复制第三方GitHub实现或携带其权重。
