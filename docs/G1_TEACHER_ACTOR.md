# G1 物理 Teacher → Actor 两阶段边界

本分支默认选择官方 HoloMotion v1.4 的 `g1_29dof`：12 个腿部、3 个腰部，以及左右
上肢各 7 个关节（肩 3、肘 1、腕 3）。这使 Actor 的关节顺序和官方 checkpoint 一致。
Isaac Lab 内置的 `G1_CFG` 是另一套 37DoF 机构，本分支不再将它作为默认资产。

代码不把 K1 的 22 维动作头、506 维 Actor 观测或 843 维 Critic 观测复制到 G1。状态、参考
动作和输出宽度均由 `RobotContract` 推导。`g1_43dof` 只保留为未来“官方 29DoF 身体 +
双手 14DoF”的扩展合同，不能直接加载官方 29维输出头，也不能写含糊的 `g1`。

## 阶段 1：训练物理 Teacher

配置：`configs/teachers/g1_29dof_holomotion_physics_teacher.yaml`。

模型使用 `PrivilegedTransformerTeacher` 的 V5 共享特权输入合同。Teacher 和 Critic 接收
完全相同宽度的 1172维张量：官方 HoloMotion G1 Critic 的 1046维状态/参考/误差信息，加上
126维滞后接触历史、物体槽和慢物理参数。接触后缀只能来自 `t-1` 或更早时刻；`t+1/t+4/t+8`
接触标志、法向力、切向力、滑移速度和摩擦利用率只作为监督标签，禁止同时进入对应输入。
二者拥有独立 Encoder 和梯度，Critic 的 value loss
不会修改 Teacher 的物理 latent。`G1TeacherActorModel(phase="teacher_training")` 会冻结
Actor/Critic，只开放在线 Teacher 参数；
每次优化器更新后调用 `update_teacher_ema()` 发布平滑权重。

Teacher 不对1172维做一次扁平压缩。1046维官方 Critic 块先经独立的
`LayerNorm + Linear + SiLU` 编码为128维，126维物理特权也经独立分支编码为128维；两路
拼接成256维后再融合回128维，才进入因果 Transformer，最终输出分组的64维物理 latent。这样保持
Teacher/Critic 的信息集合一致，同时保证较短的物理后缀拥有与状态/参考块相同的表示容量。
Critic 仍使用自己的1172维输入投影和 Transformer，不与 Teacher 共享任何参数。

64维 latent 不是一块无约束向量，而是四个独立输出头的有序拼接：接触8维、受力16维、
运动学16维、动力学24维。接触状态头只读取接触组；法向力、切向力和摩擦利用率只读取受力组；
滑移预测只读取运动学组；单步/多步状态转移读取运动学与动力学组；系统参数识别只读取动力学组。
Actor读取完整64维。当前监督包含滑移速度，Isaac提供可靠的切向方向标签后再扩展方向监督，不能
用未验证的刚体速度冒充接触点滑移方向。

Teacher 预留 4 个稳定接触槽，顺序为左脚、右脚、左腕、右腕；未来 G1 bridge 必须按此顺序
提供接触真值，不能仍沿用 K1 只有左右脚的两槽输入。

接触监督遵循严格时间边界：`(state_t, action_t, contact_{t-1})` 预测 `contact_{t+1}`；
4/8步接触头额外读取对应的已执行动作窗口，预测 `contact_{t+4}` 和 `contact_{t+8}`。
早期实验 checkpoint 曾把 `contact_t` 同时作为输入和标签，其接触重建指标属于目标泄漏，
只可保留作状态转移预训练，不得报告为接触力预测结果。

Teacher 的动作宽度为 29，normalized core 宽度为 `9 + 2*29 = 67`。checkpoint 同时保存
机器人 ID、两个模型配置、阶段和权重，防止把 K1/23/29/43DoF 权重静默装入 Isaac 模型。

## 阶段 2：冻结 Teacher，指导 Actor

配置：`configs/actors/g1_29dof_holomotion_teacher_guided.yaml`。

用 `G1TeacherActorModel.load(path, phase="teacher_guided_actor")` 读取阶段 1 checkpoint。
此时 Teacher 强制 `eval()`、`requires_grad_(False)`，其因果 latent 会 `detach()` 后进入 Actor
的独立物理分支；latent 来自阶段 1 保存的 EMA Teacher，而不是抖动的在线权重。Actor 同时读取 G1 本体状态、参考窗口和 Teacher latent，输出严格按
官方 `G1_29DOF_JOINT_NAMES` 排列的 29 维高斯动作分布。PPO 阶段更新 Actor 与独立 Critic，
不更新 Teacher；Critic 与 Teacher读取同一份特权历史，但只输出标量 value。

## Isaac Lab 边界

`unified_motion.envs.isaac_g1.IsaacG1Bridge` 读取 Isaac articulation 的真实关节名，并按名字
重排关节位置、速度、上一动作和 Actor 输出；它不假设 USD 列顺序。桥还要求显式提供左脚、
右脚、左腕、右腕四个 contact body，避免使用猜测的 link 名称。

模型和 checkpoint 工具仍可在普通 Python 中导入；articulation 必须由官方 HoloMotion 的
`29dof_training_isaaclab.yaml` 和 `SceneFunctions.build_robot_config` 创建，禁止回退到 Isaac
内置 `G1_CFG`。后续 G1 不再建设 mjlab bridge。当前仓库的精简 upstream snapshot 不复制
大型权重和机器人资产；本机完整官方工程 `/home/tcl/Desktop/start/HoloMotion` 已包含 URDF、
USD、`model_14000` 权重及既有 Isaac Lab 评测结果。仍需在本分支正式训练入口中完成 body 名称核验，
并把四个 contact body 接到 PhysX 原始接触采样器，之后才能进行正式 Teacher 数据采集。

## 官方权重

`load_original_holomotion_algorithm(...)` 会从官方运行目录的 `config.yaml` 读取
`actions_dim`、`dof_obs_size` 和 `num_bodies`，对 G1 自动建立 29维动作头、604维 Actor
观测和 1046维 Critic 观测，然后严格加载 actor/critic safetensors。旧的
`load_original_holomotion_k1_algorithm(...)` 名称仅作为兼容入口，内部同样按配置推导形态。

无需联网的真实权重冒烟测试：

```bash
/home/tcl/miniconda3/envs/holomotion_train/bin/python \
  tools/smoke_holomotion_g1_official.py
```

该命令会严格加载本机官方 `model_14000` 的 actor/critic safetensors，并执行一次 604→29
Actor 前向和 1046→1 Critic 前向；只有形状正确且输出全部有限才报告 `status=passed`。
