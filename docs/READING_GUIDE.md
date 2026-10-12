# 先看什么：按数据流读，不按第几号文件读

## 1. 先看 api.py 和 train.py

`api.py`回答“正式支持哪些类，怎么选MLP/Dense/MoE”。
`train.py`回答“谁先调用谁，模型、动作库、环境怎样放到同一设备”。
这里不需要先看Attention的公式。

## 2. 一个控制步到底发生什么

| 次序 | 函数 | 大白话 |
|---|---|---|
| 1 | task.read() | 拿到现在的真实身体、接触和物体信息 |
| 2 | dynamics.step() | 用过去预测与这次真实结果的差异，更新物理记忆 |
| 3 | task.goal() | 取得现在已知的任务目标，不拿未来真实结果冒充目标 |
| 4 | policy / policy.step() | 依据物理估计和目标产生原始动作 |
| 5 | dynamics.predict_next() | 动作已确定后，预测执行结果 |
| 6 | task.step(action) | 真正执行，得到奖励及结束原因 |
| 7 | task.read() / policy.value() | 在reset前读取实际下一状态，计算需要的bootstrap |
| 8 | collector保存快照 | 保存的动作、概率、目标、物理结果不能被下一步覆盖 |
| 9 | task.reset(done_ids) | 只重置结束的环境 |

注意：预测器不能先看到第6步的真实结果，再假装第5步已经预测到了。
`episode_start[t]`是第t次输入已经属于新回合；`terminated[t]`是执行第t次动作后结束。

## 3. 再看 models/physics.py

它把物理研究的相关内容集中在一起：
- PhysicsObservation：现在发生了什么，数据容器。
- ControlGoal：希望发生什么，目标容器。
- PhysicsEncoder：把身体、接触和物体数字变成特征。
- DynamicsModel：记住历史、预测下一时刻。
- PhysicsMLPActorCritic：最简单的物理条件化控制基线。

`context`是当前物理特征；`latent`是经过历史记忆得到的物理隐变量；
它们不等于已经“识别出真实摩擦系数”，只能通过后续实验判断是否有用。

## 4. 时序策略只多读一层

`models/physics_policy.py`把同样的物理信息交给Dense/MoE主干。
其下复用已有的 `encoder → reference_fusion → attention → FFN/MoE`。
标准Dense的FFN不是另一套PPO；MoE也不是另外一种物理引擎。

MLP没有Transformer KV，但仍然可以使用独立DynamicsModel的GRU历史。
因此公平对照时不是“MLP完全无记忆”和“Transformer有记忆”。

## 5. 理解训练的两个更新器

`PhysicsSequencePPO`只更新policy。它读取采样时已经保存的context/latent，不反传到独立动力学。
`DynamicsUpdater`用真实 `下一状态` 监督预测器，不通过PPO的动作reward猜预测误差。

两者都是神经网络，但训练目标不同。预测loss下降不自动代表控制更好，后续需要独立消融。

## 6. 数据形状备忘

- B：并行环境数。
- T：真实历史时刻数。
- N：关节数，K1=22，G1版本为23/29。
- H：同一时刻已知的多个参考目标。
- C：固定编号的接触测量槽位，不是“本帧接触数量”。没接触但该脚存在时槽位mask仍可为True。
- M：固定编号的可观察物体槽位。
- D/P/Z：网络内部特征宽度，不是关节数量。

状态数组采用 `[B,T,...]`；GAE的reward/terminated等用 `[T,B]`。
转换只在统一训练器里做，不要在不同模块随意交换轴。

## 7. 当前物理信息的实际含义

K1PhysicsBridge：两个脚-地面槽位，每只脚取maxforce代表点，不是前/后脚掌所有受力点，更不是全身接触图。
物体位置/速度可读取，但需要场景已注册该物体，不负责检测物体或规划操作。
对象速度是世界相对速度旋转到机身系，不等于对“机身系相对位置”直接作时间求导。

参考跟踪奖励默认只开关节与根部件；身体/鞋底奖励必须有同名参考数据。
不能用ankle中心冒充sole/toe/heel，不能把没有标注的阶段当作锁脚目标。


## R18物理接通阶段
先读envs/mjlab_k1.py里的k1_control_profile：一张表定义真实交给执行器的参数。
再读docs/PHYSICS_ACCEPTANCE.md，最后看envs/physics_validation.py。
不要把这里只做形状/字段的mock测试当成物理理解或零样本能力。
