# API 阅读索引（从源码生成）

正式入口 api.py；本索引帮助查找函数，不替代正文中的坐标/时间约定。

## api.py

### build_policy

创建六种策略之一，统一后面的采样与 PPO 调用方式。

输入：
    robots：{机器人ID: 关节合同}。MLP 当前一次只注册一款机器人；
            Dense/MoE 可注册多款，但每次采样批次仍只包含一种机器人。
    backbone：'mlp'、'transformer'（别名'dense'）、'moe'、'gru'、'lstm'、'lru'。
    d_model：时序策略的隐藏宽度，当前工厂固定 Q=4头/KV=2头，宽度须能被8整除。
    hidden_dim：输入编码器/MLP 的中间宽度，不是关节数量。
    n_layers/context_len：时序层数/每层历史窗口；MLP 不使用这两个设置。
    num_experts：MoE 专家总数，固定每个 token 选择2个，至少需要2个专家。
    physics_width/latent_width：必须与 DynamicsModel 输出一致。
    init_std：原始动作高斯分布的初始标准差；不是关节角范围。
输出：PhysicsMLPActorCritic 或 PhysicsConditionedActorCritic，可交给统一 Collector/PPO。
注意：这只是创建网络，随机权重没有运动技能；仿真动作缩放在环境中另行定义。

### 同一 K1 任务切换 HoloMotion / 自定义 Actor

`k1_policy_backends.build_k1_networks` 按保存的网络类型重建七种 Actor；
`k1_training.create_k1_trainer` 接入同一 PPO 和可选 Physics 残差。
`k1_policy_runtime.load_k1_checkpoint`、`resume_k1_trainer`、`frozen_k1_policy` 分别用于加载、
恢复、冻结推理，`evaluate_k1_policy` 运行有界评测。
此路线接收完整506/843维 K1观测，旧 `build_policy` 的 `ControlGoal` 接口继续保留。
详见 [Actor切换与权重兼容](K1_ACTOR_SWITCHING.md)。

## models/physics.py

### PhysicsObservation

机器人当前可观察到的物理交互信息。

功能：让 MLP、MoE 和独立动力学模型读取同一种物理数据格式。
输入：
    body_obs: [B, 6+3N] 或 [B,T,6+3N]，来自旧 ObservationBuilder。
        前6列是身体角速度和重力方向；接着分别是 N 个关节角偏移、
        N 个关节速度和 N 个上一步原始动作（策略关节顺序）。
    base_lin_vel_b: [...,3]，身体坐标下的机身线速度；来自估计器。
    contact_features: [...,C,12]，C 个稳定编号的接触测量点：
        0:3  测量点相对根部件的位置，单位米，身体坐标。
        3:6  该测量点相对接触对象的速度，单位米/秒，身体坐标。
        6:9  接触面的法线单位方向，身体坐标；未接触时可为0。
        9    估计法向接触力，牛顿；不可测量时填0。
        10   是否接触：1接触，0未接触（可使用实机接触估计器）。
        11   力是否有测量/估计来源：1有，0没有。
    contact_mask: [...,C]，True=这个槽位存在，不等于正在接触。
    object_features: [...,M,16]，M 个稳定编号的物体：
        0:3   物体相对根部件的位置，米，身体坐标。
        3:7   物体相对于根部件的朝向四元数，wxyz。
        7:10  物体相对机身的线速度，米/秒，身体坐标。
        10:13 物体相对机身的角速度，弧度/秒，身体坐标。
        13:16 物体外形尺寸（长宽高），米。
    object_mask: [...,M]，True=该槽位有实际可见/可估计物体。
    core_valid: [...,9+2N] 布尔张量，依次对应线速度、角速度、重力、关节角、关节速度。
        省略时默认全部可测；不可测分量必须为 False，不能靠填0表示缺失。
        deployable 适配器将根线速度前三列标为 False；特权监督目标独立保留真实有效性。

输出：本对象自己不做网络计算；交给 PhysicsEncoder / DynamicsModel。

重要：同一环境的 C/M 槽位身份必须在一次回合中保持稳定。
暂时没有接触点或物体时，C/M 可以为0；不要把不存在的数据伪造为真。
这不是相机识别模块，外部适配器负责把真实传感器结果整理到这里。

新 Dynamics 按 core_valid 屏蔽输入占位值、预测创新和目标损失，并编码可测性。
完整历史默认在 CPU 按64帧分块编码，活跃原始张量预算512 MiB、步数上限16384。
超预算在世界推进前报错；预算不含其他 rollout 引用、激活和进程开销。
参见 [修复合同与运行时配置](ALGORITHM_CORRECTIONS_20261012.md)。

**from_k1_bridge**

功能：把第11模块 K1 mjlab 桥里的真实机身状态包装成物理观测。

输入：
    bridge：已经 bridge.reset() 的 K1MJLabBridge。
    contact_features/contact_mask：必须来自真正的接触点检测或估计。
    object_features/object_mask：必须来自外部物体状态估计；没有物体
        时明确传 [B,0,16] 和 [B,0] 空张量。
输出：PhysicsObservation，身体状态与线速度从 mjlab 当前帧读取。
注意：目前的第11模块没有完整鞋底/物体传感器配置，这里不会伪造接触。
      这些输入必须由 mjlab 的传感器适配器真正产生。

**at**

功能：从一段 [B,T,...] 的历史里，取出第 t 帧 [B,...]。

输入：t 为从0开始的时间序号。
输出：该时间点的 PhysicsObservation，不会改变原始数据。
用途：可以用同一批数据做整段训练或逐帧推理。

### ControlGoal

给控制器的目标：允许没有完整动作参考。

输入（这三组至少有一组对当前真实时间点有效）：
    motion: 旧版 ReferenceObservation，形状 [...,H,16+2N]，可选。
    contact_features: [...,C,7]，与 PhysicsObservation 的接触槽位对齐：
        0:3 希望接触点相对根部件到达的位置（米）。
        3:6 希望接触点相对目标物体的速度（米/秒）。
        6   接触要求：1=应接触，0=应离开，-1=当前不约束接触。
    contact_mask: [...,C]，True=对此接触点指定了目标。
    object_features: [...,M,6]，相对位置(3)+相对线速度(3)目标。
    object_mask: [...,M]，True=对此物体指定了目标。

输出：目标容器，由 GoalEncoder 转成网络特征。
注意：这个文件不会自行生成合理的接触顺序。上层任务规划器需要提供目标。

### PhysicsEncoding

物理观测编码后的结果。

context：[...,D]，当前机身与接触/物体整体情况。
contact_tokens：[...,C,D]，每个接触点自己的特征。
object_tokens：[...,M,D]，每个外部物体自己的特征。
这些都是神经网络特征，并不是物理引擎输出的真实参数。

### DynamicsEncoding

独立动力学模型从历史中形成的估计。

latent：[...,Z]，历史交互形成的动力学隐变量。
context：[...,D]，当前帧物理特征。
hidden：GRU 最后一帧的记忆 [B,R]，供后续接着处理时间序列。
prediction_error：[... ,9+2N]，上一时刻预测与本帧实际观测的差，
    没有可对齐的上一预测时全零。它会参与历史记忆的更新。

### DynamicsMemory

实时推理时保存的每个机器人环境的历史记忆。

hidden：[B,R]，上一帧 GRU 的状态。
env_ids：环境编号，禁止改变环境顺序后偷用原记忆。
owner：创建它的模型版本标记；训练更新权重后必须作废。
predicted_core：上一帧做出动作后预测的下一状态 [B,9+2N]，可能为None。
prediction_valid：上次预测对哪些环境有效 [B]，可能为None。
下一次 step() 会与真实状态比较，形成预测误差后消费掉这份预测。

### DynamicsPrediction

给定当前状态和当前动作后，对下一时刻做的预测。

next_core：[...,9+2N]，身体线速度3、角速度3、重力方向3、
    关节角偏移N、关节速度N；不包括上一步动作，因为下一帧上一动作已知。
next_contact_logits：[...,C]，下一帧各接触点是否接触的未归一化分数。
next_object_pos：[...,M,3]，下一帧物体相对身体的位置。
next_object_lin_vel：[...,M,3]，下一帧物体相对身体的线速度。
next_object_ang_vel：[...,M,3]，下一帧物体相对身体的角速度。
next_object_quat：[...,M,4]，下一帧物体相对根部件的 wxyz 朝向。

### DynamicsLoss

动力学预测的监督损失；total 用于反向传播，terms 用于记录诊断。

### PhysicsPolicyOutput

MLP 策略的一步输出；mean/std 是 PPO 概率分布，不是 PD 目标。

mean/std：[B,N]，N 为机器人受控关节数。
value：[B]，Critic 的预测，仅用于训练。
joint_names：原始动作所对应的策略关节顺序。

**distribution**

功能：将每个关节的高斯分布组成整台机器人的联合动作分布。

输入：对象内部保存的 mean/std。
输出：Independent(Normal(...))，log_prob 会在关节维求和为 [B]。

**sample**

功能：从 Actor 的分布中采样原始动作。

输入：不需要其他输入。
输出：action [B,N] 和它的 log_prob [B]。
用途：PPO 必须保存这份原始 action，不能用 PD 目标或被裁剪的角度替代。

### PhysicsEncoder

从一帧或多帧物理交互中提取网络可使用的特征。

输入：PhysicsObservation（身体状态、接触点、外部物体）。
输出：PhysicsEncoding，包含整体 context 与每个槽位的 token。
用途：让同一套动力学模型处理不同数量的物体/接触点。

**__init__**

功能：创建可训练的身体、接触和物体编码器。

输入：robot 关节合同；width 特征宽度；hidden_dim MLP 宽度。
输出：无；创建神经网络参数供优化器学习。

**forward**

功能：编码当前身体、每个接触点和每个物体，再形成整体物理特征。

输入：obs，可为一帧 [B,...] 或一段历史 [B,T,...]。
输出：context [...,width]；contact_tokens [...,C,width]；
      object_tokens [...,M,width]。
说明：这里不会跨时间学习，历史因果关系交给下面的 DynamicsModel。

### DynamicsModel

独立动力学模型：从历史形成隐变量，预测当前动作的后果。

功能：
    1. 编码接触/物体和身体状态。
    2. 用 GRU 记住此前发生的物理交互，估计 z_t。
    3. 给定当前原始动作 a_t，预测下个时刻的物理状态。

注意：GRU 只是学习工具，z_t 并不保证等于真实摩擦系数或质量。
如果训练数据中没有物理变化，这个模块也无法无中生有地学会适应。

**__init__**

功能：创建物理编码器、历史记忆和下一状态预测网络。

输入：robot 关节合同；physics_width 物理特征宽度；
      memory_width 历史记忆宽度；latent_width 动力学隐变量宽度；
      hidden_dim 各小 MLP 的中间宽度。
输出：DynamicsModel 对象，本身不启动采样或训练。

**get_extra_state**

保存动力学接口，避免同维度、不同机器人或关节顺序被误加载。

输入：无。输出：随 state_dict 保存的纯 Python 字典，不包含运行时记忆。

**set_extra_state**

加载时核对模型定义，并废弃旧 GRU 记忆。输入：保存的元信息；输出：无。

**invalidate_memories**

功能：声明优化器已经修改动力学网络权重，所有旧推理记忆失效。

输入：无。输出：无。
调用时机：每次更新本模块权重之后；训练器应清理/重新建立记忆。

**forward_sequence**

功能：在训练时顺序处理一段历史，让物理特征具有因果记忆。

输入：
    obs：全部张量都为 [B,T,...]，T 按真实时间从旧到新。
    episode_start：[B,T]，True 表示该帧是重置后的新回合第一帧。
    valid_mask：[B,T]，True 表示真实帧；尾部补位写 False。
    initial_hidden：[B,R]，可选，只有从回合中途切序列时才传入。
    previous_predicted_core：[B,T,9+2N]，可选。每个 t 保存
        t-1 时执行动作后，预测 t 时刻真实状态的结果。
    previous_prediction_valid：[B,T]，表示哪个位置有可用的上一预测。
        从 rollout 采集记录中读取，不可从未来真实状态反推填充。
输出：DynamicsEncoding：latent [B,T,Z]、context [B,T,D]、hidden [B,R]、
      prediction_error [B,T,9+2N]。
重要：没有 initial_hidden 时，所有环境的第0帧必须是回合起点。
      使用 PPO 时不要把一个 rollout 随意打乱顺序。

**step**

功能：在实时采样时处理一帧，保留各环境的物理记忆。

输入：
    obs：形状 [B,...] 的物理观测。
    episode_start：[B]，本次观测是否已经进入新回合。
    env_ids：B 个环境的固定编号；顺序必须始终相同。
    memory：上次调用返回的记忆，首次调用时传 None。
输出：DynamicsEncoding 的 latent [B,Z]、context [B,D]；以及新 memory。
限制：必须 .eval()；如果首次没有记忆，所有环境必须从新回合开始。
      更新模型权重后调用 invalidate_memories() 并重建记忆。

**remember_prediction**

功能：本帧 Actor 已决定 action 后，保存模型对下一帧的预测。

输入：memory 是刚从 step() 得到的记忆；prediction 由 predict_next() 产生；
      valid [B] 表示哪些环境的预测可在下一步比较，默认全 True。
输出：一个新的 DynamicsMemory，下次 step() 就能计算真实预测误差。
注意：预测必须在当前动作确定后计算，不能在下一帧用真实结果倒填。

**predict_next**

功能：给定当前真实状态、准备执行的动作和物理记忆，预测下个状态。

输入：
    obs：[B,...] 或 [B,T,...]，当前真实状态。
    action：[...,N]，当前策略准备执行的原始动作（不能是参考角度）。
    latent：[...,Z]，DynamicsModel 在当前时刻得到的隐变量。
输出：DynamicsPrediction：下一时刻机身/关节状态、接触概率分数、物体状态。
注意：预测结果只能用于学习/辅助决策；它不是 mjlab 的真实物理结果。
      如果要把预测用于 Actor，必须保证使用的 action 是当前可知的候选动作。

**prediction_loss**

功能：拿动作执行后的真实结果，监督训练独立动力学模型。

输入：
    prediction：predict_next 在执行动作前算出的预测。
    next_observation：执行动作之后、任何 reset 之前读取的真实下一状态。
    valid_transition：[B] 或 [B,T]，哪些预测与真实结果确实相邻。
        如果下一观测已被 auto-reset 替换，必须把该 transition 设 False。
输出：DynamicsLoss.total 标量，可用 backward()；terms 为各项诊断损失。
说明：比较真实观测，不依赖购买额外动力学数据集。

### GoalEncoder

把完整动作参考、接触目标和物体目标合成同一种策略目标特征。

功能：可以三选一，也可以混合；不要求每个新任务都有完整 MoCap。
注意：能读取目标不等于会自动规划目标；零样本能力仍要靠训练验证。

**__init__**

功能：创建三种目标各自的小编码器和统一融合层。

输入：robot 关节合同、width 输出特征宽度、hidden_dim 中间宽度。
输出：一个可学习的 GoalEncoder。

**forward**

功能：把目标与当前真实物理状态对比，生成目标和误差特征。

输入：physics 当前观测；goal 是动作参考/接触目标/物体目标。
输出：[...,width]；同一时刻只使用此刻已知的目标。
说明：接触位置误差和相对速度误差是逐槽位计算的，避免混淆对象。

### PhysicsMLPActorCritic

轻量控制基线：使用物理隐变量 z 和任务目标输出动作/价值。

输入：PhysicsObservation + DynamicsEncoding + ControlGoal。
输出：PhysicsPolicyOutput（高斯动作分布参数、Critic value）。
重要：DynamicsModel 独立训练。训练本 Actor 时默认让 dynamics 特征 detach，
      避免 PPO 策略损失悄悄改坏动力学预测器。Critic 的特权输入不进入 Actor。

**__init__**

功能：建立 MLP Actor 和独立 Critic，并保留 G1/K1 各自的输出维数。

输入：机器人合同；DynamicsModel 的 physics_width/latent_width；
      privileged_dim 为 Critic 专用额外状态维数，Actor 看不到。
输出：可训练 Actor/Critic。不同机器人分别创建一个实例。

**get_extra_state**

保存身体和目标接口定义，而不仅保存同尺寸权重。

输入：无。输出：随 state_dict 保存的字典；旧版缺此字段必须显式迁移。

**set_extra_state**

核对载入权重所属的机器人和特征接口；不一致就拒绝。

输入：checkpoint 里的定义字典。输出：无；绝不按同样列数猜关节语义。

**forward**

功能：使用当前物理状态、动力学适应结果与目标，计算动作和价值。

输入：physics 是 [B,...] 的当前观测；dynamics 的 latent [B,Z] 和 context [B,D]；
      goal 可以是完整动作或接触/物体目标；privileged 只给 Critic。
输出：mean/std [B,N] 与 value [B]，仍未真正执行到机器人。
说明：这是单帧 Actor；需要 GRU 记忆时先调用 DynamicsModel.step()。

### PhysicsMoEConditioner

把独立动力学特征注入已经编码的动作状态，供旧 MoE 主干读取。

功能：不重新实现 Expert，也不替换现有 HoloMotion 风格 Transformer。
输入：EncodedMotion、DynamicsEncoding。
输出：新的 EncodedMotion；只改变 state_token，不碰 reference_tokens。
注意：旧 MultiRobotActorCritic 当前在内部自行调用 Encoder，尚无插入接口。
      正式接入 MoE 时需要对 Actor 的调用链做一次明确的 v2 接口升级。
      它不是已经完成的可训练 MoE-PPO 闭环。

**__init__**

功能：创建输入宽度对齐的小网络。

输入：原 Transformer 宽度、物理特征宽度、z 宽度、MLP 中间宽度。
输出：可学习的物理条件化注入层。

**forward**

功能：给已有的身体 token 增加物理条件，不重复融合参考动作。

输入：encoded.state_token [...,D]，dynamics.context [...,P]、latent [...,Z]。
输出：新的 EncodedMotion，形状与旧版完全相同。
说明：初始输出和旧版相同，之后可由训练逐渐学会利用物理信息。

## models/physics_policy.py

### PhysicsPolicyConfig

功能：集中设置 Dense / MoE 策略的结构。

输入：创建配置时传入所需参数。
输出：不可修改的配置对象；用于构造网络，也保存在 checkpoint 中。

backbone："dense"=普通 Transformer；"moe"=MoE Transformer。
physics_width/latent_width 必须和独立 DynamicsModel 的输出维度相同。

### PhysicsSequenceOutput

功能：保存整段策略输出，供时序 PPO 计算旧动作概率。

mean/std：[B,T,N]，原始动作高斯分布参数，不是 PD 力矩目标。
values：[B,T]，Critic 的价值预测。
valid_mask：[B,T]，True 表示该时间点是真实采样。
balance_loss：标量；只在 MoE 模式有意义，不能在本层自动加到 PPO。

**distribution**

功能：构造每个时间点的动作概率分布。

输入：对象里的 mean/std，形状 [B,T,N]。
输出：高斯联合分布；对 [B,T,N] 动作求 log_prob 得 [B,T]。

**evaluate_actions**

功能：重算 PPO 中旧的原始动作的概率和分布熵。

输入：raw_actions [B,T,N]，必须是采样时的原始动作。
输出：log_prob [B,T] 和 entropy [B,T]。
注意：不要传被执行器裁剪后的目标角度。

### DenseReferenceTransformer

功能：保持与旧 MoE Transformer 相同的输入输出，但换成普通 FFN。

输入：EncodedMotion（状态、参考 token、参考掩码）。
输出：TransformerOutput，含处理后的隐藏特征及零 MoE 诊断。
用途：只对比 MoE 是否真的有价值，而不是改变整套任务输入。

**__init__**

输入：网络配置。输出：一个 Dense Transformer 主干。

**invalidate_caches**

功能：让权重更新前生成的 KV Cache 全部失效。

输入：无。输出：无；旧缓存随后传入 step() 将被拒绝。

**train**

功能：切换训练/推理模式，并让旧缓存失效。

输入：mode=True 表示训练；False 表示推理。
输出：当前网络本身。

**forward**

功能：整段读参考目标与历史信息，允许反向传播。

输入：EncodedMotion 状态 [B,T,D]、目标 [B,T,H,D]；
      valid_mask / episode_start 都是 [B,T]。
输出：TransformerOutput.features [B,T,D]。

**step**

功能：逐帧读取旧 KV 和当前目标，供 mjlab 实时采样。

输入：状态 [B,D]、目标 [B,H,D]、固定 env_ids、上一缓存、reset 标记 [B]。
输出：TransformerOutput.features [B,D] 和新的缓存。

### PhysicsConditionedActorCritic

功能：同一个策略类切换 Dense/MoE，共用输入接口和独立动力学模型。

输入：
    robots：如 {"k1_22dof": RobotContract(...), "g1_29dof": ...}。
    cfg：PhysicsPolicyConfig，决定 Dense 还是 MoE。
输出：一个可训练的网络；按 robot_id 分组读取对应机器人的数据。

模型结构：
    各机器人专用状态/运动 Encoder + 共享接触/物体目标编码器
    → 已有 PhysicsMoEConditioner（物理隐变量进入 Actor 隐藏特征）
    → 共享 Dense 或 MoE Transformer
    → 各机器人专用动作头 + 完全独立的 Critic。

网络主干权重可以共用，但 K1/G1 的动力学隐变量语义未必天生一致。
跨机器人联合训练是否产生迁移，需要实验，不能因为特征宽度相同就认定成功。

**__init__**

功能：创建网络、动作头和 Critic，所有参数在初始化时注册。

输入：
    robots：机器人 ID → 关节合同。
    cfg：架构配置，缺省使用 MoE。
    privileged_dims：每种机器人 Critic 可用的额外仿真状态维数。
输出：无；创建好的对象可放进 torch.optim.Adam。

**get_extra_state**

功能：保存网络接口信息，避免错把不同关节顺序的权重互相加载。

输入：无。输出：包含机器人和配置的版本字典，跟随 state_dict 存储。

**set_extra_state**

功能：加载 checkpoint 时核对网络结构与机器人合同。

输入：state：此前保存的接口版本字典。输出：无；不一致直接报错。

**invalidate_caches**

功能：声明网络权重已经改变，旧时序注意力缓存不再可信。

输入：无。输出：无；优化器更新后由训练器显式调用。

**train**

功能：切换训练/推理模式。

输入：True 开启训练，False 开启推理。输出：模型自身。
说明：底层 Transformer 会在模式改变时清除旧缓存。

**forward_sequence**

功能：整段计算动作分布和价值，供时序 PPO 重新评估历史采样动作。

输入：
    robot_id：当前是哪种机器人，例如 'k1_22dof'。
    physics：物理观测 [B,T,...]，必须按采样发生的时间顺序排列。
    dynamics：采样期间形成的同一时刻的 context/latent [B,T,P/Z]。
    goal：每一时刻的目标，可以是 motion 或 contact/object Goal。
    valid_mask：[B,T]，True 是真实控制步；后面的补位 False。
    episode_start：[B,T]，True 表示该帧已经是新回合。
    privileged：可选 [B,T,Q]，只用于 Critic。
    require_fresh_start：默认要求第0帧是各环境新回合起点。
输出：动作 mean/std [B,T,N]、values [B,T]、Router 诊断。

为什么要求新回合起点：
    Transformer 有历史 KV。若从回合中间截取一段，却丢掉前面历史，
    重算的 log-prob 就不等于采样时的 log-prob，PPO 将出现隐蔽错误。
    回合中段采样由 algorithms.sequence_training 提供前缀并核对历史，不能直接关闭保护绕过它。

**value**

只读取独立 Critic，不运行时序主干、不消耗动作采样随机数、不推进 KV。

输入：单帧物理观测 [B,...]、该帧动力学特征、当前目标和可选特权状态。
输出：[B] 价值；专门用于 reset 之前的最终状态 bootstrap。
不能用 policy.step() 代替本函数，否则会把“偷看一帧”的 KV 带到下一动作。

**step**

功能：逐帧生成一个控制动作，供 mjlab 真正执行。

输入：
    robot_id：当前机器人 ID。
    physics：当前一帧观测 [B,...]。
    dynamics：独立 DynamicsModel.step() 当前时刻返回的 z/context。
    goal：当前已知参考/接触/物体目标。
    env_ids：B 个环境固定 ID，行顺序变化必须重新建立缓存。
    episode_start：[B]，True 表示当前状态是 reset 后第一帧。
    cache：上一帧返回的缓存；首次必须是 None 且 episode_start 全 True。
    privileged：如果提供，仅交给 Critic。
    deterministic：True 直接执行均值；False 随机采样（供 PPO）。
输出：
    policy_output：mean/std/value；其中动作不是执行后的角度。
    action：[B,N]，未裁剪的原始动作。
    log_prob：[B]，采样动作在当前策略下的对数概率。
    cache：更新后的历史 KV Cache，下次步进继续使用。

## algorithms/sequence_training.py

### clone_tree

复制经验中的全部张量，切断梯度和仿真器共享内存。

输入：张量、数据类、字典、元组或普通元信息。输出：同结构的独立快照。
必须在环境下一次 step/reset 之前调用，否则旧经验可能被原地覆盖。

### stack_tree

将 T 份同结构的单帧数据堆成 [B,T,...]。

输入：非空列表；可选字段必须在这段序列中保持同一种结构。
输出：保留字段名字的数据类。切换任务时用有效 mask，不临时删除某个字段。

### select_tree

按第0维选择环境，保留环境内部完整时间顺序。

输入：含 [B,...] 张量的数据类和 [K] 整数索引。输出：对应 [K,...] 数据。

### at_time

从 [B,T,...] 数据树取第 t 帧 [B,...]；不改变原始数据。

输入：时间序列数据类和有效时间编号。输出：单帧数据，供 MLP 或缓存重放。

### parameter_stamp

记录参数身份和原地修改版本，用来拒绝过期 rollout，而不是判断模型性能。

输入：网络。输出：内存内使用的版本元组；不写入 checkpoint。
optimizer.step()/load_state_dict() 会改变版本。禁止用 .data 偷改权重绕过检测。

### TaskTransition

环境执行后的结果，不包含 reset 后观测。

reward/terminated/truncated 均为 [B]。terms 是可选分项奖励字典。
真正终止和外部时间截断不能同时为真；最终状态仍须由 task.read() 读取。

### PhysicsTask

采样器对环境的最小要求；mjlab 与 CPU 接口测评均实现这个协议。

生产任务必须 auto_reset=False。模型不会根据这个接口自动获得滑板接触计划。

**episode_start**

输出 [B] bool：当前读到的状态是否是新回合第一帧。

**reset**

输入待重置环境编号；输出无。None 重置全部，其他环境不能被移动。

**read**

无额外输入；输出当前 [B,...] 物理观测，不做步进或重置。

**goal**

无额外输入；输出当前时刻的合法目标，不能读取未来真实状态。

**step**

输入策略顺序原始动作 [B,N]；输出执行结果，结束行暂不重置。

### MotionTrackingTask

把现有物理桥、动作采样器和奖励接为 PhysicsTask，不重写它们。

输入：physics_bridge/sampler/reward_fn。输出：供统一采样器使用的任务对象。
目前真实物理桥仍是 K1；G1 的网络接口可测，但实物资产桥需要单独接入。

**__init__**

保存已有模块并核对机器人、频率和奖励所需数据。

scale_reward_by_dt=False 延续旧版外部奖励尺度；开启后总奖励和分项一起乘 dt。

**episode_start**

输出桥接器的 [B] 新回合标志，不改变环境。

**reset**

重置选中的仿真环境，然后只给这些环境重新分配和对齐参考动作。

输入：None 或 [K] 环境编号。输出：无；其他环境的轨迹时钟保持不变。

**read**

输出当前真实物理观测；快照复制由采样器统一负责。

**goal**

把当前参考窗口转换为网络特征。输出 ControlGoal(motion=...)。

**step**

执行一帧后再推进参考、计算奖励，绝不在这里 reset。

输入：[B,N] 原始动作。输出：本步奖励与结束原因；片段结束是真正任务终止。

### PolicyInput

一个控制时刻的冻结输入；用于重建策略历史，不保存 K/V 当作训练数据。

physics/goal 为 [B,...]，context/latent/error 为 [B,P/Z/K]，starts 为 [B]。

**dynamics**

组合冻结的动力学特征供策略使用；hidden 不被 Actor 读取。

输出 DynamicsEncoding。真实 GRU 隐状态另存于 rollout.initial_hidden。

### SequenceRollout

一批已经采到的经验，保存原始动作、完整时间顺序和可选历史前缀。

inputs/next_observation 是 [B,T,...]。reward/next_value/结束标志为 [T,B]，便于 GAE。
prefix 是本批以前的真实输入，仅用于重算历史；没有对应的新 reward，不能参与 loss。
consumed 防止同一份 on-policy 数据被重复更新。预测监督有独立的版本检查。

**horizon**

输出本批新采样的控制步数 T，不包含前缀。

**num_envs**

输出并行环境数 B。

**observation**

输出本批的 [B,T,...] 物理历史，供独立动力学监督复用。

**episode_start**

输出 [B,T] 新回合标志。

### PhysicsSequenceCollector

同一采样器支持 MLP / Dense / MoE，物理反馈来自传入的任务对象。

输入：task、独立 dynamics、policy。输出：collect(horizon) 得到 SequenceRollout。
更新策略后自动用前缀重建缓存；更新动力学后需显式 on_dynamics_updated()。

**__init__**

核对环境接口和模型宽度，不自动开始训练。

输入：任务实例与两套网络。输出：可反复采样的对象。
Critic 特权信息由可选 task.privileged() 返回，未配置时为 None。

**reset_all**

开始新训练段：重置全部任务与两种记忆，不保留旧 episode 的状态。

输入：无；输出：无。初次采样自动调用，研究中也可主动调用。

**on_dynamics_updated**

声明动力学权重已变，立即重放已保存的真实回合历史，重建 GRU、预测反馈和 Actor 前缀。
不会 reset 或推进物理环境，也不会复用旧权重的 GRU 状态。

输入/输出：无。构造 collector 时可指定 `dynamics_history_steps`（默认 16384）；
只能丢弃所有环境最近真实 reset 之前的公共前缀，超出容量在 step 前显式报错。
策略单独更新不需要此操作；它会利用真实前缀重建 K/V。
完整 Physics checkpoint 与冻结评测见 [运行时说明](PHYSICS_RUNTIME_AND_CONTINUITY.md)。

**collect**

采集 T 步真实交互，返回可供 PPO 与动力学监督共用的经验。

输入：正整数 horizon。输出 SequenceRollout；不进行参数更新。
任何 final-state 价值都通过无缓存的独立 Critic 读取，不偷推进 Actor 记忆。

### SequencePPOStats

一次 PPO 更新的统计；old_policy_error 是更新前复算误差，不是运动误差。

### PhysicsSequencePPO

MLP、Dense、MoE 共用的 PPO 更新器，禁止打乱真实时间。

输入：策略网络和 PPOConfig；update 输入 SequenceRollout，输出统计并更新策略。
独立动力学不归此优化器管理；PPO 使用采样时冻结的 context/latent。

**__init__**

创建策略优化器。输入模型及可选配置；输出无。

**verify_old_policy**

在修改任何权重前核对同一个动作的概率、均值、标准差和价值。

输入：刚采到的 rollout；输出：最大绝对复算误差。超限则拒绝训练。

**update**

执行一次 on-policy PPO。前缀只提供上下文，不重复计算奖励或优势。

输入：一份未消费且对应当前权重的 rollout。输出：更新统计。

**state_dict**

保存策略、Adam 状态和计数。输出可交给 torch.save 的字典。

不保存环境或 rollout；加载后必须重新采样，不能声称从同一个物理状态精确续跑。

**load_state_dict**

恢复策略优化状态，并核对算法配置。输入 checkpoint 字典；输出无。

### DynamicsUpdateStats

独立动力学一次监督学习的损失、梯度与累计更新次数，不是控制成功率。

### DynamicsUpdater

使用真实 (o_t,a_t,o_(t+1)) 训练物理预测器，不更新策略。

输入独立 DynamicsModel；update 使用同一份 SequenceRollout。
一次更新后旧 GRU 起始状态已经过期；重复监督前须重新构建历史或重新采样。

**__init__**

建立独立 Adam。输入模型和正数超参数；输出无。

**update**

更新物理模型一次，使用采样前的真实 GRU 状态进行截断反向传播。

输入：未因动力学权重改变而过期的 rollout。输出监督损失和梯度统计。
下一状态必须是 reset 之前读到的那一帧；终止转移仍可用于物理监督。

**state_dict**

输出物理模型/优化器/计数快照；不包含运行时 GRU。

**load_state_dict**

输入保存的监督 checkpoint；验证并恢复，废弃任何旧记忆。

## envs/mjlab_physics.py

### ObjectObservationSpec

描述一个需要被 Actor 观察的物体，不承担物体识别。

输入：
    entity_name：mjlab 场景中注册的物体名字，如 'skateboard'。
    size_m：这个物体的长、宽、高，单位米；来自明确的物体配置。
输出：创建后，K1PhysicsBridge 会在每帧读取这个物体的根位姿和速度。

注意：这里读取仿真真值，是研究阶段的理想传感器；不能拿来证明可实机部署。

### make_k1_physics_cfg

功能：在第11模块的 K1 环境配置上，增加真正的脚地接触测量。

输入：
    asset_root：官方下载的 booster_assets 所在目录。
    num_envs：并行 K1 的数量 B。
    physics_dt、decimation：物理步长与每个策略动作执行的物理步数。
输出：mjlab 的 ManagerBasedRlEnvCfg 配置对象。还没有启动仿真。

后续：ManagerBasedRlEnv(cfg, device='cuda:0') 才会创建物理环境。
这里不修改机器人碰撞结构，也不改 reward 配置。

### K1PhysicsBridge

把原版 K1MJLabBridge 扩展成物理反馈适配器，不重写它的动作控制。

输入：
    bridge：已经创建的 K1MJLabBridge，里面保存了 mjlab 环境。
    object_specs：可选物体配置，按给定顺序形成稳定的物体槽位。
    ground_velocity_fn：可选的世界坐标地面速度估计函数。
        输入 [B,2,3] 的接触点世界位置，输出 [B,2,3] 的地面速度。
        None 表示本模块配置的静止平面地面，地面速度为零。
输出：
    read() → PhysicsObservation，供 DynamicsModel 和 PhysicsMLPActorCritic 使用。
    tracking_state() → TrackingState，供第13模块计算基本跟踪奖励。

**__init__**

功能：一次性检查接触传感器、脚部部件和物体名字是否存在。

输入：bridge、object_specs、ground_velocity_fn、check_values。
输出：无；保存固定索引，后续每帧不用查找名字。

**read**

功能：读取一帧真实仿真状态，并生成第14模块的标准物理观测。

输入：不需要参数；会读取 bridge 中 mjlab 的当前状态。
输出：PhysicsObservation：
    body_obs：[B,72]，原版 K1 身体观测（策略关节顺序）。
    base_lin_vel_b：[B,3]，身体坐标的根部件线速度。
    contact_features：[B,2,12]，左/右脚各一个真实接触槽位。
    contact_mask：[B,2]，两个脚部槽位一直存在（接触与否看第10列）。
    object_features：[B,M,16]，M 个已登记物体的相对位姿与速度。
    object_mask：[B,M]，已登记且被观察到的物体槽位。

使用时机：bridge.reset() 后、或 bridge.step() 后且 reset 前读取。
本函数复制快照，不会因为之后 reset() 而修改以前保存的 rollout 数据。

**tracking_state**

功能：读取第13模块奖励使用的真实关节与根部件状态。

输入：无；直接读取 bridge.entity.data。
输出：TrackingState，其中 joint_pos [B,N]，root_pos_w [B,3] 等。
用途：与 MotionSampler 的当前参考比较，计算跟踪奖励。

**body_tracking_state**

功能：读取全身各个真实 link 的位置、朝向和线速度。

输入：无；使用 mjlab 当前的所有 body_link 状态。
输出：BodyTrackingState，body_names 对应 [B,M,3/4] 的第 M 轴。
注意：body 原点不是鞋底 site，不能拿它冒充足底接触位置。

## data/motion_sampler.py

### MotionClip

功能：保存一条已经适配机器人关节的完整动作。

输入字段：
    name：动作名称，用来知道当前抽到的是哪一条。
    robot_id：'k1_22dof'、'g1_23dof' 或 'g1_29dof' 等。
    root_name：根部件名称；必须和机器人合同一致。
    joint_names：下面 joint_pos / joint_vel 最后一维的实际关节顺序。
    fps：数据原始每秒多少帧；不需要强行等于 50。
    joint_pos、joint_vel：[F,N]，单位 rad、rad/s。
    root_pos_w：[F,3]，参考根部件的世界位置，单位米。
    root_quat_wxyz：[F,4]，参考根部件的世界朝向。
    root_lin_vel_w、root_ang_vel_w：[F,3]，世界线/角速度。
    body_*：可选的全部身体部件轨迹，形状 [F,M,3/4]。
            供后续足部、身体姿态奖励使用，不能代替物理仿真。
输出：这是数据对象，没有神经网络输出。MotionLibrary 会进一步检查它。

**from_project_npz**

功能：读取我们项目自己的标准动作 NPZ 文件。

输入：file，NPZ 文件路径。该文件必须自带 robot_id、root_name、
     joint_names、fps 和 6 组根/关节动作数组。
输出：MotionClip；如果缺字段、文件损坏则立即报错。
用途：适配 K1/G1 后保存一次，训练时反复使用。

**from_mjlab_npz**

功能：把 mjlab 原生 tracking NPZ 读成我们统一的动作数据。

输入：
    file：mjlab 动作 NPZ 文件路径。
    robot_id：这份轨迹属于哪款机器人，必须明确指定。
    joint_names：这个 NPZ 的 joint_pos 各列真实对应哪些关节。
    body_names：这个 NPZ 的 body_pos_w 各列真实对应哪些身体部件。
    fps：该文件真实采样帧率，例如 50；绝不能凭文件名猜。
输出：MotionClip，根部件数据从 body_names 中正确提取。

警告：mjlab 原生文件可能不带关节/身体名字。调用者必须提供与
生成文件时完全相同的顺序。仅凭维度相同并不能证明数据正确。

**save_project_npz**

功能：将一条动作保存为我们项目统一格式，方便之后训练直接加载。

输入：file，准备保存的 NPZ 文件路径；父目录不存在时自动创建。
输出：没有返回值；文件写到磁盘。不做 retarget 或坐标转换。

### MotionLibrary

功能：把多条不同长度的动作放进一个库，并统一关节顺序。

输入：
    robot：第 01 个模块的 RobotContract，比如 K1 22DoF。
    clips：一条或多条 MotionClip；必须属于同款机器人。
    device：保存和采样张量的设备，例如 'cuda:0'。
    dtype：float32 或 float64。通常使用 float32。
输出：对象自身存有全部动作；不做随机抽样，随机抽样交给 MotionSampler。

这里采用连续拼接保存，而不是每条动作补成同样长度；这样长短动作混合时
不需要为很长的无效补位浪费 GPU 显存。

**__init__**

建立动作库，一次性检查各条动作并转换为策略关节顺序。

输入：robot 机器人合同；clips 多条 MotionClip；device/dtype 存储设备与精度。
输出：无直接返回，所有有效帧连续保存到本对象；不会替不同动作补相同长度。

**from_project_folder**

功能：把文件夹中的所有本项目标准 NPZ 动作载入同一个机器人动作库。

输入：folder 是动作目录（支持子目录），robot_id 明确指定机器人的类型。
输出：MotionLibrary。不能将 K1 与 G1 动作混在同一个库。

### BodyReferenceWindow

功能：保留未来奖励可能用到的身体部件姿态，不用关节角假装脚部位置。

输出字段：
    body_names：这些列是什么身体部件，顺序固定。
    pos_w：[B,H,M,3]，世界位置。
    quat_wxyz：[B,H,M,4]，世界朝向。
    lin_vel_w、ang_vel_w：[B,H,M,3]，世界速度。
    valid_mask：[B,H]，True=真实参考帧。

### MotionSampler

功能：为每个并行环境记住抽到了哪条动作、当前播放到哪里。

输入：
    library：上面的 MotionLibrary，已经载入一款机器人的所有动作。
    num_envs：并行机器人数量 B。
    step_dt：每一次 mjlab 控制步经过多少秒。50 Hz 时为 0.02。
    future_offsets_s：当前、未来几个参考时间，例如 (0,0.04,0.1,0.2)。
输出：
    reference_window() 产生第03模块原样接收的 ReferenceWindow。
    body_reference_window() 提供身体部件的目标（只有源数据有它们时可用）。

使用顺序：reset(...) → reference_window() → 仿真执行一步 → advance()。
每条环境在真实 reset 后，必须同时调用 sampler.reset(对应环境索引)。

**__init__**

为每个并行环境准备独立的动作时钟。

输入：library动作库、num_envs环境数B、step_dt控制周期秒数、future_offsets_s参考时间点偏移（从0开始严格递增）。
输出：无；创建[B]片段编号、步数和世界对齐状态。创建后须reset分配动作，不能直接读取。

**reset**

功能：给刚重置的机器人安排一条动作和起始进度，并固定世界对齐。

输入：
    env_ids：[K]，要重置的环境；None 表示全部 B 个环境。
    clip_ids：[K]，每个环境选择哪条动作；None 则均匀随机选一条。
    start_times_s：[K]，从每条动作第几秒开始；None 默认从 0 开始。
    random_start=True 且 start_times_s=None：随机挑一个起始时间。
    min_remaining_s：起始位置后至少保留多少秒动作，例如 0.5。
    current_root_pos_w：[K,3]，仿真重置后根部件的真实世界位置。
    current_root_quat_wxyz：[K,4]，仿真根部件的真实世界朝向。
        两个真实根输入必须同时给；会用它们对齐参考的初始 XY 和朝向。
        不给时，直接采用数据集原来的世界位置、方向。
输出：无返回。只改变采样器中对应环境的当前动作、时间、对齐状态。

注意：为了正确保留 root XY 跟踪误差，只在 reset 时对齐一次。

**current_times_s**

功能：得到每台机器人当前播放到动作的第几秒。

输入：env_ids，选哪些环境；None 表示全部。
输出：[K] 的秒数，片段到头后停在最后一帧，不进入下一条动作。

**finished**

功能：告诉任务管理器哪些动作已经播放到最后一帧。

输入：env_ids；None 表示全部。
输出：[K] 的 bool，True 表示动作结束；训练环境应以此触发终止并换动作。

**advance**

功能：仿真成功执行控制步之后，推进参考动作的播放进度。

输入：steps 是已经执行了多少个控制步，正常为 1；env_ids 是哪些环境。
输出：无返回。内部用整数计数，不反复累加 0.02 造成累计浮点漂移。

**reference_window**

功能：读取当前与未来 H 个动作时间点，直接交给第03个文件。

输入：env_ids 指定要读取的环境；None 表示全部 B 个环境。
输出：ReferenceWindow，包含：
    joint_pos/joint_vel：[K,H,N]，已排成策略定义的关节顺序。
    root_pos_w：[K,H,3]，已对齐世界根位置。
    root_quat_wxyz：[K,H,4]，已对齐世界根朝向。
    root_lin_vel_w/root_ang_vel_w：[K,H,3]，速度也一起旋转。
    time_offsets_s：[K,H]，例如每行 [0,0.04,0.1,0.2]。
    valid_mask：[K,H]，动作结尾以 False 标记，不会跨片段。

**body_reference_window**

功能：取出动作的身体部件世界轨迹，供之后的足端/身体奖励使用。

输入：env_ids 选择环境；None 表示全部。
输出：BodyReferenceWindow，其中位置 [K,H,M,3]、朝向 [K,H,M,4]。
要求：输入动作 NPZ 必须确实包含 body_names 和四组 body_*，否则拒绝。

## rewards/motion_tracking.py

### FootPhase

脚底点的动作类别，由参考数据的制作人标注，不能让奖励函数猜。

UNKNOWN(0)：未标注。不参与接触/锚定/滑动专项奖励。
ANCHOR(1)：参考要求这个脚底点支撑地面，并尽量保持不滑。
SLIDE(2)：参考要求这个脚底点保持接触，同时在地面上滑动。
SWING(3)：参考要求这个点离地摆动。
TRANSITION(4)：交接期。不强加接触或锁脚条件。

输入位置：FootSiteTarget.phase，[B,S]，必须是整型张量。
输出：这些是整数标签，不是神经网络预测的概率。

### TrackingState

一帧真实机器人状态：可从 mjlab 的 EntityData 读取。

输入：
    joint_names：joint_pos/joint_vel 的列顺序，可以与策略顺序不同。
    joint_pos / joint_vel：[B,N]，真实关节角(rad) / 角速度(rad/s)。
    root_pos_w：[B,3]，实际根部件世界位置(m)。
    root_quat_wxyz：[B,4]，实际根部件朝向(wxyz)。
    root_lin_vel_w：[B,3]，实际根部件世界线速度(m/s)。
    root_ang_vel_w：[B,3]，实际根部件世界角速度(rad/s)。

输出：只是记录数据的对象。forward() 会将关节按名字转换到策略顺序。
注意：这些是仿真结束后的真实状态，不是 Actor 对未来的预测。

### BodyTrackingState

一帧真实的多个身体部件，可直接对应第 12 个模块的 body 参考。

输入：
    body_names：每列属于哪个部件，比如 trunk、left_ankle_roll_link。
    pos_w：[B,M,3]，真实部件中心/链接原点的世界位置。
    quat_wxyz：[B,M,4]，每个部件在世界坐标中的朝向。
    lin_vel_w：[B,M,3]，每个部件原点的世界速度。

输出：无。这些值供身体位置、朝向、速度跟踪奖励使用。
注意：部件原点不一定是脚底。不能把它直接当作鞋底滑移测量点。

### FootSiteState

脚底测量点的真实物理状态，用来检查脚有没有按要求移动。

输入：
    site_names：真实 site 名称，如 left_sole、right_sole，名字不能重复。
    pos_w：[B,S,3]，脚底点真实世界位置(m)，不是 ankle link 的位置。
    vel_w：[B,S,3]，可选；这些脚底点原点的真实世界速度(m/s)。
    quat_wxyz：[B,S,4]，可选；脚底测量坐标系的真实朝向。
    contact：[B,S]，可选 bool；物理传感器实测是否接触地面。
    prev_pos_w：[B,S,3]，可选；上一控制步相同测量点的位置。
        必须与本帧属于同一回合；机器人重置时要清除旧帧记录。
    dt_s：可选；两帧之间的真实控制周期(s)，例如 0.02。

输出：无。forward() 会按需要读取这些物理量。
注意：如未提供 vel_w，可以同时提供 prev_pos_w 和 dt_s 估算位移速度；
      不能用脚踝关节速度代替鞋底点的世界速度。

### FootSiteTarget

脚底测量点对应的明确参考目标，不能从身体中心位置偷换得到。

输入：
    site_names：参考点名字，必须与 FootSiteState 的点一一对应；顺序可以不同。
    pos_w：[B,S,3]，已根据参考动作 FK 算出的鞋底点世界目标位置。
    vel_w：[B,S,3]，可选；同一鞋底点的世界目标速度。
    quat_wxyz：[B,S,4]，可选；参考鞋底点朝向。
    phase：[B,S]，可选，整数 0~4；锚定、滑动、摆动、过渡的**标注**。

输出：无。这份数据是“参考目标”，不是脚的真实接触状态。
重要：MotionSampler.body_reference_window() 目前提供的是身体部件轨迹；
      若要生成真正的 sole/toe/heel site 目标，必须使用同一个机器人模型
      对参考关节姿态进行 FK，并显式保存/计算 site 位姿与相位标签。

### MotionTrackingRewardConfig

奖励权重与误差容忍度。权重越大，该项对总奖励的影响越大。

输入：
    *_weight：各项奖励权重，必须 >=0；0 代表关闭该项。
    *_sigma_*：容忍度。实际误差接近 sigma 时，这项得分约为 exp(-1)。
    数值是便于启动研究的示例，不是 K1/G1 已标定的最佳值。

输出：配置对象，供 MotionTrackingReward 使用。

默认仅开启根部件和关节跟踪：
  - 有 body_* 参考时，可以打开 body_*_weight。
  - 有“同一鞋底 site”的参考和真实物理数据时，才能打开脚部专项权重。
  - 没有 phase 标签时，不能开启接触匹配、锚脚锁定或滑动速度奖励。

### MotionTrackingRewardResult

一次奖励计算返回的东西。

输出：
    total：[B]，每个环境最终得到的奖励，用于 PPO rollout。
    terms：字典，每项都是 [B]；存已经乘上权重的实际奖励贡献。
    metrics：字典，大多数值为 [B]；名称带 by_site 的值为 [B,S]，
             会保留左右脚/heel/toe 的单独误差，不用平均值掩盖单脚失败。
    foot_site_names：metrics 的第 2 维 S 对应哪个脚底点；没有脚底数据时为空。
注意：没有数据的可选奖励项不会出现在 terms 中，不用假的 0 分掩盖缺失。

### MotionTrackingReward

把当前真实状态和同一时刻的目标姿态放在一起打分。

输入：
    robot：RobotContract，指出需要跟踪的是 K1 还是某个 G1 配置。
    cfg：奖励权重和尺度，不提供就使用仅关节/根跟踪的基础配置。
    check_values：是否逐项验证 NaN/Inf 和单位四元数；GPU 会有额外开销。
输出：一个可调用模块。调用 forward(...) 返回 MotionTrackingRewardResult。

特别注意：
    这个类本身没有学习参数，不通过反向传播训练奖励。
    它只负责给本步“已经执行出来的真实物理结果”打分。
    不保存滑步进度；完整 slide 位移、左右交替、全片节奏要由评估器另外统计。

**__init__**

功能：保存机器人合同和配置，准备计算奖励。

输入：robot 指定关节名单；cfg 指定权重；check_values 控制严格数值检查。
输出：无；创建的对象可在后续每个控制步重复使用。

**forward**

功能：比较 mjlab 当前真实状态与参考轨迹，计算 PPO 本步奖励。

输入：
    state：TrackingState，[B,N] 关节+ [B,3/4] 根部件真实状态。
    reference：MotionSampler.reference_window() 的输出，[B,H,N] 等。
        **第0个参考时间点必须是当前时刻**，其 time_offsets_s[:,0] 为0。
    bodies：可选 BodyTrackingState，真实部件位姿。
    body_reference：可选 BodyReferenceWindow，参考身体部件轨迹。
        两者均提供后，才可开启 body_* 奖励。
    feet：可选 FootSiteState，mjlab 真实鞋底 site 的位置/接触。
    foot_reference：可选 FootSiteTarget，参考鞋底 site 的位置/阶段标签。
        两者均提供后，才可开启脚部专项奖励。

输出：MotionTrackingRewardResult：
    total：[B]，最终奖励；交给后续 PPO collector 保存。
    terms：字典，已乘权重的每项奖励 [B]。
    metrics：字典，通常是 [B]；foot_*_by_site 是 [B,S] 的逐脚底点误差。
    foot_site_names：逐脚底点指标的真实列名，方便单独验收左右脚。

使用顺序：mjlab 执行一步 → sampler.advance() → 获取当前 reference →
          读取 mjlab 此刻的真实状态 → 调用本函数 → 保存 total。
注意：mjlab 第11模块环境当前 rewards={}；因此不能使用 bridge.step()
      返回的零 reward 冒充此函数计算的追踪奖励。
