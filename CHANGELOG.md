# 0.21.0：LRU与无MoE Transformer别名

- 增加五组物理适应结构消融：无历史latent、预测latent、直接系统辨识、
  Teacher–Student和不可部署Oracle上限；Actor动力学输入由显式配置控制。

基准：用户提供的`unified_motion_r20_network_plugins.zip`；原始附件未被修改。

## 实际新增
- 2026-10-12 增加可选的 Dynamics 多跨度直接监督（t+4/t+8）：读取同一在线 rollout
  的未来真值和有序动作序列，不自回滚，并屏蔽跨 reset/终止窗口。
- 增加训练专用的足底切向力、滑移速度和摩擦利用率监督；它们只进入特权 Critic、
  system-ID 和 Teacher，不进入可部署 Actor/Dynamics 观测。
- `latent_width`、多步损失和相邻时刻 latent 慢变正则均可配置；默认值保持旧 checkpoint
  结构，启用新版结构必须新建输出目录。冻结评测可按 checkpoint 参数重建两种结构。
- `models/lru.py`：可学习复数对角线性递推、gamma初始化、配对并行扫描、顺序对照、层间GLU与残差。
- 逐环境reset、末尾padding、LRU实/虚双分量记忆；复用原循环缓存和完整历史PPO重建。
- `backbone=lru`与LRU专用YAML；`backbone=transformer`规范到原dense，只有一份实现。
- 新配置包含set/GNN×LRU/Transformer及真实K1候选配置，后者仍待真实物理验收。
- `inspect`记录真实MoE层数和LRU记忆实数个数，不凭名称推测是否轻量。

## 兼容
- 非LRU结构不把新LRU字段写入权重元数据，保留R20五种已有Actor的checkpoint。
- 旧10份诊断checkpoint做了严格加载；实验级resume仍须保持原始配置。
- 不改DynamicsModel的GRU，不复制PPO，不强行串联全部网络。
- 裸LRU与论文数学对照，但完整机器人网络不声称复现论文训练成绩。
- 接触桥同时读取 mjlab 1.6 的公开 `primary_names` 与现有 mjlab 1.3 的初始化槽位，二者保持同一 primary 张量顺序。

## 自检与限制
- 新回归独立使用复数公式与有限差分，不仅比较两个都可能写错的同源函数。
- 12组合跑短程CPU诊断训练与冻结评测；成功率仍为0。
- 本次实际数字见reports/current，真实机器人和CUDA运行不冒充通过。
- 2026-10-10 本机已有 mjlab 1.3 兼容性补测跑通 K1 CUDA 物理桥及一次最小训练更新；
  严格 mjlab 1.6 依赖门禁仍未通过，且没有真实K1已训练策略。
- 2026-10-12 修正并行 mjlab 环境中 terrain 关系遗漏 `scene.env_origins` 的问题；旧行为会让
  非中心环境追世界原点目标，使首步位置误差约1.414m并令跟踪奖励下溢为0。
- 物理反馈 A/B 工具新增侧推间隔、训练/评测推力范围和连续保持时长参数，避免任务在首次
  扰动前已经成功，导致所谓鲁棒性评测实际没有施加扰动。
- 冻结评测新增实际推力次数、恢复率和平均恢复步数；周期扰动场景不再只用“连续保持到
  成功”这一项容易与推力周期冲突的指标判断反馈效果。
- 新增 `configs/k1_lateral_step.yaml` 动态接触任务：左脚抬起、向外落脚并稳定，训练与
  评测使用分离的摩擦/推力范围，用于比原地站立更有辨识力的快速反馈实验。

---

# 0.20.0：可选GRU/LSTM Actor与物理GNN插口

基准是用户提供的R19 zip，原附件不修改。本次不宣称补齐真实滑板/平台/G1环境。

## 新增并接入

1. models/recurrent.py：GRU/LSTM Core、h/c回合隔离、权重/环境绑定缓存、原Actor主干适配。
2. models/graph_encoder.py：GraphBatch、按显式边聚合的消息传递、固定身体—关系—对象图。
3. DynamicsModel.scene_encoder=set/gnn，GNN参与独立物理监督，不由PPO暗中更新。
4. build_policy支持gru/lstm；原MLP/Dense/MoE不删除，原目标编码/动作头/Critic/PPO复用。
5. 循环Actor保留真实回合历史：没有有限注意力窗口假设；更新后用新权重重放。
6. YAML新增recurrent_hidden_dim/max_replay_steps/scene_encoder/graph_layers/关系映射。
7. 10种小配置（五Actor×图开关）、inspect参数统计、训练保存加载与冻结诊断矩阵。

## 额外修复

- 旧tools/evaluate.py仅用unittest收集，会漏掉R19及之后的函数式pytest用例。
  新入口在独立pytest进程收集所有测试，父子用例分开计数，不预填通过数字。
- 图映射不仅检查整数范围，还与后端公开对象槽位名称顺序核对。
- 循环Actor的缓存额外检查整个Actor的参数版本，输入Encoder改变也会拒绝旧h/c。

## 保留的限制

真实mjlab/K1/G1物理未验收；真实平台/滑板未新增；统一混合任务训练未补完；
CoP/总足底力、GNN自动骨架构建/动态对象跟踪、Dynamics历史LSTM未实现；
MoE带前缀时仍禁用balance_coef。R20解决网络插口而非宣称通用技能完成。

本轮实际结果见reports/current，历史R19报告已移至reports/r19_history。

---

# 0.19.0：可运行YAML闭环、接触计划、交互误差回灌、冻结评测

本轮实际改动与边界见docs/R19_CHANGES.md；当前验证报告见reports/current/。
新增真实mjlab接触接口待物理验收，默认CPU解析场景明确不是人形机器人。

---

# 0.18.0：物理验收与执行参数一致性

基准是用户提供的unified_motion_r17_clean.zip。所有网络、动力学模型、PPO/collector及奖励算法保持R17实现。

## 新增
- envs/physics_validation.py：五级验收，独立PASS/FAIL/BLOCKED和退出码。
- tools/verify_physics.py：统一CLI；编译、原生CPU与CUDA结果明确分开。
- 资产XML与mesh/texture指纹；每次真实训练保存实际控制表。
- constraints/sim-r18.txt：依照mjlab固定源码版本的候选组合，不标成已测试的环境锁。
- 36项CPU回归：真实引擎相关审计逻辑使用明确标记的mock字段，不冒充物理编译。

## 修复/约束
- K1 XML未声明radian、include未展开、浮动根挂错body、显式关闭限幅时拒绝误读。
- mjlab构造与验收共享K1ControlSpec，避免复制两套Kp/Kd公式。
- 静态Scene.compile之后显式应用cfg.sim.mujoco；此前工具漏掉训练积分器/步长设置。
- 零动作与初始姿态不一致的膝角差异公开记录，没有默默改变物理任务。
- 旧静态检查工具变为薄转发，不再另有一套编译逻辑。
- 训练启动检查仿真包版本和资产齐全性，不替换用户CUDA/PyTorch。

## 兼容性
R17神经网络权重结构没有改变。旧非法/含糊XML现在会更早报错。
旧测试XML样本补上显式radian声明，与原测试的“这些数字是弧度”的语义一致；未删除旧断言。

## 实际测评
474项：461通过，13 CUDA跳过；pytest独立复跑一致。
本环境依赖/真实mesh缺失且下载失败，原生物理和mjlab仍BLOCKED。未用假环境替代它们。

---

# 0.17.0 修改清单

基准：用户上传的 `unified_motion_module16.zip`。未改写或覆盖该原始附件。
本版按源代码审计与实际CPU回归修订；不从历史消息里的“测试通过”直接继承结论。

## 新增正式功能

- `algorithms/sequence_training.py`：统一 PhysicsTask、MotionTrackingTask、完整快照采样、MLP/Dense/MoE PPO及独立DynamicsUpdater。
- 采样保存真实原始动作和概率、最终观测、两种结束原因、动力学隐变量和历史前缀。
- 多层有限窗口使用 `L*(C-1)` 有效前文，不误以为只补一层C帧就足够。
- 策略更新后按新权重回放前缀建立KV；回合reset与环境行ID严格隔离。
- `policy.value()` 单独评估Critic，bootstrap不推进Actor缓存、不采样动作。
- PPO更新前复算概率/均值/标准差/价值，并拒绝过期权重、重复批次或缺失历史。
- PPO与动力学分开优化；动力学更新后必须显式清理运行时记忆。
- 统一建模/训练入口 `api.py`、`train.py`，新增标准包配置和集中测评工具。

## 修复已有模块

1. **混合目标补位：** 各类目标分别有padding时，直接拼接会产生 `True,False,True`；现在稳定压紧有效token，并同步重排mask，不关闭校验。
2. **物理槽位补位：** 先屏蔽无效contact/object输入，再进入MLP；避免有限但很大的无效数字先溢出，再在输出处乘0导致错误梯度。
3. **等价四元数：** 物体朝向 q/-q 进入物理编码/预测前采用同一符号；同一朝向不再被认为是两个物理状态。
4. **退化单位向量：** 预测重力/朝向接近零时使用当前有效方向后备，避免“归一化后仍为零”。
5. **空动力学监督：** 原代码先clamp样本数再判断小于1，空批次检测不会触发；改成先检查有效转移。
6. **模型语义与记忆：** Dynamics及MLP checkpoint增加机器人/宽度/版本定义；载入动力学后旧GRU记忆失效。相同尺寸不代表相同关节顺序。
7. **随机动作剩余长度：** 自动选片段先筛选满足 `min_remaining_s` 的动作，避免随机抽中短片段后间歇报错。
8. **动作库类型与列顺序：** 拒绝重复body名字/非法fps；重排索引跟随实际输入device；末帧插值索引显式夹在当前片段内。
9. **首次partial reset：** 初次必须重置全量环境，之后才允许部分重置，防止把未初始化的环境标成ready。
10. **传感器语义：** K1桥核对maxforce、单槽位、世界系force配置，不将接触系力或多槽位错当两只脚。
11. **XML执行合同：** 拒绝非单自由度转动关节、错误浮动根数量和非单位电机gear，避免直接用错力矩尺度。
12. **MLP数值：** 检查非法宽度与非有限Critic输出，不把坏value送入GAE。

## 目录与注释

- `physics_dynamics.py` 的唯一实现改为 `models/physics.py`。
- `physics_policy_v2.py` 的唯一实现改为 `models/physics_policy.py`，规范配置名为 `PhysicsPolicyConfig`。
- 老名字仍通过薄转发兼容；不保留两份可分别修改的实现。
- 旧参考Actor/PPO、旧单步MLP训练移到 `legacy/`；公共GAE/配置/高斯头提取到正式公共模块。
- 删除生产代码中的 `_self_test`；回归测试集中管理，mjlab操作CLI不是自测，仍保留。
- 补充旧核心模块的中文功能/输入/输出说明，以及新采样闭环的时间与形状约定。
- 修正“Router完全与状态无关”等超过实际代码支持范围的说法。

## 本版没有假装解决的事项

- 不证明物理动力学正确、长训收敛、毫米级接触控制或零样本技能迁移。
- 无真实mjlab/MuJoCo环境验收；CPU适配器测试只验证接口数学。
- 有前缀时暂不支持MoE balance loss；默认0且显式拒绝误用。
- 动力学更新后重新开始环境，尚未实现权重变化时无缝重建整段GRU历史。
- 未增加AMP、通用接触规划器、G1资产适配，也不复制私有数据。
# 2026-10-12：可部署在线物理辨识与特权 Critic

- 新增显式 `adaptation` 配置。`actor_observation: deployable` 会屏蔽仿真器精确根线速度、足端对地速度、接触法线/力和无真实估计器的物体状态。
- `RandomizedTask.privileged()` 只向 Critic 和动力学监督提供归一化摩擦、质量、延迟、推力幅值、平台速度及上一控制步外力；Actor 接口不接收这些真值。
- `DynamicsModel` 新增可选系统辨识头，从可部署观测历史的 `latent` 预测低维物理真值；首个无动作响应帧不计算该辅助损失。
- 新增训练期 `Teacher Physics Encoder`：仿真真值编码为 `z_priv`，Student 的 `z_online` 只对齐 `stop_gradient(z_priv)`；Teacher Decoder 的真值重建损失防止 latent 塌缩。Teacher 不进入 Actor 推理路径。
- 策略工厂现在真正接通 asymmetric Actor-Critic 的 `privileged_dims`，并增加“特权变化不能改变 Actor 动作”的回归测试。
