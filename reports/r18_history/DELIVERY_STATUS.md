# R18交付验收状态

## 本次实际完成

- CPU统一回归：474项总计，461通过，13 CUDA跳过，0失败、0错误。
- pytest独立复跑：461 passed, 13 skipped, 419 subtests passed。子用例不重复计入总数。
- 新增36项CPU回归，没有删除旧测试断言；原XML样本补充显式radian声明。
- 可编辑安装成功：unified-motion-research 0.18.0。
- 训练入口和物理验收入口的help可运行。
- 依赖检查确实执行，报告BLOCKED/退出2。
- native阶段入口确实调用，前置检查BLOCKED，因此没有调用真实求解器。
- 神经网络、独立动力学、PPO和collector源码逐字节与R17相同。

## 非通过项

- 缺少mjlab、mujoco、mujoco-warp、warp-lang，外部下载在当前环境失败。
- 没有完整Booster资产和mesh。
- 没有执行真实K1 XML/mesh的本地编译、原生MuJoCo CPU物理步进或mjlab CUDA步进。
- 13项CUDA跳过不能包含或掩盖上面这些非CUDA阻塞。
- 没有PPO物理长训、站立质量、足端精度、实机或零样本实验结论。

## 判读要求

本次只能说“执行过的CPU回归全部通过”。不能说“所有非CUDA物理运行都已通过”。
新增测试中的人工XML、字节mesh和model字段明确是单元测试，不能替代真实引擎。
候选仿真约束文件依据公开源码制定，不是本次已安装并验证的完整依赖锁。

从ZIP独立解压后实际复跑：474项，461通过、13 CUDA跳过、0失败、0错误。
完整结果见 clean_extract_evaluation.json / clean_extract_regression.txt。
