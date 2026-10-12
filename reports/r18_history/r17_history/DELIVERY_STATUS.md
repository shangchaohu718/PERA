# 本次交付验收状态

## 已执行

- 原有356项：343 CPU通过、13 CUDA跳过。
- 新增82项：动力学、三种物理策略、时序采样/PPO、标准动作NPZ、奖励、K1桥接、包入口/兼容路径。
- 合计438项：**425通过，13 CUDA跳过，0失败，0错误**。
- 独立pytest复跑：425 passed, 13 skipped（另409子用例通过，不重复累加总数）。
- ZIP完整性检查通过；重新解压到独立目录后再次执行438项：425通过、13CUDA跳过、0失败/错误。
- `pip install --no-build-isolation --no-deps -e .` 实际成功。
- `python -m unified_motion.train --help` 实际成功。
- MLP / Dense / MoE 在CPU数学任务上实际进行了rollout、PPO梯度更新及独立动力学更新，包含多轮和部分环境重置。
- 桥接集成使用模拟mjlab接口，不启动接触求解器。

## 跳过

13个已有CUDA测试用例：当前 torch 为2.10.0+cpu，无可用CUDA。

## 阻塞 / 未执行（不是通过，也不计入13个CUDA跳过）

- mjlab、MuJoCo、Warp不在本环境；依赖下载失败。
- 因此真实K1模型的CPU编译与实际执行器/接触行为没有验收。
- 真实GPU rollout、性能、训练收敛、视频动作质量、跨物理引擎/实机/零样本能力没有验收。

**所以本次满足“定义的非CUDA代码回归套件全部通过”，不满足“所有非CUDA物理运行均已完成”的更强表述。**
最后的交付回复必须保留这个区别。

完整逐项结果：evaluation.json / cpu_regression.txt / pytest.txt。
