# R18：怎样判断“接到仿真器”是真的？

## 先区分四件事

- 网络和接口回归通过：张量顺序、梯度、PPO概率、缓存等逻辑通过。
- XML与文件审计通过：名字、限位、文件引用可读取；**没有编译mesh或物理步进**。
- 原生MuJoCo CPU通过：真实引擎用训练同源的场景求解；**不是mjlab/Warp结果**。
- mjlab CUDA通过：真正运行GPU环境，得到正式桥的物理观测；**仍不是策略训练成功**。

不能因为后两项未能运行，就拿人工XML或假环境补一个PASS。

## 一、准备依赖（在你的训练机执行）

在新目录使用项目，保留原来代码和训练结果。先确认当前解释器：

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
```

仿真目标为mjlab1.6.0，对应上游commit：
`0fb8a681136be94ffc636a3dd423cabb97d91f10`。
其公开依赖为Python>=3.10,<3.14、MuJoCo/MuJoCo-Warp 3.11系列、Warp>=1.14。
项目原交付核心实测是Python3.13.5、torch2.10.0+cpu。2026-10-10 又使用机器上已有、
未重新安装的 mjlab 1.3.0 / MuJoCo 3.7.1.dev / MuJoCo-Warp 3.7.0.1 /
Warp 1.12.1 / torch 2.11.0+cu130 做了兼容性实测。该组合不满足下面的1.6锁版门禁，
因此只能记为“现有环境兼容通过”，不能冒充目标依赖组合正式通过。

在已有相容CUDA PyTorch的隔离环境内安装：

```bash
python -m pip install -e '.[sim]' -c constraints/sim-r18.txt
python -m pip check
python tools/verify_physics.py --phase dependencies --output reports/dependencies.json
```

`constraints/sim-r18.txt`是按源码核对的候选组合，不是本次已验证的环境锁。
未固定全部传递依赖；安装成功后额外保存 `python -m pip freeze` 输出。
项目不自动升级CUDA、不自动卸载Torch，也不绕过TLS或使用未经确认的轮子。
某些平台的Warp轮子需要遵循mjlab官方的安装源设置，见本文底部来源。

## 二、准备真实资产

固定一个可追溯的官方版本，而不是每次使用漂移的main：

```bash
git clone https://github.com/BoosterRobotics/booster_assets.git ~/booster_assets_r18
git -C ~/booster_assets_r18 checkout 38a0ae84b17743a8aa21511f69ed38e7d22d1664
```

本项目没有重新打包Booster mesh、没有替换缺失几何。已经有资产的用户可直接传入其路径；
验收报告会保存实际XML、mesh/texture的SHA256。参考commit只是获取路径，并不强制把不同的本地资产假称为同一版本。

```bash
python tools/verify_physics.py --phase assets \
  --asset-root ~/booster_assets_r18 --output reports/assets.json
```

检查内容：弧度声明、浮动根归属、22关节和电机、单位gear、开启的限幅、文件路径与内容指纹。
遇到未展开的include/class继承，明确拒绝，不按猜测解析。

**PASS仅代表资产合同通过。** 任意非空字节也能满足“文件存在”，只有实际MuJoCo编译才检查STL内容；
本项目的单位测试刻意使用标注为人工的字节样本，不能把这些样本当官方mesh。

## 三、零动作必须保持重置目标

保留的R17控制链是：

```text
q_raw = q_default + action * action_scale
q_execute = clamp(q_raw, soft_lower, soft_upper)
PD(q_execute, q_actual, velocity_actual)
```

官方该版本膝角范围为[0, 2.321]rad，项目重置膝角为0。机械地把硬限位两端各缩5%
会把最低执行目标变成0.11605rad，使action=0在第一控制步改变双膝目标。

当前执行边界仍优先使用内缩5%的软边界；若合法重置角位于软边界之外，则只把该侧边界
扩展到重置角。这样零动作严格保持重置目标，同时其他动作继续受执行边界约束。
`zero_action_changes`正常应为空；非空表示配置重新引入了零动作姿态跳变，必须重新验收。
若要改变初始站姿，应创建明确的实验配置并重新验证接触和平衡。

本项目PD仍按XML名义armature和effort计算，并不是完整Booster Train延迟/速度-力矩电机。
因此不同资料中的电机参数不能混用，本轮没有为追求更好结果切换参数来源。

## 四、先编译，再CPU物理步进

```bash
python tools/verify_physics.py --phase compile \
  --asset-root ~/booster_assets_r18 --output reports/compile.json

python tools/verify_physics.py --phase native \
  --asset-root ~/booster_assets_r18 --steps 100 --output reports/native.json
```

编译使用`make_k1_physics_cfg -> Scene -> compile`，然后显式应用`cfg.sim.mujoco`。
只调用Scene.compile不会自动应用训练配置中的仿真步长/积分器；这是R17静态工具遗漏的环节。

编译阶段逐项核对：
- nq=29、nv=28、nu=22；
- actuator绑定的关节和qpos/qvel地址，不假设ctrl顺序；
- Kp/Kd及符号、单位gear、力矩上限；
- init_state默认关节角与项目控制表相同。

native阶段是真正的MuJoCo CPU求解，但不经过Warp或mjlab环境step。它保持模型重力/碰撞、
在每个控制步执行10个2ms物理步，并刷新派生位姿后记录数据。
控制器仅为“零原始action经过原执行链后的PD目标”。

报告包括：完成步数、根高度、接触数量、最大穿透、关节目标误差、力矩限幅比例和引擎警告。
根高度低于项目已有fallen阈值0.24m或出现非有限状态/限幅错误，会FAIL，不重置刷满100步。

接触数量为**全场接触**，可能包含自碰撞，不能当成左右脚承重率。
该探针不衡量训练动作精度，也没有证明板面等新物体任务可行。

## 五、真实mjlab/Warp接通

```bash
python tools/verify_physics.py --phase cuda \
  --asset-root ~/booster_assets_r18 --device cuda:0 --steps 20 \
  --output reports/mjlab_cuda.json
```

使用正式K1MJLabBridge和K1PhysicsBridge。零动作20步、读取真实身体/接触观测，
出现结束信号时保留final-state，报告失败，不立即reset掩盖问题。没有CUDA会BLOCKED，不切换到假环境。

当前脚部观测使用maxforce单接触点：**不是足底总压力或CoP**。ContactData.force在global_frame=True时才是世界坐标。
R18没有把这组信息扩大宣传成通用多物体接触图。

## 六、退出码与报告

- 0：请求范围内检查PASS。
- 1：实际执行的检查FAIL。
- 2：缺包/缺资产/缺CUDA等BLOCKED，没有完成请求的检查。

报告`not_requested`是未请求的阶段，不是通过，也不混入CUDA跳过计数。
C/C++级崩溃或人工中断不会被当成通过，应同时保留终端日志。
每个CLI建议使用不同的`--output`，避免覆盖之前的报告。

## 七、本次执行范围

原交付环境的历史结论仍保存在 `reports/*_history`，不回写成当时已通过。2026-10-10
在整机已有环境和已有K1资产上补做了以下本地验收，全程没有下载或安装依赖：

- 资产审计与 MuJoCo 编译通过：nq=29、nv=28、nu=22；
- 原生 MuJoCo 零动作 100/100 控制步通过，根高度最低 0.54927m；
- 现有 mjlab 1.3 在 RTX 5080 上零动作 20/20 步通过，最低根高度 0.54998m，
  检测到34个逐步足接触标志；
- 真实 `mjlab_k1_real` 后端完成4步采样、一次PPO更新和一次动力学监督更新，数值有限；
- 当前全量回归为608 passed、18 CUDA skipped、427 subtests passed。

严格依赖检查仍会因版本不满足 mjlab 1.6 目标组合而 FAIL，这是正确门禁。兼容性实测记录见
`reports/current/local_mjlab_compatibility_smoke.json`。以上仍不证明策略训练成功或实机安全。

## 上游依据（源码核对，不是物理实测）

- mjlab 1.6.0版本与依赖：https://github.com/mujocolab/mjlab/blob/0fb8a681136be94ffc636a3dd423cabb97d91f10/pyproject.toml
- 安装指南：https://mujocolab.github.io/mjlab/main/source/installation.html
- Scene组装：https://github.com/mujocolab/mjlab/blob/0fb8a681136be94ffc636a3dd423cabb97d91f10/src/mjlab/scene/scene.py
- 接触传感器：https://github.com/mujocolab/mjlab/blob/0fb8a681136be94ffc636a3dd423cabb97d91f10/src/mjlab/sensor/contact_sensor.py
- 官方K1 XML：https://github.com/BoosterRobotics/booster_assets/blob/38a0ae84b17743a8aa21511f69ed38e7d22d1664/robots/K1/K1_22dof.xml
- MuJoCo Python原生接口：https://mujoco.readthedocs.io/en/stable/python.html
