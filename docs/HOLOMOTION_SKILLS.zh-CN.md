# HoloMotion 版专家与单权重清单

核对日期：2026-11-16。共 25 个任务专家，均绑定官方 G1 v1.4 `model_14000`，公共哈希和观测／动作接口哈希一致。木梁及自训练基座的 competitive suite31 不在此清单中。

一个任务条目对应动作与场景组合；同类动作在不同场景或不同参考接口下可以有不同专家，25 个条目不等于 25 类完全不同的动作。

| 序号 | 任务／场景 | 专家 ID | 十五场景复现包范围 |
|---|---|---|---|
| 1 | 滑滑板 | `husky_skateboard` | 已完成闭环回放 |
| 2 | 网球接球回击 | `athlete_tennis` | 已完成闭环回放 |
| 3 | 回旋踢 | `roundhouse` | 尚未纳入该包验收 |
| 4 | 按速度指令走路、转向与停止 | `velocity_walk` | 尚未纳入该包验收 |
| 5 | 连续上楼梯（MimicX） | `mimicx_stairs` | 已完成闭环回放 |
| 6 | 上楼梯（Instinct） | `parkour_stairs_up` | 尚未纳入该包验收 |
| 7 | 下楼梯（Instinct） | `parkour_stairs_down` | 尚未纳入该包验收 |
| 8 | 上高台阶 | `parkour_stairs_up_high` | 尚未纳入该包验收 |
| 9 | 下高台阶 | `parkour_stairs_down_high` | 尚未纳入该包验收 |
| 10 | 斜坡行走 | `parkour_slope` | 尚未纳入该包验收 |
| 11 | 不平路面行走 | `parkour_rough` | 已完成闭环回放 |
| 12 | 跨沟 | `parkour_gaps` | 已完成闭环回放 |
| 13 | 跨障碍 | `parkour_obstacles` | 尚未纳入该包验收 |
| 14 | PMT 优化参考地形动作 | `pmt_optimized_scene0` | 尚未纳入该包验收 |
| 15 | 搬箱子 | `ultra_largebox` | 已完成闭环回放 |
| 16 | 平地前滚翻 | `diveroll_lab_no_wall` | 已完成闭环回放 |
| 17 | 斜坡前滚翻 | `diveroll_ramp_no_wall` | 已完成闭环回放 |
| 18 | 爬上台阶 | `knee_climb_no_wall` | 尚未纳入该包验收 |
| 19 | 跳坐平台 | `jump_sit_no_wall` | 已完成闭环回放 |
| 20 | 翻越（无墙场景） | `roll_vault_no_wall` | 已完成闭环回放 |
| 21 | 动作参考下楼梯（MimicX） | `mimicx_descent` | 已完成闭环回放 |
| 22 | 武术套路 | `mimicx_kungfu` | 已完成闭环回放 |
| 23 | 舞蹈 | `mimicx_dance` | 已完成闭环回放 |
| 24 | 足球控球动作跟踪 | `mimicx_football` | 已完成闭环回放 |
| 25 | 翻越（有墙场景） | `roll_vault_wall` | 已完成闭环回放 |

上表最后一列表示十五场景复现包的覆盖范围；其他十个专家有历史原生实验记录，但尚未纳入这份场景包重新验收。单文件网络检查也不能替代物理任务成功率测试。

## 一个权重包含什么

完整本地权重：一份冻结公共 Actor（含原生公共专家、Attention、归一化和动作头）＋全部 25 组新增 FFN/router＋每组原始合并偏置和上下文接口契约。不是 25 个完整策略，也不会在切换时重载网络。

每次推理最多开启一个新增专家，其他新增专家不参与计算；全部关闭时走原公共路径。不同机器人共享这份 Actor 参数，但注意力缓存、动作历史和场景观测分别保存。路由保持原检查点的连续 anchor-group 组合，公共组内部仍是原生 Top-2。

场景几何、参考动作和外部冻结编码器资源仍需配套提供；这些输入资源与 Actor 权重是两回事。不能把一个 `.pt` 文件说成整个仿真环境。

## 运行与发布

本地完整权重约 1.69 GB，位于源代码目录之外。私有 Release 提供约 53 MB 的单文件专家包，官方基座仍从原发布地址下载；用户在本地一次性生成包含基座与全部专家的权重。

```bash
python tools/materialize_holomotion_bank.py \
  --experts checkpoints/holomotion_experts/pera_25skills.experts.pt \
  --base-dir /path/to/official-holomotion-v1.4 \
  --output /path/outside/PERA/pera_holomotion_25skills.pt
```

使用单个完整权重一键运行十五场景，分三组串行回放，不需要再传官方基座目录或逐个插件文件：

```bash
bash scripts/replay_all_scenes.sh \
  --bank /path/outside/PERA/pera_holomotion_25skills.pt \
  --device cuda:0 --video \
  --output runs/bank25_demo
```

也可用 `--bank checkpoints/holomotion_experts/pera_25skills.experts.pt --base-dir /path/to/official-holomotion-v1.4` 直接在内存中组装同一个 Actor。

完整来源、原始权重 SHA256、偏置、上下文和历史评估备注见 [holomotion_experts25.json](../configs/holomotion_experts25.json)。本次没有新训练；分发与下载验收状态见 [复现报告](../provenance/SCENE_REPRODUCTION_CHECK.json)。

## 本次核验结果

完整单权重和仅专家单权重均已对全部 25 个专家完成逐项网络输出检查：每项连续 3 步、批量 2，动作与原独立插件严格一致；其他专家的 FFN 调用为零，拒绝同时开启两个专家，关闭后公共输出完全恢复。

完整单文件此前在七区域场景完成 12 秒物理回放，所有任务记录与此前原演示逐帧一致；全程只加载这一份 Actor 权重，教师动作调用为零。没有新增训练。详细记录见 [HOLOMOTION_BANK25_CHECK.json](../provenance/HOLOMOTION_BANK25_CHECK.json)。
