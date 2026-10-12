"""第 11 模块：Booster K1 (22DoF) → mjlab 的真实仿真接口与安全烟雾测试。

本文件专注于物理环境接通：官方 K1 MJCF → 原生位置 PD → 50 Hz 控制、
联合关节顺序审计、环境读取/原始动作映射、手动 reset、静态站姿参考演示。

不含动作数据重定向、跟踪奖励或 PPO rollout collector。无奖励时不能开始学习。
训练循环后续仍需提供真实 reference、reward、next_value、history prefix。

运行（MJLab/CUDA 已安装、官方 booster_assets 已克隆）：
python -m unified_motion.envs.mjlab_k1 --asset-root /path/booster_assets --steps 100
python -m unified_motion.envs.mjlab_k1 --asset-root /path/booster_assets --controller actor --steps 20

兼容目标：mjlab 1.6.0（0fb8a681136be94ffc636a3dd423cabb97d91f10）。
此处沿用 XML 名义惯量与力矩构造接通用 PD，不声称等价于 Booster Train 的延迟电机。
本模块除了创建环境的函数外不主动导入 mjlab，方便无 GPU 的接口单元测试。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import torch

from ..observations import ObservationBuilder, RobotState
from ..reference_observations import (
    ReferenceObservation,
    ReferenceObservationBuilder,
    ReferenceWindow,
)
from ..robot_contract import get_robot_contract

ROBOT_ID = "k1_22dof"


@dataclass(frozen=True)
class K1JointSpec:
    """从官方 XML 读出的物理上限；名字是唯一 ID，拒绝按数字下标猜顺序。"""

    name: str
    lower: float
    upper: float
    armature: float
    effort: float


def official_xml_path(asset_root: str | Path) -> Path:
    """接受仓库根目录或 K1 模型文件所在目录，不把资产复制到项目里。"""
    path = Path(asset_root).expanduser().resolve()
    candidates = (path / "robots/K1/K1_22dof.xml", path / "K1_22dof.xml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到官方 K1_22dof.xml；请克隆 BoosterRobotics/booster_assets，"
        "并用 --asset-root 指向其仓库根目录。"
    )


def read_k1_joint_specs(xml_file: str | Path) -> tuple[K1JointSpec, ...]:
    """解析/检查 XML 的 22 个有角度限位的受控关节及对应的力矩电机。"""
    root = ET.parse(xml_file).getroot()
    # MuJoCo 的默认角度单位是度。我们后面的计算全部用弧度，不能猜单位。
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle", "degree") != "radian":
        raise ValueError("当前 K1 读取器要求显式 compiler angle='radian'")
    # 本读取器不会展开 include/class 继承。发现这类新资产，先审计而不是误读。
    if root.findall(".//include"):
        raise ValueError(
            "当前 K1 原始 XML 读取器不支持 include，请先显式展开并审计"
        )
    robot = get_robot_contract(ROBOT_ID)
    if root.tag != "mujoco":
        raise ValueError("K1 模型应为 MuJoCo MJCF 文件")
    if root.find("./worldbody/body[@name='trunk']") is None:
        raise ValueError("官方 K1 根 body 应为 trunk")
    all_nodes = root.findall(".//worldbody//joint")
    free_nodes = [
        x for x in all_nodes if x.get("type") == "free"
    ] + root.findall(".//worldbody//freejoint")
    if len(free_nodes) != 1:
        raise ValueError("当前 K1 适配器要求恰好一个浮动根")
    trunk = root.find("./worldbody/body[@name='trunk']")
    if trunk is not None and free_nodes[0] not in list(trunk):
        raise ValueError("浮动关节必须直接挂在 trunk 上，不能把其他部件误当根")
    all_joints = [x for x in all_nodes if x.get("type") != "free"]
    if any(x.get("type", "hinge") != "hinge" for x in all_joints):
        raise ValueError(
            "当前 K1 合同只支持单自由度转动关节，不能传入 ball/slide"
        )
    names = [x.get("name") for x in all_joints]
    robot.check_joint_set(names, source="K1 MJCF")
    if len(all_joints) != robot.dof:
        raise ValueError("K1 MJCF 的非 free joints 必须恰好 22 个")
    motors = root.findall("./actuator/motor")
    if (
        len(root.findall("./actuator/*")) != robot.dof
        or len(motors) != robot.dof
    ):
        raise ValueError(
            "K1 原始 XML 必须有 22 个直接电机；未知执行器类型不能静默替换"
        )
    motor_by_joint: dict[str, ET.Element] = {}
    for motor in motors:
        if motor.get("forcelimited", "auto").lower() == "false":
            raise ValueError(
                "电机显式关闭了力矩限幅，不能把 forcerange 当成已启用上限"
            )
        gear = tuple(map(float, motor.get("gear", "1").split()))
        if not gear or gear[0] != 1.0 or any(v != 0.0 for v in gear[1:]):
            raise ValueError(
                "原始电机 gear 必须为单位传动，不能把电机侧力矩当关节侧力矩"
            )
        joint_name = motor.get("joint")
        if not joint_name or joint_name in motor_by_joint:
            raise ValueError("K1 电机必须每个关节唯一对应")
        motor_by_joint[joint_name] = motor
    robot.check_joint_set(tuple(motor_by_joint), source="K1 XML 电机")
    specs: list[K1JointSpec] = []
    for joint in all_joints:
        if joint.get("limited", "auto").lower() == "false":
            raise ValueError(
                "关节显式关闭了角度限位，不能把 range 当成已启用上限"
            )
        name = joint.attrib["name"]
        limits = joint.get("range")
        forces = motor_by_joint[name].get("forcerange")
        armature = float(joint.get("armature", "0"))
        if not limits or not forces:
            raise ValueError(f"{name} 缺关节范围或力矩上限")
        lo, hi = map(float, limits.split())
        f_lo, f_hi = map(float, forces.split())
        if (
            not all(map(math.isfinite, (lo, hi, f_lo, f_hi, armature)))
            or lo >= hi
            or f_lo >= 0
            or f_hi <= 0
            or armature <= 0
        ):
            raise ValueError(f"{name} 非法角度/力矩/armature")
        effort = min(-f_lo, f_hi)
        specs.append(K1JointSpec(name, lo, hi, armature, effort))
    return tuple(specs)


def make_k1_entity_spec(xml_file: str | Path):
    """去掉官方 XML 自带地面和 direct-torque motors，让 mjlab 添加自己的 PD。

    不改变机器人刚体、关节、惯性、碰撞、mesh 和传感器定义。
    mjlab 的 SceneCfg.terrain 会单独创建地面，避免重复地面碰撞。
    """
    read_k1_joint_specs(
        xml_file
    )  # 不绕过读取器对结构、角度单位和电机语义的校验。
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml_file))
    native_ground = spec.geom("ground")
    if native_ground is None:
        raise ValueError(
            "预期官方 K1 XML 包含 ground geom，资产版本需要重新审计"
        )
    spec.delete(native_ground)
    if len(spec.actuators) != 22:
        raise ValueError("原生 K1 actuator 数不为22，停止修改资产")
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    if len(spec.actuators) != 0:
        raise RuntimeError("删除 XML 内置 direct-torque actuator 失败")
    return spec


def _joint_hz(name: str) -> tuple[float, float]:
    """基础烟雾测试 PD 频率/阻尼；不是经过调参验证的最终机器人控制器。"""
    if any(k in name for k in ("hip", "knee", "ankle")):
        return 4.0, 1.5 if "knee" not in name else 1.0
    return 10.0, 2.0


@dataclass(frozen=True)
class K1ControlSpec:
    """一个关节真正交给位置控制器的参数表，不包含网络权重。

    输入来源：官方 XML 的限位、惯量、力矩，以及本项目的接通用频率设置。
    输出用途：mjlab 配置和原生 MuJoCo 验收都读取同一张表，避免各算各的。
    default_pos 是重置角度；zero_target 是零动作经过执行限位后的目标角度。
    执行限位优先采用硬限位内缩5%的软边界，但必须包含合法的重置角度，
    避免 action=0 在第一控制步制造姿态跳变。
    """

    name: str
    kp: float
    kd: float
    effort: float
    armature: float
    action_scale: float
    default_pos: float
    soft_lower: float
    soft_upper: float
    zero_target: float


def k1_control_profile(
    specs: Sequence[K1JointSpec],
) -> tuple[K1ControlSpec, ...]:
    """把 XML 参数转换为本项目的名义 PD 表；不启动仿真、不调整任何网络。

    输入：read_k1_joint_specs() 返回的22个关节，顺序可不同。
    输出：同顺序的 K1ControlSpec；角度 rad、力矩 N·m、惯量 kg·m²。
    Kp=J*(2πf)²，Kd=2ζJ*(2πf)，action_scale=0.25*effort/Kp。
    保留 R17 初始姿态，不偷偷把膝盖改成另一种站姿；零动作保持该重置目标。
    """
    robot = get_robot_contract(ROBOT_ID)
    robot.check_joint_set(
        tuple(item.name for item in specs), source="K1控制表"
    )
    default = {
        "left_shoulder_roll_joint": -1.3,
        "right_shoulder_roll_joint": 1.3,
    }
    result = []
    for item in specs:
        hz, ratio = _joint_hz(item.name)
        omega = 2.0 * math.pi * hz
        kp = item.armature * omega**2
        kd = 2.0 * ratio * item.armature * omega
        span = item.upper - item.lower
        lower, upper = item.lower + 0.05 * span, item.upper - 0.05 * span
        q0 = default.get(item.name, 0.0)
        if not item.lower <= q0 <= item.upper:
            raise ValueError(
                f"{item.name} 的默认角度不在 XML 硬限位内，不能初始化"
            )
        # K1 膝关节的官方下限和项目 reset 角度都是 0。若机械地把两端各
        # 内缩 5%，零动作会立刻把双膝目标改成 0.11605 rad。真实 MuJoCo
        # 探针表明这个非零跳变会显著提前跌倒。执行边界仍使用软边界，
        # 但在 reset 姿态位于软边界外时只扩展到该合法的硬限位内姿态。
        lower, upper = min(lower, q0), max(upper, q0)
        result.append(
            K1ControlSpec(
                item.name,
                kp,
                kd,
                item.effort,
                item.armature,
                0.25 * item.effort / kp,
                q0,
                lower,
                upper,
                q0,
            )
        )
    return tuple(result)


def make_k1_mjlab_cfg(
    asset_root: str | Path,
    *,
    num_envs: int = 1,
    physics_dt: float = 0.002,
    decimation: int = 10,
    control_profile: Sequence[K1ControlSpec] | None = None,
):
    """按 mjlab ManagerBasedRlEnvCfg 官方接口构造 K1 环境配置。

    这里只构造环境配置（不启动 GPU）。动作是归一化位置增量：
    q_target = q_default + action * (0.25*effort/kp)，之后按 XML 软角度裁剪。
    物理马达上限由内置 PD 的 effort_limit 负责；暂未包含速度-力矩曲线。
    """
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs 必须为正整数")
    if type(decimation) is not int or decimation < 1:
        raise ValueError("decimation 必须为正整数")
    if (
        isinstance(physics_dt, bool)
        or not isinstance(physics_dt, (int, float))
        or not math.isfinite(physics_dt)
        or physics_dt <= 0
    ):
        raise ValueError("physics_dt 必须为正有限数")
    specs = read_k1_joint_specs(official_xml_path(asset_root))
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.envs import mdp
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import SimulationCfg, MujocoCfg
    from mjlab.terrains import TerrainEntityCfg

    # 与静态/动态验收共用一份控制表；这些是模型名义值，不是实机标定结果。
    controls = (
        k1_control_profile(specs)
        if control_profile is None
        else tuple(control_profile)
    )
    if len(controls) != len(specs) or any(
        not isinstance(item, K1ControlSpec) for item in controls
    ):
        raise ValueError("control_profile 必须包含22个 K1ControlSpec")
    get_robot_contract(ROBOT_ID).check_joint_set(
        tuple(item.name for item in controls), source="K1 mjlab控制覆盖表"
    )
    actuators, scales, clips = [], {}, {}
    for item in controls:
        exact = f"^{re.escape(item.name)}$"
        actuators.append(
            BuiltinPositionActuatorCfg(
                target_names_expr=(exact,),
                stiffness=item.kp,
                damping=item.kd,
                effort_limit=item.effort,
            )
        )
        scales[exact] = item.action_scale
        clips[exact] = (item.soft_lower, item.soft_upper)

    xml_path = official_xml_path(asset_root)
    entity = EntityCfg(
        spec_fn=lambda: make_k1_entity_spec(xml_path),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.57),  # booster_train 官方 K1 的初始根高度。
            joint_pos={item.name: item.default_pos for item in controls},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9
        ),
    )

    # 接到 mjlab 原生 observation manager，关闭噪声；真正 Actor 仍由我们自己的
    # ObservationBuilder 构造。下面的内容用于 env.reset/step 管理和调试验收。
    actor_terms = {
        "ang_vel_b": ObservationTermCfg(func=mdp.base_ang_vel),
        "gravity_b": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    cfg = ManagerBasedRlEnvCfg(
        decimation=decimation,
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": entity},
        ),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=float(physics_dt))),
        observations={
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                concatenate_terms=True,
                enable_corruption=False,
            )
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=scales,
                use_default_offset=True,
                clip=clips,
            )
        },
        # 烟雾测试没有跟踪 reward。绝不拿 0 reward 假装训练成功。
        rewards={},
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fallen": TerminationTermCfg(
                func=mdp.root_height_below_minimum,
                params={"minimum_height": 0.24},
            ),
        },
        episode_length_s=10.0,
        auto_reset=False,  # PPO 时间截断使用终止前状态，不允许自动把它覆盖掉。
        scale_rewards_by_dt=False,
    )
    if abs(physics_dt * decimation - 0.02) > 1e-9:
        raise ValueError(
            "该 K1 参考桥按 50Hz 假设构造；请使用 timestep*decimation=0.02"
        )
    return cfg


@dataclass
class K1Transition:
    """env.step 真实返回。reward 为当前 cfg 定义的奖励（烟雾 cfg 恒为0）。"""

    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_start: torch.Tensor
    info: dict


class K1MJLabBridge:
    """将 mjlab 原生状态/动作顺序变为统一机器人合同；不会偷偷 auto-reset。"""

    def __init__(
        self, env: Any, *, reference_steps: int = 3, check_values: bool = True
    ) -> None:
        """把已创建的mjlab环境接到本项目数据合同。

        输入：env，必须手动reset、50Hz且包含K1机器人和joint_pos动作项。输出：无；读取真实名字并缓存关节/动作双向重排索引，不创建新物理引擎。"""
        if type(reference_steps) is not int or reference_steps < 1:
            raise ValueError("reference_steps 必须是正整数")
        if getattr(env.cfg, "auto_reset", None) is not False:
            raise ValueError(
                "PPO 接入必须使用 auto_reset=False 以保留 final state"
            )
        self.env = env
        self.robot_id = ROBOT_ID
        self.robot = get_robot_contract(ROBOT_ID)
        self.entity = env.scene["robot"]
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.joint_names = tuple(self.entity.joint_names)
        self.robot.check_joint_set(self.joint_names, source="mjlab 实体关节")
        self.action_term = env.action_manager.get_term("joint_pos")
        self.action_names = tuple(self.action_term.target_names)
        self.robot.check_joint_set(self.action_names, source="mjlab 动作通道")
        if env.action_manager.total_action_dim != self.robot.dof:
            raise ValueError("mjlab 的动作维度与 K1 22DoF 不一致")
        if abs(float(env.step_dt) - 0.02) > 1e-8:
            raise ValueError("K1 接口当前要求 50Hz")
        # 默认姿态允许多个环境，但必须完全一致；未来个体随机化要显式处理。
        raw_default = self.entity.data.default_joint_pos
        if raw_default.shape != (self.num_envs, self.robot.dof):
            raise ValueError("mjlab default_joint_pos 的形状不匹配")
        if not torch.allclose(
            raw_default, raw_default[:1].expand_as(raw_default)
        ):
            raise ValueError(
                "每个环境的默认关节姿态不同：当前 Builder 不支持此情况"
            )
        default_policy = self.robot.to_policy_order(
            raw_default[0], self.joint_names
        )
        self.observation_builder = ObservationBuilder(
            self.robot,
            self.joint_names,
            default_policy.clone(),
            check_values=check_values,
        ).to(device=self.device)
        self.reference_builder = ReferenceObservationBuilder(
            self.robot,
            self.robot.joint_names,
            default_policy.clone(),
            check_values=check_values,
        ).to(device=self.device)
        self.reference_steps = reference_steps
        self.env_ids = tuple(range(self.num_envs))
        self._begins = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pending = torch.zeros_like(self._begins)
        self._anchor_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=default_policy.dtype
        )
        self._anchor_quat = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=default_policy.dtype
        )
        self._anchor_quat[:, 0] = 1.0
        self._has_reset = False

    @property
    def episode_start(self) -> torch.Tensor:
        """返回当前观测是否属于新回合第一帧。

        输入：无。输出：[B]布尔快照；它不是执行动作之后才发生的done。"""
        return self._begins.clone()

    @property
    def action_dim(self) -> int:
        """返回该桥所需的原始动作宽度。

        输入：无。输出：22；动作必须按K1合同顺序，不能混入G1动作。"""
        return self.robot.dof

    def _ids(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> torch.Tensor:
        """将环境编号转换成合法索引。

        输入：env_ids=None表示全部，或一维不重复的编号序列。输出：同设备long索引；负数、越界和重复编号会被拒绝。"""
        if env_ids is None:
            return torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("env_ids 必须是一维整数索引")
        ids = ids.long()
        if (ids < 0).any().item() or (ids >= self.num_envs).any().item():
            raise ValueError("env_ids 越界")
        if ids.unique().numel() != ids.numel():
            raise ValueError("env_ids 不能重复")
        return ids

    @torch.no_grad()
    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """重置指定机器人并清理对应回合状态。

        输入：env_ids为None或[K]环境索引，首次必须全量重置。输出：无；更新默认姿态参考、新回合标志和待重置标志，其他环境不被清零。"""
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        if not self._has_reset and ids.numel() != self.num_envs:
            raise RuntimeError("首次必须重置全部环境；之后才允许只重置一部分")
        self.env.reset(env_ids=ids)
        data = self.entity.data
        self._anchor_pos[ids] = data.root_link_pos_w[ids]
        self._anchor_quat[ids] = data.root_link_quat_w[ids]
        self._begins[ids] = True
        self._pending[ids] = False
        self._has_reset = True

    @torch.no_grad()
    def observation(self) -> torch.Tensor:
        """实际状态；即使某行刚 done，也允许读取它的 reset 前 final state 做 GAE。"""
        self._ensure_ready(allow_pending=True)
        data = self.entity.data
        previous_policy = self.robot.to_policy_order(
            self.action_term.raw_action,
            self.action_names,
        )
        state = RobotState(
            joint_names=self.joint_names,
            joint_pos=data.joint_pos,
            joint_vel=data.joint_vel,
            root_ang_vel_b=data.root_link_ang_vel_b,
            gravity_b=data.projected_gravity_b,
        )
        return self.observation_builder(state, previous_policy)

    @torch.no_grad()
    def standing_probe_reference(self) -> ReferenceObservation:
        """仅供无数据时 smoke test 的静态目标，不是训练数据或动捕轨迹。

        站立姿态：关节朝初始 q_default，根位置固定为最近一次 reset 的位置。
        用当前真实身体位置/姿态作为观察坐标变换，因此不是作弊读取未来状态。
        """
        self._ensure_ready(allow_pending=True)
        data = self.entity.data
        b, n, h = self.num_envs, self.robot.dof, self.reference_steps
        pose = self.reference_builder.default_joint_pos.expand(b, h, n)
        dt = torch.arange(h, dtype=pose.dtype, device=pose.device) * float(
            self.env.step_dt
        )
        ref = ReferenceWindow(
            root_name=self.robot.root_name,
            joint_names=self.robot.joint_names,
            joint_pos=pose,
            joint_vel=torch.zeros_like(pose),
            root_pos_w=self._anchor_pos[:, None, :].expand(b, h, 3),
            root_quat_wxyz=self._anchor_quat[:, None, :].expand(b, h, 4),
            root_lin_vel_w=torch.zeros(
                b, h, 3, device=pose.device, dtype=pose.dtype
            ),
            root_ang_vel_w=torch.zeros(
                b, h, 3, device=pose.device, dtype=pose.dtype
            ),
            time_offsets_s=dt.expand(b, h),
            valid_mask=torch.ones(b, h, dtype=torch.bool, device=pose.device),
        )
        return self.reference_builder(
            ref,
            current_root_pos_w=data.root_link_pos_w,
            current_root_quat_wxyz=data.root_link_quat_w,
        )

    def _ensure_ready(self, *, allow_pending: bool = False) -> None:
        """检查是否已经完成首次reset。

        输入：无。输出：无；未初始化时禁止读取观测或参考，避免拿未定义物理状态决策。"""
        if not self._has_reset:
            raise RuntimeError("使用前必须显式 bridge.reset()")
        if not allow_pending and self._pending.any().item():
            raise RuntimeError(
                "有环境已经 done，必须先 bridge.reset(done_ids) 后再继续"
            )

    @torch.no_grad()
    def step(self, action_policy_order: torch.Tensor) -> K1Transition:
        """严格要求原始采样 action，不在 PPO 概率定义上隐式转换或裁剪。"""
        self._ensure_ready()
        if (
            not isinstance(action_policy_order, torch.Tensor)
            or action_policy_order.shape != (self.num_envs, self.robot.dof)
            or action_policy_order.device != self.device
            or action_policy_order.dtype
            != self.observation_builder.default_joint_pos.dtype
        ):
            raise ValueError(
                "action 必须为 [B,22]，与 mjlab 同设备同 dtype，且按策略顺序"
            )
        if not torch.isfinite(action_policy_order).all().item():
            raise ValueError("原始 action 包含 NaN/Inf")
        # 顺序转换不改变数值；scale、偏置和 clip 全部由 mjlab action term执行。
        action_sim = self.robot.to_simulator_order(
            action_policy_order, self.action_names
        )
        _, reward, terminated, truncated, info = self.env.step(action_sim)
        reward = reward.clone()
        terminated = terminated.to(torch.bool).clone()
        truncated = truncated.to(torch.bool).clone() & ~terminated
        for name, value in (
            ("reward", reward),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if value.shape != (self.num_envs,) or value.device != self.device:
                raise RuntimeError(f"mjlab {name} 的 batch 维/设备不正确")
        if not torch.isfinite(reward).all().item():
            raise RuntimeError("mjlab 奖励含 NaN/Inf")
        # 接通验收的必要条件：积分后根姿态与关节物理状态必须是有限值。
        # 即使这一帧终止，也不能把 NaN 当成普通 done 然后宣称 SMOKE PASS。
        data = self.entity.data
        for label, tensor in (
            ("root pose", data.root_link_pose_w),
            ("joint pos", data.joint_pos),
            ("joint vel", data.joint_vel),
        ):
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError(
                    f"mjlab 物理状态 {label} 含 NaN/Inf，立即停止"
                )
        previous = self._begins.clone()
        self._begins.fill_(False)
        self._pending.copy_(terminated | truncated)
        return K1Transition(reward, terminated, truncated, previous, info)

    def close(self) -> None:
        """关闭底层环境占用的资源。

        输入：无。输出：无；不保存策略或训练checkpoint，保存由训练入口负责。"""
        self.env.close()


def run_smoke(
    asset_root: str | Path,
    *,
    num_envs: int = 1,
    steps: int = 100,
    device: str = "cuda:0",
    controller: str = "zero",
) -> None:
    """仅证明环境 reset/step 与动作维度；不训练，也不宣称随机策略表现有效。"""
    if controller not in ("zero", "actor") or steps < 1:
        raise ValueError("controller 必须为 zero/actor，steps 必须大于0")
    cfg = make_k1_mjlab_cfg(asset_root, num_envs=num_envs)
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=device)
    bridge = K1MJLabBridge(env)
    model = None
    cache = None
    if controller == "actor":
        from ..models.actor_critic import MultiRobotActorCritic
        from ..models.transformer import TransformerConfig

        robot = get_robot_contract(ROBOT_ID)
        # 未训练的小模型仅验证接口；采用 deterministic 均值避免随机采样大幅摆动。
        config = TransformerConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            context_len=8,
            num_fine_experts=4,
            top_k=2,
            expert_hidden_dim=64,
            router_hidden_dim=32,
            check_values=False,
        )
        model = (
            MultiRobotActorCritic(
                {ROBOT_ID: robot},
                transformer_config=config,
                encoder_hidden_dim=64,
                critic_hidden_dim=64,
                check_values=False,
            )
            .to(device=bridge.device)
            .eval()
        )
    try:
        bridge.reset()
        for t in range(steps):
            obs = bridge.observation()
            if model is None:
                action = torch.zeros(
                    num_envs,
                    bridge.action_dim,
                    device=bridge.device,
                    dtype=obs.dtype,
                )
            else:
                reference = bridge.standing_probe_reference()
                # reset 标志传给注意力缓存，每个环境单独清掉旧回合历史。
                result, cache = model.act_step(
                    ROBOT_ID,
                    obs,
                    reference,
                    env_ids=bridge.env_ids,
                    cache=cache,
                    episode_start=bridge.episode_start,
                    deterministic=True,
                )
                action = result.action
            result = bridge.step(action)
            if t % 10 == 0 or t == steps - 1:
                print(
                    f"step={t + 1:04d} done={int((result.terminated | result.truncated).sum())} "
                    f"reward_sum={float(result.reward.sum()):.4f} "
                    f"root_z_mean={float(bridge.entity.data.root_link_pos_w[:, 2].mean()):.3f}"
                )
            done = (result.terminated | result.truncated).nonzero().flatten()
            if done.numel():
                bridge.reset(done)
        print(
            "SMOKE PASS：mjlab K1 完成真实物理步进。尚未进行 PPO/动作跟踪训练。"
        )
    finally:
        bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Booster K1 mjlab smoke test")
    parser.add_argument(
        "--asset-root", required=True, help="booster_assets 仓库目录"
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--controller", choices=("zero", "actor"), default="zero"
    )
    args = parser.parse_args()
    run_smoke(
        args.asset_root,
        num_envs=args.num_envs,
        steps=args.steps,
        device=args.device,
        controller=args.controller,
    )


if __name__ == "__main__":
    main()
