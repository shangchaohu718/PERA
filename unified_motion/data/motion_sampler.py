"""第 12 个模块：从动作文件里选出机器人接下来要模仿的参考动作。

这份代码只负责“读动作、选片段、按时间取动作”。不计算奖励，也不训练 PPO。

它和前面文件的关系：
    动作 NPZ → MotionLibrary → MotionSampler.reference_window()
             → 第 03 模块 ReferenceObservationBuilder
             → Encoder → Transformer → Actor

术语（后面每个函数都沿用这几个字母）：
    B = 并行机器人环境数量，例如 1024。
    H = 每次向网络展示的参考时间点数量，例如 5。
    N = 机器人受控关节数量，例如 K1 为 22、G1 29DoF 为 29。
    F = 一条动作总帧数，例如 500。
    M = 一条动作中保留的身体部件数量，例如骨盆、左右脚等。

重要前提：
    1. 必须使用已经重定向到目标机器人的轨迹。原始 LAFAN1 人体关节不能直接使用。
    2. 关节角单位 rad；位置米；速度 rad/s 或 m/s；世界坐标 z 向上。
    3. 四元数顺序是 (w,x,y,z)，表示身体坐标到世界坐标的旋转。
    4. 本文件不自动循环、拼接、补造动作。片段到头以后，未来目标标为无效。
    5. 参考世界位置对齐只在重置时计算一次，不能每帧重新对齐掩盖跟踪误差。
    6. 不包含 __main__ 或自测代码；正式验收由项目统一执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..reference_observations import ReferenceWindow
from ..robot_contract import RobotContract, get_robot_contract


# -------------------------- 一、很小的数学工具 --------------------------


def _unit_quaternion(q: torch.Tensor) -> torch.Tensor:
    """功能：将四元数变成长度为 1 的四元数，减少小数误差。

    输入：q，形状 [...,4]，依次为 w、x、y、z。
    输出：同形状的单位四元数。这个函数不改变旋转方向。
    """
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor,
                      fraction: torch.Tensor) -> torch.Tensor:
    """功能：补出两帧中间的朝向，防止旋转被错误地当成普通四个数字插值。

    输入：
        q0、q1：前后两帧的朝向，形状相同，最后一维为 4。
        fraction：插值比例，形状等于 q0.shape[:-1]，范围 0~1。
            0 表示完全取前一帧，1 表示完全取后一帧。
    输出：两帧之间的朝向，仍是 [...,4]，且长度为 1。
    """
    q0 = _unit_quaternion(q0)
    q1 = _unit_quaternion(q1)
    dot = (q0 * q1).sum(dim=-1)
    # q 和 -q 表示同一朝向。统一符号，选旋转角度更短的路径。
    q1 = torch.where((dot < 0)[..., None], -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sine = torch.sin(theta)
    safe_sine = sine.clamp_min(1e-7)
    weight0 = torch.sin((1.0 - fraction) * theta) / safe_sine
    weight1 = torch.sin(fraction * theta) / safe_sine
    spherical = weight0[..., None] * q0 + weight1[..., None] * q1
    # 两帧朝向几乎一样时，改用普通线性插值，避免除以很小的数。
    linear = (1.0 - fraction)[..., None] * q0 + fraction[..., None] * q1
    return _unit_quaternion(torch.where((dot > 0.9995)[..., None], linear, spherical))


def _quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """功能：把两个旋转叠加在一起。

    输入：a、b 是 [...,4] 四元数；前面的维度可以按 PyTorch 规则广播。
    输出：先按 b 的旋转，再按 a 的旋转得到的四元数 [...,4]。
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ), dim=-1)


def _yaw(q: torch.Tensor) -> torch.Tensor:
    """功能：读取朝向中的“水平转身角度”（绕世界竖直 z 轴）。

    输入：q，形状 [...,4]，顺序 wxyz。
    输出：角度 [...], 单位 rad；不是俯仰角，也不是关节角。
    """
    w, x, y, z = _unit_quaternion(q).unbind(-1)
    return torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))


def _rotate_z(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """功能：把世界坐标里的位置/速度绕竖直方向旋转一个角度。

    输入：
        vector：[B,H,3] 或 [B,H,M,3] 等，最后三个数字是 xyz。
        yaw：[B]，每个机器人环境一个旋转角度。
    输出：与 vector 相同形状的新 xyz；z 高度不变。
    """
    while yaw.ndim < vector.ndim - 1:
        yaw = yaw.unsqueeze(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y, z = vector.unbind(-1)
    return torch.stack((c*x - s*y, s*x + c*y, z), dim=-1)


def _apply_yaw_to_quat(q: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """功能：让参考动作整体转向指定方向，但不改变身体自己的俯仰/侧倾。

    输入：q 为 [B,H,4] 或 [B,H,M,4]；yaw 为 [B]。
    输出：旋转后的同形状四元数。
    """
    half = yaw / 2.0
    yaw_quat = torch.stack((torch.cos(half), torch.zeros_like(half),
                            torch.zeros_like(half), torch.sin(half)), dim=-1)
    while yaw_quat.ndim < q.ndim:
        yaw_quat = yaw_quat.unsqueeze(-2)
    return _unit_quaternion(_quaternion_multiply(yaw_quat, q))


def _string_scalar(value: np.ndarray, name: str) -> str:
    """功能：读取 NPZ 里的一条文本元信息，例如机器人 ID。

    输入：value 是 NPZ 中的数组；name 用于报告哪个字段出错。
    输出：一个非空字符串，避免把多个字符串误当成一个名字。
    """
    if value.ndim != 0 or value.dtype.kind not in ('U', 'S'):
        raise ValueError(f"{name} 必须是单个字符串，不能是对象数组")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode('utf-8')
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{name} 不能是空字符串")
    return result


def _string_names(value: np.ndarray, name: str) -> tuple[str, ...]:
    """功能：读取 NPZ 中按顺序保存的关节名或身体部件名。

    输入：value 为一维字符串数组，例如 ['left_hip...', 'right_hip...']。
    输出：名字组成的 tuple，顺序与原文件完全相同。
    """
    if value.ndim != 1 or value.dtype.kind not in ('U', 'S') or len(value) == 0:
        raise ValueError(f"{name} 必须是一维非空字符串数组")
    names = tuple(x.decode('utf8') if isinstance(x, bytes) else str(x)
                  for x in value.tolist())
    if any(not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError(f"{name} 不能有空名字或重复名字")
    return names


def _array(data, key: str) -> torch.Tensor:
    """功能：从 NPZ 读取指定的浮点数组，不偷偷把整数或坏数据当动作。

    输入：data 是已经打开的 np.load 文件，key 是字段名。
    输出：CPU 上的 torch.Tensor，保留原始数值精度。
    """
    if key not in data:
        raise ValueError(f"NPZ 缺少 {key!r}；不能自动补造运动数据")
    arr = np.asarray(data[key])
    if arr.dtype.kind != 'f' or arr.dtype.itemsize not in (4, 8):
        raise TypeError(f"{key} 必须使用 float32 或 float64")
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} 含 NaN 或 Inf")
    return torch.from_numpy(np.array(arr, copy=True))


# -------------------------- 二、一条动作代表什么 --------------------------


@dataclass(frozen=True)
class MotionClip:
    """功能：保存一条已经适配机器人关节的完整动作。

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
    """
    name: str
    robot_id: str
    root_name: str
    joint_names: tuple[str, ...]
    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_wxyz: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    body_names: tuple[str, ...] = ()
    body_pos_w: torch.Tensor | None = None
    body_quat_wxyz: torch.Tensor | None = None
    body_lin_vel_w: torch.Tensor | None = None
    body_ang_vel_w: torch.Tensor | None = None

    @classmethod
    def from_project_npz(cls, file: str | Path) -> 'MotionClip':
        """功能：读取我们项目自己的标准动作 NPZ 文件。

        输入：file，NPZ 文件路径。该文件必须自带 robot_id、root_name、
             joint_names、fps 和 6 组根/关节动作数组。
        输出：MotionClip；如果缺字段、文件损坏则立即报错。
        用途：适配 K1/G1 后保存一次，训练时反复使用。
        """
        path = Path(file).expanduser()
        with np.load(path, allow_pickle=False) as data:
            required = ('schema_version', 'units', 'world_frame', 'quaternion_order',
                        'robot_id', 'root_name', 'joint_names', 'fps')
            for key in required:
                if key not in data:
                    raise ValueError(f"{path} 缺少 {key!r} 元信息")
            version = np.asarray(data['schema_version'])
            if version.ndim != 0 or version.dtype.kind not in 'iu' or int(version.item()) != 1:
                raise ValueError('动作 NPZ 的 schema_version 必须是 1')
            expected_metadata = {
                'units': 'meters_radians_seconds',
                'world_frame': 'z_up',
                'quaternion_order': 'wxyz_body_to_world',
            }
            for key, expected_value in expected_metadata.items():
                if _string_scalar(np.asarray(data[key]), key) != expected_value:
                    raise ValueError(f'{key} 必须为 {expected_value!r}，请先统一坐标和单位')
            robot_id = _string_scalar(np.asarray(data['robot_id']), 'robot_id')
            root_name = _string_scalar(np.asarray(data['root_name']), 'root_name')
            joint_names = _string_names(np.asarray(data['joint_names']), 'joint_names')
            fps_data = np.asarray(data['fps'])
            if fps_data.ndim != 0 or fps_data.dtype.kind not in 'fi':
                raise ValueError('fps 必须是单个数值')
            fps = float(fps_data.item())
            optional_names = ()
            optional_data: dict[str, torch.Tensor | None] = {
                'body_pos_w': None, 'body_quat_wxyz': None,
                'body_lin_vel_w': None, 'body_ang_vel_w': None,
            }
            if 'body_names' in data:
                optional_names = _string_names(np.asarray(data['body_names']), 'body_names')
                for key in optional_data:
                    optional_data[key] = _array(data, key)
            elif any(k in data for k in optional_data):
                raise ValueError('提供 body_* 数据时必须同时提供 body_names')
            return cls(
                name=path.stem, robot_id=robot_id, root_name=root_name,
                joint_names=joint_names, fps=fps,
                joint_pos=_array(data, 'joint_pos'),
                joint_vel=_array(data, 'joint_vel'),
                root_pos_w=_array(data, 'root_pos_w'),
                root_quat_wxyz=_array(data, 'root_quat_wxyz'),
                root_lin_vel_w=_array(data, 'root_lin_vel_w'),
                root_ang_vel_w=_array(data, 'root_ang_vel_w'),
                body_names=optional_names, **optional_data,
            )

    @classmethod
    def from_mjlab_npz(
        cls, file: str | Path, *, robot_id: str,
        joint_names: Sequence[str], body_names: Sequence[str], fps: float,
    ) -> 'MotionClip':
        """功能：把 mjlab 原生 tracking NPZ 读成我们统一的动作数据。

        输入：
            file：mjlab 动作 NPZ 文件路径。
            robot_id：这份轨迹属于哪款机器人，必须明确指定。
            joint_names：这个 NPZ 的 joint_pos 各列真实对应哪些关节。
            body_names：这个 NPZ 的 body_pos_w 各列真实对应哪些身体部件。
            fps：该文件真实采样帧率，例如 50；绝不能凭文件名猜。
        输出：MotionClip，根部件数据从 body_names 中正确提取。

        警告：mjlab 原生文件可能不带关节/身体名字。调用者必须提供与
        生成文件时完全相同的顺序。仅凭维度相同并不能证明数据正确。
        """
        path = Path(file).expanduser()
        robot = get_robot_contract(robot_id)
        joints = tuple(joint_names)
        bodies = tuple(body_names)
        robot.check_joint_set(joints, source='mjlab motion NPZ')
        if len(set(bodies)) != len(bodies) or robot.root_name not in bodies:
            raise ValueError('body_names 必须无重复，且包含机器人 root_name')
        with np.load(path, allow_pickle=False) as data:
            # 如果文件碰巧自带了关节/身体名称，也必须与调用者传入的顺序一致。
            for key, expected in (('joint_names', joints), ('body_names', bodies)):
                if key in data and _string_names(np.asarray(data[key]), key) != expected:
                    raise ValueError(f'{key} 与文件内的顺序不一致')
            if 'fps' in data and abs(float(np.asarray(data['fps']).item()) - float(fps)) > 1e-5:
                raise ValueError('指定 fps 与文件元数据不一致')
            body_pos = _array(data, 'body_pos_w')
            body_quat = _array(data, 'body_quat_w')
            body_lin = _array(data, 'body_lin_vel_w')
            body_ang = _array(data, 'body_ang_vel_w')
            root_idx = bodies.index(robot.root_name)
            return cls(
                name=path.stem, robot_id=robot_id, root_name=robot.root_name,
                joint_names=joints, fps=float(fps),
                joint_pos=_array(data, 'joint_pos'),
                joint_vel=_array(data, 'joint_vel'),
                root_pos_w=body_pos[:, root_idx, :].clone(),
                root_quat_wxyz=body_quat[:, root_idx, :].clone(),
                root_lin_vel_w=body_lin[:, root_idx, :].clone(),
                root_ang_vel_w=body_ang[:, root_idx, :].clone(),
                body_names=bodies, body_pos_w=body_pos,
                body_quat_wxyz=body_quat,
                body_lin_vel_w=body_lin, body_ang_vel_w=body_ang,
            )

    @classmethod
    def from_named_mjlab_npz(cls, file: str | Path, *, robot_id: str) -> 'MotionClip':
        """读取自带 fps、joint_names、body_names 的 MJLab/HoloMotion 动作文件。

        该适配器不改写源文件，也不按维度猜名字。HoloMotion 的 ``body_quat_w``
        在 metadata 中必须明确声明 wxyz；根状态从 robot.root_name 对应 body 提取。
        """
        path = Path(file).expanduser()
        with np.load(path, allow_pickle=False) as data:
            for key in ('fps', 'joint_names', 'body_names', 'metadata'):
                if key not in data:
                    raise ValueError(f'{path} 缺少 {key!r}，不是带名字的 MJLab 动作')
            fps_value = np.asarray(data['fps'])
            if fps_value.ndim != 0 or fps_value.dtype.kind not in 'fi':
                raise ValueError('fps 必须是单个数值')
            joints = _string_names(np.asarray(data['joint_names']), 'joint_names')
            bodies = _string_names(np.asarray(data['body_names']), 'body_names')
            metadata_text = _string_scalar(np.asarray(data['metadata']), 'metadata')
            try:
                import json
                metadata = json.loads(metadata_text)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError('metadata 必须是 JSON 字符串') from error
            order = metadata.get('quaternion_order', metadata.get('root_quaternion_order', ''))
            if 'wxyz' not in str(order).lower():
                raise ValueError('HoloMotion/MJLab 动作必须明确声明四元数顺序为 wxyz')
        return cls.from_mjlab_npz(path, robot_id=robot_id, joint_names=joints,
                                  body_names=bodies, fps=float(fps_value.item()))

    def save_project_npz(self, file: str | Path) -> None:
        """功能：将一条动作保存为我们项目统一格式，方便之后训练直接加载。

        输入：file，准备保存的 NPZ 文件路径；父目录不存在时自动创建。
        输出：没有返回值；文件写到磁盘。不做 retarget 或坐标转换。
        """
        path = Path(file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, object] = {
            'schema_version': 1,
            'units': 'meters_radians_seconds',
            'world_frame': 'z_up',
            'quaternion_order': 'wxyz_body_to_world',
            'robot_id': self.robot_id, 'root_name': self.root_name,
            'joint_names': np.asarray(self.joint_names), 'fps': float(self.fps),
        }
        for name in ('joint_pos', 'joint_vel', 'root_pos_w', 'root_quat_wxyz',
                     'root_lin_vel_w', 'root_ang_vel_w'):
            arrays[name] = getattr(self, name).detach().cpu().numpy()
        if self.body_names:
            arrays['body_names'] = np.asarray(self.body_names)
            for name in ('body_pos_w', 'body_quat_wxyz',
                         'body_lin_vel_w', 'body_ang_vel_w'):
                value = getattr(self, name)
                if value is None:
                    raise ValueError('body_names 非空时，四组 body_* 数据必须齐全')
                arrays[name] = value.detach().cpu().numpy()
        np.savez_compressed(path, **arrays)


# -------------------------- 三、多条动作组成一个库 --------------------------


class MotionLibrary:
    """功能：把多条不同长度的动作放进一个库，并统一关节顺序。

    输入：
        robot：第 01 个模块的 RobotContract，比如 K1 22DoF。
        clips：一条或多条 MotionClip；必须属于同款机器人。
        device：保存和采样张量的设备，例如 'cuda:0'。
        dtype：float32 或 float64。通常使用 float32。
    输出：对象自身存有全部动作；不做随机抽样，随机抽样交给 MotionSampler。

    这里采用连续拼接保存，而不是每条动作补成同样长度；这样长短动作混合时
    不需要为很长的无效补位浪费 GPU 显存。
    """

    def __init__(self, robot: RobotContract, clips: Sequence[MotionClip],
                 *, device: str | torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32) -> None:
        """建立动作库，一次性检查各条动作并转换为策略关节顺序。

        输入：robot 机器人合同；clips 多条 MotionClip；device/dtype 存储设备与精度。
        输出：无直接返回，所有有效帧连续保存到本对象；不会替不同动作补相同长度。
        """
        if not isinstance(robot, RobotContract):
            raise TypeError('robot 必须是 RobotContract')
        if dtype not in (torch.float32, torch.float64):
            raise TypeError('dtype 只能是 torch.float32 或 torch.float64')
        clips = tuple(clips)
        if not clips or any(not isinstance(c, MotionClip) for c in clips):
            raise ValueError('clips 必须包含至少一条 MotionClip')
        self.robot = robot
        self.device = torch.device(device)
        self.dtype = dtype
        self.clip_names = tuple(c.name for c in clips)
        if any(not isinstance(s, str) or not s.strip() for s in self.clip_names) or len(set(self.clip_names)) != len(clips):
            raise ValueError('动作名称不能重复或为空；需要先给每条动作一个唯一名字')
        self.name_to_index = {name: i for i, name in enumerate(self.clip_names)}
        bodies_present = [bool(c.body_names) for c in clips]
        if any(bodies_present) and not all(bodies_present):
            raise ValueError('同一动作库中不能混用“有 body 数据”和“没有 body 数据”的动作')
        self.body_names = tuple(clips[0].body_names) if all(bodies_present) else ()
        if self.body_names and self.robot.root_name not in self.body_names:
            raise ValueError('body_names 应包含 robot.root_name')
        all_fields: dict[str, list[torch.Tensor]] = {key: [] for key in (
            'joint_pos', 'joint_vel', 'root_pos_w', 'root_quat_wxyz',
            'root_lin_vel_w', 'root_ang_vel_w',
        )}
        body_fields: dict[str, list[torch.Tensor]] = {key: [] for key in (
            'body_pos_w', 'body_quat_wxyz', 'body_lin_vel_w', 'body_ang_vel_w',
        )} if self.body_names else {}
        lengths, rates = [], []
        for clip in clips:
            # ID 必须在注册表里，并确实对应这一套关节合同。
            try:
                file_robot = get_robot_contract(clip.robot_id)
            except KeyError as error:
                raise ValueError(f'{clip.name} 的 robot_id={clip.robot_id!r} 没有注册') from error
            if file_robot != self.robot:
                raise ValueError(f'{clip.name} 的机器人合同与当前动作库不一致')
            if clip.root_name != self.robot.root_name:
                raise ValueError(f'{clip.name} 根部件应是 {self.robot.root_name!r}')
            self.robot.check_joint_set(clip.joint_names, source=f'动作 {clip.name}')
            if isinstance(clip.fps, bool) or not isinstance(clip.fps, (int, float)) or not np.isfinite(clip.fps) or clip.fps <= 0:
                raise ValueError(f'{clip.name} 的 fps 必须为正数')
            if (not isinstance(clip.joint_pos, torch.Tensor) or len(clip.joint_pos.shape) != 2 or clip.joint_pos.shape[0] < 2
                    or clip.joint_pos.shape[1] != robot.dof):
                raise ValueError(f'{clip.name} 的 joint_pos 必须为 [F,{robot.dof}] 且 F>=2')
            frames = clip.joint_pos.shape[0]
            expected = {
                'joint_pos': (frames, robot.dof), 'joint_vel': (frames, robot.dof),
                'root_pos_w': (frames, 3), 'root_quat_wxyz': (frames, 4),
                'root_lin_vel_w': (frames, 3), 'root_ang_vel_w': (frames, 3),
            }
            for name, shape in expected.items():
                val = getattr(clip, name)
                self._validate_tensor(clip.name, name, val, shape)
            if self.body_names:
                if len(set(clip.body_names)) != len(clip.body_names):
                    raise ValueError(f'{clip.name} 的 body_names 有重复名字')
                if set(clip.body_names) != set(self.body_names):
                    raise ValueError(f'{clip.name} 的 body_names 与其他动作不一致')
                m = len(self.body_names)
                for name in body_fields:
                    self._validate_tensor(clip.name, name, getattr(clip, name),
                                          (frames, m, 4 if 'quat' in name else 3))
                body_index = torch.tensor([clip.body_names.index(name) for name in self.body_names])
            quat_length = torch.linalg.vector_norm(clip.root_quat_wxyz, dim=-1)
            if ((quat_length - 1).abs() > 1e-2).any().item():
                raise ValueError(f'{clip.name} 的 root_quat_wxyz 不是单位四元数')
            if self.body_names:
                body_quat_length = torch.linalg.vector_norm(clip.body_quat_wxyz, dim=-1)
                if ((body_quat_length - 1).abs() > 1e-2).any().item():
                    raise ValueError(f'{clip.name} 的 body_quat_wxyz 不是单位四元数')
            # 每条动作的关节顺序都转换一次；训练时就不需要反复查关节名称。
            for name in all_fields:
                value = getattr(clip, name)
                if name.startswith('joint_'):
                    value = self.robot.to_policy_order(value, clip.joint_names)
                all_fields[name].append(value.detach().to(self.device, dtype=self.dtype))
            for name in body_fields:
                source_value = getattr(clip, name)
                value = source_value.index_select(1, body_index.to(source_value.device))
                body_fields[name].append(value.detach().to(self.device, dtype=self.dtype))
            lengths.append(frames)
            rates.append(float(clip.fps))
        # 所有帧连续保存，同时用每条动作的起始地址保证不会读串片段。
        for name, values in all_fields.items():
            setattr(self, name, torch.cat(values, dim=0).contiguous())
        for name, values in body_fields.items():
            setattr(self, name, torch.cat(values, dim=0).contiguous())
        self.clip_lengths = torch.tensor(lengths, device=self.device, dtype=torch.long)
        self.clip_fps = torch.tensor(rates, device=self.device, dtype=self.dtype)
        self.clip_starts = torch.cat((self.clip_lengths.new_zeros(1),
                                      self.clip_lengths.cumsum(0)[:-1]), dim=0)
        self.clip_durations_s = (self.clip_lengths - 1).to(self.dtype) / self.clip_fps

    @staticmethod
    def _validate_tensor(clip: str, name: str, value: torch.Tensor,
                         expected: tuple[int, ...]) -> None:
        """功能：检查数据的形状、浮点类型和有限性，发现坏轨迹立刻停。

        输入：clip 是动作名，name 是字段名，value 是张量，expected 是预期形状。
        输出：无返回；不符合要求时抛出异常。
        """
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f'{clip}.{name} 应为 {expected}')
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f'{clip}.{name} 必须是 float32 或 float64')
        if not torch.isfinite(value).all().item():
            raise ValueError(f'{clip}.{name} 包含 NaN/Inf')

    @classmethod
    def from_project_folder(cls, folder: str | Path, *, robot_id: str,
                            device: str | torch.device = 'cpu',
                            dtype: torch.dtype = torch.float32) -> 'MotionLibrary':
        """功能：把文件夹中的所有本项目标准 NPZ 动作载入同一个机器人动作库。

        输入：folder 是动作目录（支持子目录），robot_id 明确指定机器人的类型。
        输出：MotionLibrary。不能将 K1 与 G1 动作混在同一个库。
        """
        folder = Path(folder).expanduser()
        paths = sorted(folder.rglob('*.npz'))
        if not paths:
            raise FileNotFoundError(f'{folder} 没有找到标准动作 NPZ 文件')
        clips = []
        for path in paths:
            clip = MotionClip.from_project_npz(path)
            # 子目录允许相同文件名：用相对目录构成唯一名称。
            relative_name = path.relative_to(folder).with_suffix('').as_posix().replace('/', '__')
            clips.append(MotionClip(
                name=relative_name, robot_id=clip.robot_id, root_name=clip.root_name,
                joint_names=clip.joint_names, fps=clip.fps,
                joint_pos=clip.joint_pos, joint_vel=clip.joint_vel,
                root_pos_w=clip.root_pos_w, root_quat_wxyz=clip.root_quat_wxyz,
                root_lin_vel_w=clip.root_lin_vel_w, root_ang_vel_w=clip.root_ang_vel_w,
                body_names=clip.body_names, body_pos_w=clip.body_pos_w,
                body_quat_wxyz=clip.body_quat_wxyz,
                body_lin_vel_w=clip.body_lin_vel_w, body_ang_vel_w=clip.body_ang_vel_w,
            ))
        return cls(get_robot_contract(robot_id), clips, device=device, dtype=dtype)

    @classmethod
    def from_named_mjlab_file(cls, file: str | Path, *, robot_id: str,
                              device: str | torch.device = 'cpu',
                              dtype: torch.dtype = torch.float32) -> 'MotionLibrary':
        """把单个带名字的 HoloMotion/MJLab NPZ 直接建成动作库。"""
        clip = MotionClip.from_named_mjlab_npz(file, robot_id=robot_id)
        return cls(get_robot_contract(robot_id), (clip,), device=device, dtype=dtype)

    @classmethod
    def from_named_mjlab_files(cls, files: Sequence[str | Path], *, robot_id: str,
                               device: str | torch.device = 'cpu',
                               dtype: torch.dtype = torch.float32) -> 'MotionLibrary':
        """把多条具名 HoloMotion/MJLab NPZ 合成均匀采样动作库。"""
        paths = tuple(Path(file).expanduser() for file in files)
        if not paths:
            raise ValueError('files 必须至少包含一个动作 NPZ')
        clips = tuple(
            MotionClip.from_named_mjlab_npz(path, robot_id=robot_id)
            for path in paths
        )
        return cls(get_robot_contract(robot_id), clips, device=device, dtype=dtype)

    def _interpolate(self, clip_ids: torch.Tensor,
                     times_s: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """功能：从多条动作里按指定时间取值，不足一帧时自动插值。

        输入：
            clip_ids：[B]，每个环境选中的动作编号，从 0 开始。
            times_s：[B,H]，每个环境要读取的动作内部时间，单位秒。
        输出：
            data：字典，包含 joint_pos 等参考字段，形状 [B,H,N/3/4]。
            valid：[B,H]，True 表示该时间确实在对应动作片段内部。
        注意：超出片段的数据暂时取边界帧，调用方还必须根据 valid 屏蔽。
        """
        b, h = times_s.shape
        if (clip_ids.shape != (b,) or clip_ids.dtype != torch.long
                or clip_ids.device != self.device or times_s.device != self.device
                or times_s.dtype != self.dtype):
            raise ValueError('clip_ids 应为 [B] int64，times_s 应为 [B,H] 且与动作库同设备/精度')
        if not torch.isfinite(times_s).all().item():
            raise ValueError('times_s 不能包含 NaN/Inf')
        if ((clip_ids < 0) | (clip_ids >= len(self.clip_names))).any().item():
            raise ValueError('clip_ids 越界')
        durations = self.clip_durations_s[clip_ids, None]
        # 允许极小的浮点舍入误差；不能用下条动作的数据补齐当前动作。
        valid = (times_s >= -1e-6) & (times_s <= durations + 1e-6)
        safe_times = times_s.clamp_min(0.0).minimum(durations)
        frame = safe_times * self.clip_fps[clip_ids, None]
        lower = torch.minimum(torch.floor(frame).long(), (self.clip_lengths[clip_ids]-1)[:, None])
        upper = torch.minimum(lower + 1, (self.clip_lengths[clip_ids] - 1)[:, None])
        alpha = (frame - lower.to(self.dtype)).clamp(0.0, 1.0)
        # 重要：每次索引加上所属动作的起始地址，防止跨 clip 串帧。
        first = self.clip_starts[clip_ids, None] + lower
        second = self.clip_starts[clip_ids, None] + upper
        result: dict[str, torch.Tensor] = {}
        fields = ('joint_pos', 'joint_vel', 'root_pos_w', 'root_quat_wxyz',
                  'root_lin_vel_w', 'root_ang_vel_w')
        if self.body_names:
            fields += ('body_pos_w', 'body_quat_wxyz', 'body_lin_vel_w', 'body_ang_vel_w')
        for field in fields:
            buffer = getattr(self, field)
            left, right = buffer[first], buffer[second]
            if 'quat' in field:
                proportion = alpha if left.ndim == 3 else alpha.unsqueeze(-1)
                result[field] = _quaternion_slerp(left, right, proportion)
            else:
                proportion = alpha
                while proportion.ndim < left.ndim:
                    proportion = proportion.unsqueeze(-1)
                result[field] = left + proportion * (right - left)
        return result, valid


# -------------------------- 四、并行机器人怎样取不同动作 --------------------------


@dataclass(frozen=True)
class BodyReferenceWindow:
    """功能：保留未来奖励可能用到的身体部件姿态，不用关节角假装脚部位置。

    输出字段：
        body_names：这些列是什么身体部件，顺序固定。
        pos_w：[B,H,M,3]，世界位置。
        quat_wxyz：[B,H,M,4]，世界朝向。
        lin_vel_w、ang_vel_w：[B,H,M,3]，世界速度。
        valid_mask：[B,H]，True=真实参考帧。
    """
    body_names: tuple[str, ...]
    pos_w: torch.Tensor
    quat_wxyz: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor
    valid_mask: torch.Tensor


class MotionSampler:
    """功能：为每个并行环境记住抽到了哪条动作、当前播放到哪里。

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
    """

    def __init__(
        self, library: MotionLibrary, num_envs: int,
        *, step_dt: float = 0.02,
        future_offsets_s: Sequence[float] = (0.0, 0.04, 0.10, 0.20),
    ) -> None:
        """为每个并行环境准备独立的动作时钟。

        输入：library动作库、num_envs环境数B、step_dt控制周期秒数、future_offsets_s参考时间点偏移（从0开始严格递增）。
        输出：无；创建[B]片段编号、步数和世界对齐状态。创建后须reset分配动作，不能直接读取。"""
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('num_envs 必须是正整数')
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError('step_dt 必须为正有限数')
        offsets = tuple(float(v) for v in future_offsets_s)
        if (not offsets or any(not np.isfinite(v) or v < 0 for v in offsets)
                or any(y <= x for x, y in zip(offsets, offsets[1:]))):
            raise ValueError('未来时间偏移必须是非负且严格递增的数字')
        if not isinstance(library, MotionLibrary):
            raise TypeError('library 必须是 MotionLibrary')
        self.library = library
        self.num_envs = num_envs
        self.step_dt = float(step_dt)
        self.offsets_s = torch.tensor(offsets, device=library.device, dtype=library.dtype)
        self.clip_ids = torch.full((num_envs,), -1, device=library.device, dtype=torch.long)
        self.start_times_s = torch.zeros(num_envs, device=library.device, dtype=library.dtype)
        self.step_counts = torch.zeros(num_envs, device=library.device, dtype=torch.long)
        # 世界位置平移 + 绕 z 的角度，只在 reset 后根据实际初始身体计算一次。
        self.yaw_delta = torch.zeros_like(self.start_times_s)
        self.xy_shift = torch.zeros(num_envs, 2, device=library.device, dtype=library.dtype)
        self.sampling_weights = None

    def set_sampling_weights(self, weights):
        """Set positive per-clip curriculum weights; affect only subsequent episode resets."""
        value = torch.as_tensor(weights, device=self.library.device, dtype=self.library.dtype)
        if (value.shape != (len(self.library.clip_names),) or not torch.isfinite(value).all()
                or (value <= 0).any()):
            raise ValueError('motion sampling weights must be finite, positive and match clip count')
        self.sampling_weights = (value/value.mean()).detach().clone()

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """功能：把环境编号统一成 GPU/CPU 上的一维整数索引，并检查是否越界。

        输入：env_ids 可以是 [0,2,5]，或者 torch.long Tensor；None 表示全部环境。
        输出：shape [K] 的 int64 Tensor，K 是这次操作涉及的环境数量。
        """
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.library.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1 or env_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError('env_ids 必须是一维整数索引')
            ids = env_ids.to(device=self.library.device, dtype=torch.long)
        else:
            raw = tuple(env_ids)
            if any(type(x) is not int for x in raw):
                raise TypeError('env_ids 只能包含整数')
            ids = torch.tensor(raw, dtype=torch.long, device=self.library.device)
        if (((ids < 0) | (ids >= self.num_envs)).any().item()
                or ids.unique().numel() != ids.numel()):
            raise ValueError('env_ids 越界或有重复编号')
        return ids

    def _check_ready(self, env_ids: torch.Tensor) -> None:
        """功能：防止还没 reset 的环境就读取动作。

        输入：env_ids 为本次读取的环境编号 [K]。
        输出：无返回；没有有效 clip_id 时抛出异常。
        """
        if (self.clip_ids[env_ids] < 0).any().item():
            raise RuntimeError('采样器还没 reset：请先为这些环境分配参考动作')

    @torch.no_grad()
    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | None = None,
        *, clip_ids: torch.Tensor | Sequence[int] | None = None,
        start_times_s: torch.Tensor | Sequence[float] | None = None,
        random_start: bool = False,
        min_remaining_s: float = 0.0,
        current_root_pos_w: torch.Tensor | None = None,
        current_root_quat_wxyz: torch.Tensor | None = None,
    ) -> None:
        """功能：给刚重置的机器人安排一条动作和起始进度，并固定世界对齐。

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
        """
        ids = self._env_ids(env_ids)
        k = ids.numel()
        if k == 0:
            return
        dev, dtype = self.library.device, self.library.dtype
        if isinstance(min_remaining_s, bool) or not np.isfinite(min_remaining_s) or min_remaining_s < 0:
            raise ValueError('min_remaining_s 必须是非负有限秒数')
        if type(random_start) is not bool:
            raise TypeError('random_start 必须是 bool')
        if random_start and start_times_s is not None:
            raise ValueError('random_start 与显式起始时间不能同时指定')
        if clip_ids is None:
            # 先筛掉太短的动作，避免随机抽中它们后训练时不定期报错。
            candidates = (self.library.clip_durations_s >= float(min_remaining_s)).nonzero().flatten()
            if candidates.numel() == 0:
                raise ValueError('动作库中没有满足 min_remaining_s 的片段')
            chosen = candidates[(torch.randint(candidates.numel(), (k,), device=dev)
                if self.sampling_weights is None else
                torch.multinomial(self.sampling_weights[candidates], k, replacement=True))]
        else:
            chosen = torch.as_tensor(clip_ids, device=dev)
            if chosen.ndim != 1 or chosen.shape[0] != k or chosen.dtype not in (torch.int32, torch.int64):
                raise ValueError('clip_ids 必须是 [K] 的整数张量')
            chosen = chosen.long()
        if ((chosen < 0) | (chosen >= len(self.library.clip_names))).any().item():
            raise ValueError('clip_ids 包含不存在的动作编号')
        if not np.isfinite(min_remaining_s) or min_remaining_s < 0:
            raise ValueError('min_remaining_s 不能为负数或无穷大')
        max_start = self.library.clip_durations_s[chosen] - float(min_remaining_s)
        if (max_start < -1e-6).any().item():
            raise ValueError('选中的动作太短，不满足 min_remaining_s')
        max_start = max_start.clamp_min(0.0)
        if start_times_s is None:
            start = torch.rand(k, device=dev, dtype=dtype) * max_start if random_start else torch.zeros(k, device=dev, dtype=dtype)
        else:
            start = torch.as_tensor(start_times_s, device=dev, dtype=dtype)
            if start.shape != (k,) or not torch.isfinite(start).all().item():
                raise ValueError('start_times_s 必须是 [K] 的有限数值')
            if ((start < 0) | (start > max_start + 1e-6)).any().item():
                raise ValueError('起始时间越界或剩余动作长度不足')
        if (current_root_pos_w is None) != (current_root_quat_wxyz is None):
            raise ValueError('真实 root 位置与朝向必须同时提供，不能只给一个')
        # 先计算所有数据，检查成功后再改变运行时状态，避免部分写入失败。
        next_yaw = torch.zeros(k, device=dev, dtype=dtype)
        next_shift = torch.zeros(k, 2, device=dev, dtype=dtype)
        if current_root_pos_w is not None:
            pos = torch.as_tensor(current_root_pos_w, device=dev, dtype=dtype)
            quat = torch.as_tensor(current_root_quat_wxyz, device=dev, dtype=dtype)
            if pos.shape != (k, 3) or quat.shape != (k, 4):
                raise ValueError('真实 root 位置/朝向必须为 [K,3] 和 [K,4]')
            if not torch.isfinite(pos).all().item() or not torch.isfinite(quat).all().item():
                raise ValueError('真实 root 包含 NaN/Inf')
            if ((torch.linalg.vector_norm(quat, dim=-1) - 1.0).abs() > 1e-3).any().item():
                raise ValueError('真实 root 的四元数必须是单位四元数')
            sampled, _ = self.library._interpolate(chosen, start.unsqueeze(-1))
            root_pos = sampled['root_pos_w'][:, 0]
            root_quat = sampled['root_quat_wxyz'][:, 0]
            next_yaw = _yaw(quat) - _yaw(root_quat)
            rotated_xy = _rotate_z(root_pos.unsqueeze(1), next_yaw)[:, 0, :2]
            next_shift = pos[:, :2] - rotated_xy
        self.clip_ids[ids] = chosen
        self.start_times_s[ids] = start
        self.step_counts[ids] = 0
        self.yaw_delta[ids] = next_yaw
        self.xy_shift[ids] = next_shift

    def current_times_s(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """功能：得到每台机器人当前播放到动作的第几秒。

        输入：env_ids，选哪些环境；None 表示全部。
        输出：[K] 的秒数，片段到头后停在最后一帧，不进入下一条动作。
        """
        ids = self._env_ids(env_ids)
        self._check_ready(ids)
        time = self.start_times_s[ids] + self.step_counts[ids].to(self.library.dtype) * self.step_dt
        return time.minimum(self.library.clip_durations_s[self.clip_ids[ids]])

    def finished(self, env_ids: Sequence[int] | torch.Tensor | None = None, *,
                 min_remaining_s: float = 0.0) -> torch.Tensor:
        """功能：告诉任务管理器哪些动作已经播放到最后一帧。

        输入：env_ids；None 表示全部。
        输出：[K] 的 bool，True 表示动作结束，或剩余时间不足
            ``min_remaining_s``；需未来参考的任务应传入最大预览偏移。
        """
        ids = self._env_ids(env_ids)
        self._check_ready(ids)
        if (isinstance(min_remaining_s, bool) or not np.isfinite(min_remaining_s)
                or min_remaining_s < 0):
            raise ValueError('min_remaining_s 必须是非负有限秒数')
        times = self.start_times_s[ids] + self.step_counts[ids].to(self.library.dtype) * self.step_dt
        return (times + float(min_remaining_s)
                >= self.library.clip_durations_s[self.clip_ids[ids]] - 1e-6)

    @torch.no_grad()
    def advance(self, steps: int = 1,
                env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """功能：仿真成功执行控制步之后，推进参考动作的播放进度。

        输入：steps 是已经执行了多少个控制步，正常为 1；env_ids 是哪些环境。
        输出：无返回。内部用整数计数，不反复累加 0.02 造成累计浮点漂移。
        """
        if type(steps) is not int or steps < 1:
            raise ValueError('steps 必须是正整数')
        ids = self._env_ids(env_ids)
        self._check_ready(ids)
        self.step_counts[ids] += steps

    def _window_query(self, env_ids: Sequence[int] | torch.Tensor | None):
        """功能：准备一次参考查询用到的时间和动作编号。

        输入：env_ids 选择机器人；None 表示全部。
        输出：(ids, clips, times)，分别是 [K]、[K]、[K,H]。
        """
        ids = self._env_ids(env_ids)
        self._check_ready(ids)
        base = self.current_times_s(ids)
        times = base[:, None] + self.offsets_s[None, :]
        return ids, self.clip_ids[ids], times

    def _world_alignment(self, ids: torch.Tensor,
                         pose: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """功能：把动作世界位置/方向一次性变换到当前环境重置时的世界坐标。

        输入：ids [K]；pose 包含 root_pos_w、root_quat_wxyz 和世界速度。
        输出：包含同名字段的新字典，关节角不变，根位置/朝向/速度已对齐。
        """
        yaw = self.yaw_delta[ids]
        shifted = pose.copy()
        for name in ('root_pos_w', 'root_lin_vel_w', 'root_ang_vel_w'):
            shifted[name] = _rotate_z(pose[name], yaw)
        shifted['root_pos_w'] = shifted['root_pos_w'].clone()
        shifted['root_pos_w'][..., :2] += self.xy_shift[ids, None, :]
        shifted['root_quat_wxyz'] = _apply_yaw_to_quat(pose['root_quat_wxyz'], yaw)
        if self.library.body_names:
            for name in ('body_pos_w', 'body_lin_vel_w', 'body_ang_vel_w'):
                shifted[name] = _rotate_z(pose[name], yaw)
            shifted['body_pos_w'] = shifted['body_pos_w'].clone()
            shifted['body_pos_w'][..., :2] += self.xy_shift[ids, None, None, :]
            shifted['body_quat_wxyz'] = _apply_yaw_to_quat(pose['body_quat_wxyz'], yaw)
        return shifted

    @torch.no_grad()
    def reference_window(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> ReferenceWindow:
        """功能：读取当前与未来 H 个动作时间点，直接交给第03个文件。

        输入：env_ids 指定要读取的环境；None 表示全部 B 个环境。
        输出：ReferenceWindow，包含：
            joint_pos/joint_vel：[K,H,N]，已排成策略定义的关节顺序。
            root_pos_w：[K,H,3]，已对齐世界根位置。
            root_quat_wxyz：[K,H,4]，已对齐世界根朝向。
            root_lin_vel_w/root_ang_vel_w：[K,H,3]，速度也一起旋转。
            time_offsets_s：[K,H]，例如每行 [0,0.04,0.1,0.2]。
            valid_mask：[K,H]，动作结尾以 False 标记，不会跨片段。
        """
        ids, clips, query_times = self._window_query(env_ids)
        pose, valid = self.library._interpolate(clips, query_times)
        pose = self._world_alignment(ids, pose)
        # 所有补位数值都改为有限的零；四元数补位用单位四元数。
        mask = valid.unsqueeze(-1)
        identity = pose['root_quat_wxyz'].new_tensor((1.0, 0.0, 0.0, 0.0))
        return ReferenceWindow(
            root_name=self.library.robot.root_name,
            joint_names=self.library.robot.joint_names,
            joint_pos=torch.where(mask, pose['joint_pos'], 0.0),
            joint_vel=torch.where(mask, pose['joint_vel'], 0.0),
            root_pos_w=torch.where(mask, pose['root_pos_w'], 0.0),
            root_quat_wxyz=torch.where(mask, pose['root_quat_wxyz'], identity),
            root_lin_vel_w=torch.where(mask, pose['root_lin_vel_w'], 0.0),
            root_ang_vel_w=torch.where(mask, pose['root_ang_vel_w'], 0.0),
            time_offsets_s=self.offsets_s[None, :].expand(ids.numel(), -1).clone(),
            valid_mask=valid,
        )

    @torch.no_grad()
    def body_reference_window(
        self, env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> BodyReferenceWindow:
        """功能：取出动作的身体部件世界轨迹，供之后的足端/身体奖励使用。

        输入：env_ids 选择环境；None 表示全部。
        输出：BodyReferenceWindow，其中位置 [K,H,M,3]、朝向 [K,H,M,4]。
        要求：输入动作 NPZ 必须确实包含 body_names 和四组 body_*，否则拒绝。
        """
        if not self.library.body_names:
            raise RuntimeError('这个动作库没有 body_* 轨迹；不能凭关节角冒充足部世界位置')
        ids, clips, query_times = self._window_query(env_ids)
        pose, valid = self.library._interpolate(clips, query_times)
        pose = self._world_alignment(ids, pose)
        mask = valid[..., None, None]
        identity = pose['body_quat_wxyz'].new_tensor((1.0, 0.0, 0.0, 0.0))
        return BodyReferenceWindow(
            body_names=self.library.body_names,
            pos_w=torch.where(mask, pose['body_pos_w'], 0.0),
            quat_wxyz=torch.where(mask, pose['body_quat_wxyz'], identity),
            lin_vel_w=torch.where(mask, pose['body_lin_vel_w'], 0.0),
            ang_vel_w=torch.where(mask, pose['body_ang_vel_w'], 0.0),
            valid_mask=valid,
        )
