"""R18：真正进入物理引擎前后的分层验收，不再把接口测试当成物理成功。

这个文件做什么：
    assets：只用 XML 和文件检查，列出实际 Kp/Kd、执行边界与资产指纹。
    compile：通过训练同源的 mjlab Scene 创建 MuJoCo 模型，核对每个执行器。
    native：用上一步模型在原生 MuJoCo CPU 上执行零策略动作，保留接触动力学。
    cuda：在真实 mjlab / MuJoCo Warp 环境走零策略动作，读取正式物理观测。

所有步骤都可以单独执行。缺包或缺资产是 BLOCKED，不会自动退到假环境。
PASS 仅表示“本次所选验收项通过”，不是动作训练收敛、实机安全或零样本成功。
默认只检查环境版本。实际编译/步进必须显式选择阶段，不能导入模块就启动物理。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any
import xml.etree.ElementTree as ET

from .mjlab_k1 import official_xml_path, read_k1_joint_specs, k1_control_profile

# 这是已核对上游源码的兼容目标，不是“已在本环境安装成功”的声明。
MJLAB_VERSION = "1.6.0"
MJLAB_COMMIT = "0fb8a681136be94ffc636a3dd423cabb97d91f10"
ASSET_REFERENCE_COMMIT = "38a0ae84b17743a8aa21511f69ed38e7d22d1664"
PHASES = ("dependencies", "assets", "compile", "native", "cuda")


def _record(phase: str, status: str, reason: str, **details: Any) -> dict[str, Any]:
    """功能：统一记录结果，避免某条检查失败却还打印总体 PASS。

    输入：阶段、PASS/FAIL/BLOCKED、原因和可序列化的测量结果。
    输出：字典；最终写入 JSON。NOT_RUN 属于报告中未请求的阶段，不伪装成跳过测试。
    """
    return {"phase": phase, "status": status, "reason": reason, **details}


def sha256_file(path: Path) -> str:
    """功能：给 XML 或 mesh 计算内容指纹，记录本次到底用了哪份资产。

    输入：实际存在的文件路径。输出：64位十六进制字符串。
    用途：另一台机器复现时比较文件内容；同名文件不代表同一模型。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compatible(distribution: str, version: str) -> bool:
    """功能：检查本项目核对过的仿真依赖范围，不执行网络安装。

    输入：发行包名与实际安装版本。输出：bool。
    mjlab固定1.6.0；MuJoCo和Warp后端使用其声明的3.11系列；warp-lang>=1.14。
    相容版本仍必须通过物理验收，版本号相同并不能证明计算结果正确。
    """
    from packaging.version import Version
    v = Version(version)
    if v.is_prerelease or v.is_devrelease:
        return False
    if distribution == "mjlab":
        return v == Version(MJLAB_VERSION)
    if distribution in ("mujoco", "mujoco-warp"):
        return Version("3.11.0") <= v < Version("3.12.0")
    if distribution == "warp-lang":
        return v >= Version("1.14.0")
    raise ValueError(f"未知仿真依赖 {distribution}")


def dependency_report() -> dict[str, Any]:
    """功能：检查安装版本和真实导入；发现包但无法导入，同样不能算通过。

    输入：无。输出：dependencies 阶段报告。
    不调用 pip，不修改用户 CUDA/PyTorch，不把缺依赖解释为软件代码已通过。
    """
    packages, missing, bad = {}, [], []
    for dist, module in (("mjlab", "mjlab"), ("mujoco", "mujoco"),
                         ("mujoco-warp", "mujoco_warp"), ("warp-lang", "warp")):
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            packages[dist] = {"installed": False}
            missing.append(dist)
            continue
        item: dict[str, Any] = {"installed": True, "version": version}
        try:
            item["compatible"] = _compatible(dist, version)
            importlib.import_module(module)
            item["importable"] = True
            if not item["compatible"]:
                bad.append(dist)
        except Exception as exc:
            item.update(importable=False, error=f"{type(exc).__name__}: {exc}")
            bad.append(dist)
        packages[dist] = item
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        bad.append("Python: mjlab 1.6.0 requires >=3.10,<3.14")
    status = "FAIL" if bad else ("BLOCKED" if missing else "PASS")
    return _record("dependencies", status,
                   "依赖版本和导入检查" if status == "PASS" else f"缺失={missing}；不相容/导入失败={bad}",
                   packages=packages, mjlab_source_commit=MJLAB_COMMIT)


def audit_k1_assets(asset_root: str | Path) -> dict[str, Any]:
    """功能：检查真实 XML、mesh/texture 是否齐全，并输出同源 PD 控制表。

    输入：booster_assets 仓库根目录，或 K1_22dof.xml 所在文件夹。
    输出：包含文件 SHA256、22 个关节控制参数、零动作偏移警告的 assets 报告。
    不编译 mesh，不计算接触。文件存在只能证明齐全，不能证明几何或动力学可行。
    只接受本适配器支持的自包含 K1 XML，不自动补 mesh、不下载替代模型。
    """
    xml = official_xml_path(asset_root)
    specs = read_k1_joint_specs(xml)
    root = ET.parse(xml).getroot()
    compiler = root.find("compiler")
    assert compiler is not None  # read_k1_joint_specs 已检查，非输入修复逻辑。
    if compiler.get("strippath", "false").lower() == "true":
        raise ValueError("资产启用了 strippath；本验收器不猜测路径重写结果")
    if len(root.findall("./worldbody/body")) != 1:
        raise ValueError("K1 资产必须只有 trunk 一个顶层身体，外部物体应独立注册")
    if len(root.findall("./worldbody/geom[@name='ground']")) != 1:
        raise ValueError("缺少或重复 ground；不能确认替换地面是否正确")
    # 本读取器只解析显式关节/电机参数；继承的物理默认值需人工展开后重审。
    if root.findall("./default//joint") or root.findall("./default//motor"):
        raise ValueError("关节/电机含 default 继承，本验收器不将继承值猜成默认值")
    if any(node.get("class") or node.get("childclass") for node in root.iter()):
        raise ValueError("K1 资产含 class/childclass 继承，需展开并核对后接入")

    base = xml.parent.resolve()
    files = {xml.name: {"size_bytes": xml.stat().st_size, "sha256": sha256_file(xml)}}
    missing: list[str] = []
    for tag, directory in (("mesh", compiler.get("meshdir", "")),
                           ("texture", compiler.get("texturedir", ""))):
        for node in root.findall(f"./asset/{tag}[@file]"):
            path = (base / directory / node.attrib["file"]).resolve()
            if not path.is_relative_to(base):
                raise ValueError("资产引用越过模型目录，无法生成可移植的完整资产指纹")
            relative = str(path.relative_to(base))
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(relative)
            else:
                files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    controls = k1_control_profile(specs)
    changes = [{"joint": c.name, "default_rad": c.default_pos,
                "zero_action_target_rad": c.zero_target,
                "difference_rad": c.zero_target-c.default_pos}
               for c in controls if not math.isclose(c.default_pos, c.zero_target, abs_tol=1e-12)]
    manifest_text = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return _record("assets", "BLOCKED" if missing else "PASS",
                   "缺少 mesh/texture" if missing else "仅资产文件及 XML 合同通过；尚未编译",
                   xml=str(xml), files=files, missing_files=sorted(set(missing)),
                   fingerprint=hashlib.sha256(manifest_text.encode()).hexdigest(),
                   controls=[asdict(c) for c in controls],
                   zero_action_changes=changes,
                   control_profile="xml_nominal_pd_r17_preserved",
                   warnings=(["零策略动作仍触发初始目标裁剪，不能把该探针称为保持初始姿态"]
                             if changes else []))


def _compile_scene(asset_root: str | Path):
    """功能：使用正式训练配置创建原生模型，不另写一套玩具 K1。

    输入：真实资产路径。输出：(cfg, scene, mjModel)。
    没有创建 MuJoCo Warp 世界或 GPU 环境。必须额外 apply(cfg.sim.mujoco)，
    因为只调用 Scene.compile() 不会自动应用训练配置的 timestep/integrator 等参数。
    """
    from mjlab.scene import Scene
    from .mjlab_physics import make_k1_physics_cfg
    cfg = make_k1_physics_cfg(str(Path(asset_root).expanduser()), num_envs=1)
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    return cfg, scene, model


def _model_control_layout(model, controls) -> dict[str, Any]:
    """功能：从编译结果读真实地址，核对每个 Kp/Kd、力矩限幅和单位传动。

    输入：真正的 mjModel，以及 XML 导出的控制表。
    输出：按执行器顺序的关节、qpos/qvel 地址、零动作目标和力矩上限。
    不能假定 ctrl 顺序就是关节顺序；不允许编译器悄悄增加或遗漏执行器。
    """
    import mujoco
    import numpy as np
    from ..robot_contract import get_robot_contract
    if (model.nq, model.nv, model.nu) != (29, 28, 22):
        raise ValueError(f"编译后不是单 K1 22DoF：{model.nq, model.nv, model.nu}")
    by_name = {c.name: c for c in controls}
    names, qadr, vadr, targets, efforts = [], [], [], [], []
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("存在非直接关节传动，不能把 ctrl 当作本项目的位置目标")
        j = int(model.actuator_trnid[i, 0])
        full_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not full_name or not full_name.startswith("robot/"):
            raise ValueError("执行器没有绑定 robot/ 下的关节")
        name = full_name.removeprefix("robot/")
        if name not in by_name or name in names:
            raise ValueError(f"未知或重复的执行器目标 {name}")
        c = by_name[name]
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name} 不是单自由度转动关节")
        if not np.allclose(model.actuator_gear[i], [1, 0, 0, 0, 0, 0]):
            raise ValueError(f"{name} 的 gear 不是单位传动")
        if (model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED or
                model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE):
            raise ValueError(f"{name} 不是可核对的原生位置 PD 执行器")
        if not np.isclose(model.actuator_gainprm[i, 0], c.kp, rtol=1e-6):
            raise ValueError(f"{name} 编译后 Kp 不符合控制表")
        if not np.allclose(model.actuator_biasprm[i, :3], [0, -c.kp, -c.kd], rtol=1e-6):
            raise ValueError(f"{name} 编译后 PD 符号或 Kd 不符合控制表")
        if not model.actuator_forcelimited[i] or not np.allclose(
                model.actuator_forcerange[i], [-c.effort, c.effort]):
            raise ValueError(f"{name} 编译后力矩边界不符合控制表")
        names.append(name); qadr.append(int(model.jnt_qposadr[j])); vadr.append(int(model.jnt_dofadr[j]))
        targets.append(c.zero_target); efforts.append(c.effort)
    get_robot_contract("k1_22dof").check_joint_set(names, source="编译执行器目标")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk")
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if root < 0 or key < 0:
        raise ValueError("编译模型缺少 robot/trunk 或 init_state")
    defaults = [by_name[name].default_pos for name in names]
    if not np.allclose(model.key_qpos[key, qadr], defaults):
        raise ValueError("编译模型的默认关节角与控制表不同")
    return dict(names=names, qpos_addresses=qadr, qvel_addresses=vadr,
                zero_targets=targets, effort_limits=efforts, root_body_id=int(root), key_id=int(key))


def compile_k1(asset_root: str | Path) -> dict[str, Any]:
    """功能：CPU 编译验收。输入资产路径，输出编译后尺寸与执行器地址。

    只证明模型可以编译、执行参数与配置相符；没有执行控制步，不证明能站住。
    """
    cfg, scene, model = _compile_scene(asset_root)
    layout = _model_control_layout(model, k1_control_profile(read_k1_joint_specs(official_xml_path(asset_root))))
    return _record("compile", "PASS", "训练同源场景编译和执行器合同通过，尚未步进",
                   nq=int(model.nq), nv=int(model.nv), nu=int(model.nu), nbody=int(model.nbody),
                   physics_dt=float(model.opt.timestep), decimation=cfg.decimation,
                   layout=layout)


def native_probe(asset_root: str | Path, *, steps: int = 100) -> dict[str, Any]:
    """功能：使用同源编译模型执行原生 MuJoCo CPU 的零动作探针。

    输入：资产路径、正整数控制步数；每个控制步按 cfg.decimation 执行物理步。
    输出：真实 qpos/速度/接触/力矩摘要与独立 hold 诊断。
    不使用神经策略、不修改摩擦、不冻结根节点、不关闭碰撞。
    原生 CPU probe 不等于 mjlab/Warp 验收；root下降超过项目终止高度会 FAIL。
    """
    import mujoco
    import numpy as np
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps 必须是正整数")
    cfg, scene, model = _compile_scene(asset_root)
    controls = k1_control_profile(read_k1_joint_specs(official_xml_path(asset_root)))
    layout = _model_control_layout(model, controls)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, layout["key_id"])
    mujoco.mj_forward(model, data)
    lower_height, observed_contacts, saturation_count, element_count = float("inf"), 0, 0, 0
    penetration = 0.0
    root = layout["root_body_id"]
    efforts = np.asarray(layout["effort_limits"])
    completed, violations, max_error = 0, [], 0.0
    contact_points = 0
    # ctrl 接受位置目标。这里精确重现零策略动作经过 scale/offset/clip 后的目标。
    data.ctrl[:] = layout["zero_targets"]
    for step in range(steps):
        for _ in range(cfg.decimation):
            mujoco.mj_step(model, data)
            # MuJoCo 派生位姿在积分前计算；刷新后才记录当前位置，防止差一物理帧。
            mujoco.mj_forward(model, data)
            arrays = (data.qpos, data.qvel, data.ctrl, data.qfrc_actuator, data.sensordata)
            if any(not np.isfinite(x).all() for x in arrays):
                violations.append("出现非有限物理状态"); break
            lower_height = min(lower_height, float(data.xpos[root, 2]))
            forces = np.abs(data.actuator_force)
            saturation_count += int(np.count_nonzero(forces >= efforts*(1-1e-5)))
            element_count += model.nu
            if np.any(forces > efforts + 1e-5):
                violations.append("实际执行器力超过声明的力矩边界"); break
            contact_points += int(data.ncon)
            observed_contacts += int(data.ncon > 0)
            if data.ncon:
                penetration = max(penetration, max(0.0, -float(np.min(data.contact.dist[:data.ncon]))))
        if violations:
            break
        completed += 1
        error = np.max(np.abs(data.qpos[layout["qpos_addresses"]]-np.asarray(layout["zero_targets"])))
        max_error = max(max_error, float(error))
        if lower_height < 0.24:
            violations.append("根部件低于0.24米；与项目 fallen 条件一致"); break
    warnings = np.asarray(data.warning.number).astype(int).tolist()
    if any(warnings):
        violations.append("MuJoCo 发出数值/容量等警告，请检查 warning_counts")
    expected = completed * cfg.decimation * cfg.sim.mujoco.timestep
    if not violations and not np.isclose(float(data.time), expected, atol=1e-9):
        violations.append("仿真时钟与请求步数不一致")
    return _record("native", "FAIL" if violations else "PASS",
                   "; ".join(violations) if violations else "原生CPU零动作探针通过；不是训练或Warp结果",
                   requested_steps=steps, completed_steps=completed, seconds_simulated=float(data.time),
                   minimum_root_height_m=lower_height if math.isfinite(lower_height) else None,
                   final_root_height_m=(float(data.xpos[root, 2]) if np.isfinite(data.xpos[root, 2]) else None),
                   substeps_with_any_contact=observed_contacts, summed_contact_count=contact_points,
                   maximum_penetration_m=penetration, maximum_joint_target_error_rad=max_error,
                   torque_saturation_element_fraction=saturation_count/max(1, element_count),
                   warning_counts=warnings, controller="raw_action_zero_then_affine_and_clip",
                   contact_note="统计全场接触，可能含自碰撞；不能当作左右脚承重率")


def cuda_probe(asset_root: str | Path, *, steps: int = 20, device: str = "cuda:0") -> dict[str, Any]:
    """功能：在真实 mjlab 中走物理步，检查正式桥返回的数据；不做 PPO。

    输入：资产路径、步数、CUDA设备。输出：真实执行状态与接触观测数量。
    当前探针为单K1，零原始动作；终止时先保留诊断，不自动reset刷出完整步数。
    没有CUDA会返回BLOCKED，绝不会退到假环境或原生CPU却仍称为mjlab通过。
    """
    import torch
    if not torch.cuda.is_available():
        return _record("cuda", "BLOCKED", "没有可用 CUDA，没有启动 mjlab 物理世界")
    if not device.startswith("cuda"):
        raise ValueError("cuda阶段必须使用CUDA设备；原生CPU验收使用native阶段")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps 必须是正整数")
    from mjlab.envs import ManagerBasedRlEnv
    from .mjlab_k1 import K1MJLabBridge
    from .mjlab_physics import K1PhysicsBridge, make_k1_physics_cfg
    cfg = make_k1_physics_cfg(str(asset_root), num_envs=1)
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device=device)
    completed, contacts = 0, 0
    try:
        bridge = K1MJLabBridge(env)
        physics = K1PhysicsBridge(bridge)
        bridge.reset()
        frame = physics.read()
        minimum_z = float(bridge.entity.data.root_link_pos_w[:, 2].min().item())
        for _ in range(steps):
            transition = bridge.step(torch.zeros(1, 22, device=device))
            frame = physics.read()  # 先读取final-state，不允许reset后的数据冒充执行结果。
            completed += 1
            # 第11列（从0起算的10）是真实接触标志，来自既有物理观测合同。
            contacts += int(frame.contact_features[..., 10].gt(0).sum().item())
            minimum_z = min(minimum_z, float(bridge.entity.data.root_link_pos_w[:, 2].min().item()))
            if (transition.terminated | transition.truncated).any().item():
                return _record("cuda", "FAIL", "零动作探针提前终止；请检查动力学，不等于代码异常",
                               requested_steps=steps, completed_steps=completed,
                               minimum_root_height_m=minimum_z,
                               terminated=transition.terminated.tolist(), truncated=transition.truncated.tolist())
        return _record("cuda", "PASS", "真实mjlab零动作与物理桥接通过；未训练策略",
                       completed_steps=completed, minimum_root_height_m=minimum_z,
                       detected_foot_contacts=contacts, body_obs_shape=list(frame.body_obs.shape))
    finally:
        env.close()


def _guard(phase: str, function, *args, **kwargs) -> dict[str, Any]:
    """功能：把实际异常保存为有原因的结果，绝不吞掉错误继续宣称通过。

    输入：阶段名和待运行的函数。输出：阶段报告；缺文件/模块BLOCKED，其余FAIL。
    KeyboardInterrupt和进程级崩溃不被吞掉；调用者应保留终端日志。
    """
    try:
        return function(*args, **kwargs)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        return _record(phase, "BLOCKED", f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        return _record(phase, "FAIL", f"{type(exc).__name__}: {exc}")


def run_validation(*, phase: str, asset_root: str | Path | None = None,
                   steps: int = 100, device: str = "cuda:0") -> dict[str, Any]:
    """功能：顺序执行所选阶段所需的检查，独立记录未请求/被阻塞的项目。

    输入：phase选择dependencies/assets/compile/native/cuda；后4项需要真实资产。
    输出：可写成JSON的总报告，含退出码：0=本阶段通过，1=失败，2=条件不足。
    请求cuda会先编译，但不擅自运行native；两种后端的结果必须分别验证。
    """
    if phase not in PHASES:
        raise ValueError(f"phase 必须属于 {PHASES}")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps 必须是正整数")
    results = []
    needed = (["assets"] if phase == "assets" else ["dependencies"])
    if phase in ("compile", "native", "cuda"):
        needed += ["assets", "compile"]
        if phase != "compile":
            needed.append(phase)
    for stage in needed:
        if stage == "dependencies":
            result = _guard(stage, dependency_report)
        elif stage == "assets":
            result = (_record(stage, "BLOCKED", "未提供真实资产路径") if asset_root is None else
                      _guard(stage, audit_k1_assets, asset_root))
        elif any(r["status"] != "PASS" for r in results):
            result = _record(stage, "BLOCKED", "前置检查未通过，没有执行此物理阶段")
        elif stage == "compile":
            result = _guard(stage, compile_k1, asset_root)
        elif stage == "native":
            result = _guard(stage, native_probe, asset_root, steps=steps)
        else:
            result = _guard(stage, cuda_probe, asset_root, steps=steps, device=device)
        results.append(result)
    status = ("FAIL" if any(r["status"] == "FAIL" for r in results) else
              "BLOCKED" if any(r["status"] != "PASS" for r in results) else "PASS")
    return dict(schema=1, evaluated_at=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), platform=platform.platform(), phase=phase,
                status=status, exit_code={"PASS": 0, "FAIL": 1, "BLOCKED": 2}[status],
                results=results, not_requested=[p for p in PHASES if p not in needed],
                scope="仅所选项目；不代表学习效果、足部精度、G1适配或实机安全")


def main(argv: list[str] | None = None) -> int:
    """功能：命令行入口，打印摘要、原子写入报告；不是生产模块内的自测。

    输入：终端参数；输出：JSON文件和严格退出码。默认仅检查依赖，不启动GPU。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="dependencies")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("reports/physics_validation.json"))
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps 必须大于0")
    report = run_validation(phase=args.phase, asset_root=args.asset_root,
                            steps=args.steps, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"phase": args.phase, "status": report["status"],
                      "report": str(args.output), "exit_code": report["exit_code"]}, ensure_ascii=False))
    return report["exit_code"]
