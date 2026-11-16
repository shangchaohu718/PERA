"""Isolated canonical-G1 simulation conversion and action convention audit.

This tests MuJoCo compatibility, not IsaacLab/real-robot equivalence. Source
URDF, public Actor and upstream ATHLETE files are never edited.
"""

import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from athlete.scripts import play_student_onnx as player

from pera.paths import ROOT

URDF = ROOT / "integrations/HoloMotion/assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf"
MOTORS = ROOT / "integrations/HoloMotion/holomotion/src/env/isaaclab_components/isaaclab_scene.py"
OFFICIAL_XML = ROOT / "third_party/ATHLETE/robots/replay_unitree_description/mjcf/g1.xml"
CONTRACT = ROOT / "configs/g1_actor_io_contract.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def motor_config():
    tree = ast.parse(MOTORS.read_text())
    (assignment,) = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "unitree_actuator_config_hardcoded"
            for t in n.targets
        )
    ]
    (call,) = assignment.value.values
    assert call.func.id == "UnitreeActuatorCfg"
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}


def by_joint(config, key, names):
    value = config[key]
    if isinstance(value, dict):
        result = []
        for name in names:
            matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
            assert len(matches) == 1, (key, name, matches)
            result.append(matches[0])
        return np.array(result, dtype=float)
    return np.full(len(names), value, dtype=float)


def action_contract(policy):
    meta = policy.metadata
    names = tuple(x.strip() for x in meta["joint_names"].split(","))
    holo = json.loads(CONTRACT.read_text())
    gather = np.array([names.index(n) for n in holo["joint_names"]])
    default = player._parse_float_metadata(meta, "default_joint_pos")
    scale = player._parse_float_metadata(meta, "action_scale")
    rng = np.random.default_rng(0)
    action = rng.normal(size=(128, 29))
    qtarget = default + scale * action
    labels = (qtarget[:, gather] - holo["action_offset"]) / holo["action_scale"]
    recovered = np.array(holo["action_offset"]) + np.array(holo["action_scale"]) * labels
    error = float(np.max(np.abs(qtarget[:, gather] - recovered)))
    assert error < 1e-12
    motor = motor_config()
    differences = []
    for i, name in enumerate(names):
        hi = holo["joint_names"].index(name)
        kp = by_joint(motor, "stiffness", [name])[0]
        kd = by_joint(motor, "damping", [name])[0]
        differences.append(
            {
                "name": name,
                "athlete_default": float(default[i]),
                "holo_default": holo["action_offset"][hi],
                "athlete_scale": float(scale[i]),
                "holo_scale": holo["action_scale"][hi],
                "athlete_kp": float(player._parse_float_metadata(meta, "joint_stiffness")[i]),
                "holo_kp": kp,
                "athlete_kd": float(player._parse_float_metadata(meta, "joint_damping")[i]),
                "holo_kd": kd,
            }
        )
    return {
        "joint_name_sets_equal": set(names) == set(holo["joint_names"]),
        "athlete_order": list(names),
        "holo_order": holo["joint_names"],
        "athlete_to_holo_gather": gather.tolist(),
        "round_trip_max_error_rad": error,
        "label_conversion": "a_holo=(gather(default_athlete+scale_athlete*a_athlete)-offset_holo)/scale_holo",
        "conversion_clips_actions": False,
        "dt_equal_50Hz": holo["step_dt"] == 0.02,
        "joints": differences,
        "canonical_urdf_sha256": sha(URDF),
        "motor_source_sha256": sha(MOTORS),
        "contract_sha256": sha(CONTRACT),
    }


def prepare_robot(output):
    spec = mujoco.MjSpec.from_file(str(URDF))
    # Resolve assets absolutely; the derived XML lives outside the URDF folder.
    mesh_dir = (URDF.parent / "../meshes").resolve()
    for mesh in spec.meshes:
        mesh.file = str(mesh_dir / Path(mesh.file).name)
    spec.compiler.meshdir = ""
    pelvis = spec.body("pelvis")
    pelvis.add_joint(name="floating_base_joint", type=mujoco.mjtJoint.mjJNT_FREE)
    pelvis.pos = [0.0, 0.0, 0.76]
    pelvis.add_site(name="imu_in_pelvis", pos=[0.04525, 0.0, -0.08339], size=[0.01, 0.0, 0.0])
    spec.add_sensor(
        name="imu_ang_vel",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu_in_pelvis",
        dim=3,
    )
    wrist = spec.body("right_wrist_yaw_link")
    xml = ET.parse(OFFICIAL_XML)
    headlight = xml.find("visual/headlight")
    for key in ("ambient", "diffuse", "specular"):
        setattr(spec.visual.headlight, key, np.fromstring(headlight.attrib[key], sep=" "))
    for light in xml.findall("worldbody/light"):
        spec.worldbody.add_light(
            pos=np.fromstring(light.attrib["pos"], sep=" "),
            dir=np.fromstring(light.attrib["dir"], sep=" "),
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
    site = xml.find(".//site[@name='racket_sweet_spot']")
    visual = xml.find(".//geom[@name='racket_visual']")
    mesh = xml.find(".//mesh[@name='racket_mesh']")
    spec.add_mesh(
        name="athlete_racket_mesh",
        file=str(OFFICIAL_XML.parent.parent / "meshes" / mesh.attrib["file"]),
        scale=np.fromstring(mesh.attrib["scale"], sep=" "),
    )
    wrist.add_site(
        name="racket_sweet_spot",
        pos=np.fromstring(site.attrib["pos"], sep=" "),
        quat=np.fromstring(site.attrib["quat"], sep=" "),
        size=[0.015, 0.0, 0.0],
    )
    wrist.add_geom(
        name="racket_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="athlete_racket_mesh",
        pos=np.fromstring(visual.attrib["pos"], sep=" "),
        quat=np.fromstring(visual.attrib["quat"], sep=" "),
        contype=0,
        conaffinity=0,
        density=0,
        group=2,
        rgba=[0.16, 0.16, 0.18, 1.0],
    )
    for i, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"holo_geom_{i}"
    spec.compile()
    path = output / "canonical_g1_with_racket_mount.xml"
    path.write_text(spec.to_xml())
    return path


def install(profile, output):
    """Only change factories in this evaluation process, leaving vendor intact."""
    robot = prepare_robot(output)
    config = motor_config()
    original_build = player._build_standalone_model

    def build(xml, dt, names, kp, kd, court):
        model = original_build(robot, dt, names, kp, kd, court)
        if profile == "canonical_unitree":
            aids = np.array(
                [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
            )
            jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
            dofs = model.jnt_dofadr[jids]
            model.actuator_gainprm[aids] = 0.0
            model.actuator_biasprm[aids] = 0.0
            model.dof_armature[dofs] = by_joint(config, "armature", names)
            model.dof_damping[dofs] = 0.0
            model.dof_frictionloss[dofs] = 0.0
        return model

    player._build_standalone_model = build
    parent = player._StandaloneStudentSimulation

    class CanonicalSimulation(parent):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.motor = {
                k: by_joint(config, k, self.joint_names)
                for k in ("stiffness", "damping", "X1", "X2", "Y1", "Y2", "Fs", "Fd", "Va")
            }

        def _apply_ball_aerodynamics(self):
            super()._apply_ball_aerodynamics()
            if profile != "canonical_unitree":
                return
            m = self.motor
            dq = self.data.qvel[self.joint_dof_adr]
            q = self.data.qpos[self.joint_qpos_adr]
            effort = m["stiffness"] * (self.data.ctrl[self.actuator_ids] - q) - m["damping"] * dq
            maximum = np.where(dq * effort > 0, m["Y1"], m["Y2"])
            limit = np.maximum(0.0, maximum * (m["X2"] - np.abs(dq)) / (m["X2"] - m["X1"]))
            limit = np.where(np.abs(dq) < m["X1"], maximum, limit)
            effort = np.clip(effort, -limit, limit) - m["Fs"] * np.tanh(dq / m["Va"]) - m["Fd"] * dq
            self.data.qfrc_applied[self.joint_dof_adr] = effort

    return CanonicalSimulation, {
        "profile": profile,
        "robot_source": str(URDF),
        "robot_source_sha256": sha(URDF),
        "derived_xml": str(robot),
        "derived_xml_sha256": sha(robot),
        "motor": "canonical nominal Unitree explicit PD, torque-speed curve, armature and friction; no delay/ERFI"
        if profile == "canonical_unitree"
        else "ATHLETE deployment position PD on canonical URDF",
        "backend": "MuJoCo",
        "IsaacLab_or_real_robot_equivalence_claimed": False,
        "racket": "official attachment pose, mesh, collision and inertial properties; original canonical hand retained",
    }
