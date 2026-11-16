"""HUSKY teacher and dynamic-board DAgger bridge for the frozen G1 Actor.

Observation and control conventions derive from TeleHuman/humanoid_skateboarding
test_scene/sim.py (CC BY-NC 4.0). Upstream assets/checkpoints stay in that repo.
The optional 29-joint scene adds six physical wrist hinges with zero teacher
targets; the original 23 robot actuators and all board mechanics are retained.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pera.paths import ROOT

HUSKY = ROOT / "third_party/HUSKY"
DOMAIN = "husky_skateboard"
DEFAULT23 = np.array(
    [
        0,
        0,
        0,
        0.23,
        -0.20,
        0,
        -0.7,
        0,
        0,
        1.17,
        -0.45,
        0,
        0,
        0,
        0,
        -0.03,
        0.45,
        -0.21,
        1.32,
        -0.7,
        -0.845,
        0.83,
        1.19,
    ],
    dtype=np.float32,
)
SCALE23 = np.array(
    [
        0.5475,
        0.3507,
        0.5475,
        0.3507,
        0.4386,
        0.4386,
        0.5475,
        0.3507,
        0.5475,
        0.3507,
        0.4386,
        0.4386,
        0.5475,
        0.4386,
        0.4386,
        *([0.4386] * 8),
    ],
    dtype=np.float32,
)
OBS_DIMS = (2, 1, 3, 3, 23, 23, 23, 1)
OBS_STARTS = np.cumsum([0, *OBS_DIMS[:-1]])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReleasedHuskyTeacher(nn.Module):
    """Exact released ONNX MLP, including its original normalization."""

    def __init__(self, device="cpu"):
        super().__init__()
        import onnx
        from onnx.numpy_helper import to_array

        self.path = HUSKY / "ckpts/test.onnx"
        model = onnx.load(str(self.path))
        nodes = model.graph.node
        if [n.op_type for n in nodes] != [
            "Sub",
            "Div",
            "Gemm",
            "Elu",
            "Gemm",
            "Elu",
            "Gemm",
            "Elu",
            "Gemm",
        ]:
            raise ValueError("Released HUSKY graph changed")
        tensors = {x.name: torch.from_numpy(to_array(x).copy()) for x in model.graph.initializer}
        self.register_buffer("mean", tensors[nodes[0].input[1]])
        self.register_buffer("divisor", tensors[nodes[1].input[1]])
        self.layers = nn.ModuleList()
        for i in (2, 4, 6, 8):
            node = nodes[i]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get("transB") != 1 or attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1:
                raise ValueError("Unexpected HUSKY Gemm attributes")
            weight, bias = (tensors[name] for name in node.input[1:])
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
            self.layers.append(layer)
        self.to(device).requires_grad_(False).eval()

    def encode(self, observation):
        x = (observation - self.mean) / self.divisor
        for layer in self.layers[:-1]:
            x = F.elu(layer(x))
        return x

    def forward(self, observation):
        return self.layers[-1](self.encode(observation))

    @torch.no_grad()
    def verify_onnx(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        generator = torch.Generator().manual_seed(241)
        obs = self.mean.cpu() + self.divisor.cpu() * torch.randn(32, 395, generator=generator) * 0.3
        expected = np.concatenate(
            [session.run(None, {session.get_inputs()[0].name: x[None].numpy()})[0] for x in obs]
        )
        actual = self(obs.to(self.mean.device)).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        return {
            "passed": True,
            "samples": 32,
            "max_abs_error": float(np.max(abs(actual - expected))),
        }


def context_contract():
    return {
        "context_dim": 128,
        "context_schema": {
            "kind": "released_husky_frozen_proprioceptive_encoder_v1",
            "input": "395D block-major five-frame actual proprioception/executed-action history, requested speed/absolute heading, 6s clock",
            "output": "third released actor hidden layer after ELU",
            "uses_vision": False,
            "uses_teacher_output_or_future_state": False,
            "reference": "constant HUSKY initial joint posture plus requested planar velocity; no teacher future trajectory",
        },
        "encoder_sha256": digest(HUSKY / "ckpts/test.onnx"),
    }


def scene_model(wrists=True):
    """Compile without editing the original scene or any of its assets."""
    path = HUSKY / "test_scene/mjlab_scene.xml"
    if not wrists:
        model = mujoco.MjModel.from_xml_path(str(path))
    else:
        xml = ET.parse(path).getroot()
        compiler = xml.find("compiler")
        compiler.set("meshdir", str((path.parent / compiler.get("meshdir")).resolve()))
        xml.remove(xml.find("keyframe"))
        actuator = xml.find("actuator")
        original_g1 = ET.parse(ROOT / "assets/g1/g1_29dof.xml").getroot()
        for side in ("left", "right"):
            for axis in ("roll", "pitch", "yaw"):
                name = f"{side}_wrist_{axis}_joint"
                source = original_g1.find(f".//joint[@name='{name}']")
                body = xml.find(f".//body[@name='robot/{side}_wrist_{axis}_link']")
                ET.SubElement(
                    body,
                    "joint",
                    name="robot/" + name,
                    pos="0 0 0",
                    axis=source.get("axis"),
                    range=source.get("range"),
                    armature="0.004",
                )
                ET.SubElement(
                    actuator,
                    "general",
                    name="robot/" + name,
                    joint="robot/" + name,
                    ctrllimited="false",
                    forcelimited="true",
                    forcerange="-5 5",
                    biastype="affine",
                    gainprm="4",
                    biasprm="0 -4 -0.3",
                )
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.opt.timestep = 0.005
    model.opt.iterations, model.opt.ls_iterations, model.opt.ccd_iterations = 10, 20, 50
    return model


class HuskySkateTask:
    """CPU MuJoCo physics, batched frozen encoder/Actor on the chosen device."""

    def __init__(self, num_envs=8, device="cuda:0", wrists=True):
        from modumotion_g1.depth_features import tensor_module_digest

        self.N, self.device = num_envs, device
        self.wrists = wrists
        self.model = scene_model(wrists)
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.teacher = ReleasedHuskyTeacher(device)
        self.teacher_digest = tensor_module_digest(self.teacher)
        contract = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())
        self.names = contract["joint_names"]
        self.default_holo = torch.tensor(contract["default_joint_pos"], device=device)
        self.offset = torch.tensor(contract["action_offset"], device=device)
        self.scale = torch.tensor(contract["action_scale"], device=device)
        original = scene_model(False)
        self.teacher_names = [
            original.joint(i).name.removeprefix("robot/")
            for i in range(original.njnt)
            if original.joint(i).name.startswith("robot/")
            and original.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(self.teacher_names) != 23 or len(set(self.names) - set(self.teacher_names)) != 6:
            raise ValueError("Unexpected HUSKY joint correspondence")
        self.teacher_holo_indices = [self.names.index(n) for n in self.teacher_names]
        self.qadr, self.vadr, self.actuator_ids = [], [], []
        for name in self.names:
            try:
                joint = self.model.joint("robot/" + name)
                self.qadr.append(int(joint.qposadr[0]))
                self.vadr.append(int(joint.dofadr[0]))
                self.actuator_ids.append(self.model.actuator("robot/" + name).id)
            except KeyError:
                self.qadr.append(-1)
                self.vadr.append(-1)
                self.actuator_ids.append(-1)
        self.present = np.flatnonzero(np.array(self.qadr) >= 0)
        self.init_joint = np.zeros(29, np.float32)
        self.init_joint[self.teacher_holo_indices] = DEFAULT23
        self.init_tensor = torch.tensor(self.init_joint, device=device)
        self.history = np.zeros((num_envs, 5, 79), np.float32)
        self.executed_teacher_action = np.zeros((num_envs, 23), np.float32)
        self.last_action = torch.zeros(num_envs, 29, device=device)
        self.steps = np.zeros(num_envs, np.int64)
        self.command = np.tile([1.0, 0.0], (num_envs, 1)).astype(np.float32)
        self.sensor_address = int(self.model.sensor_adr[self.model.sensor("robot/imu_ang_vel").id])
        self.pelvis = self.model.body("robot/pelvis").id
        self.board = self.model.body("skateboard/skateboard_deck").id
        self._observed = None

    def reset(self, seed, actor=None, ids=None, perturb=False):
        ids = list(range(self.N)) if ids is None else list(ids)
        rng = np.random.default_rng(seed)
        for i in ids:
            data = self.data[i]
            mujoco.mj_resetData(self.model, data)
            data.qpos[self.qadr_array] = self.init_joint[self.present]
            if perturb:
                data.qpos[0:2] += rng.uniform(-0.005, 0.005, 2)
                data.qpos[self.qadr_array] += rng.uniform(-0.003, 0.003, len(self.present))
            data.ctrl[self.actuator_array] = self.init_joint[self.present]
            mujoco.mj_forward(self.model, data)
        self.history[ids] = 0
        self.executed_teacher_action[ids] = 0
        self.last_action[ids] = 0
        self.steps[ids] = 0
        self._observed = None
        if actor is not None:
            if len(ids) == self.N:
                actor.reset_kv_cache(self.N, self.device)
                actor.clear_env_cache(None)
            else:
                actor.clear_env_cache(torch.tensor(ids, device=self.device))

    @property
    def qadr_array(self):
        return np.array(self.qadr)[self.present]

    @property
    def actuator_array(self):
        return np.array(self.actuator_ids)[self.present]

    def joints(self):
        q = np.zeros((self.N, 29), np.float32)
        qd = np.zeros_like(q)
        for i, data in enumerate(self.data):
            q[i, self.present] = data.qpos[self.qadr_array]
            qd[i, self.present] = data.qvel[np.array(self.vadr)[self.present]]
        return q, qd

    def observe(self):
        from modumotion_g1.perceptive_observations import _gravity, _yaw

        if self._observed is None:
            q, qd = self.joints()
            quat = torch.tensor(
                np.stack([d.qpos[3:7] for d in self.data]),
                dtype=torch.float32,
                device=self.device,
            )
            yaw = _yaw(quat)
            gyro = torch.tensor(
                np.stack(
                    [d.sensordata[self.sensor_address : self.sensor_address + 3] for d in self.data]
                ),
                dtype=torch.float32,
                device=self.device,
            )
            phase = (((self.steps + 1) * 0.02 / 6) % 1).astype(np.float32)
            frame = np.concatenate(
                (
                    self.command * [2, 1],
                    yaw.cpu().numpy()[:, None] / np.pi,
                    gyro.cpu().numpy() * 0.25,
                    _gravity(quat).cpu().numpy(),
                    q[:, self.teacher_holo_indices] - DEFAULT23,
                    qd[:, self.teacher_holo_indices] * 0.05,
                    self.executed_teacher_action,
                    phase[:, None],
                ),
                -1,
            ).astype(np.float32)
            self.history[:, :-1] = self.history[:, 1:].copy()
            self.history[:, -1] = frame
            observation = np.concatenate(
                [
                    self.history[:, :, start : start + width].reshape(self.N, -1)
                    for start, width in zip(OBS_STARTS, OBS_DIMS)
                ],
                -1,
            )
            encoded = self.teacher.encode(torch.tensor(observation, device=self.device)).detach()
            self._observed = q, qd, quat, yaw, gyro, encoded
        return self._observed

    @torch.no_grad()
    def teacher_targets(self):
        context = self.observe()[-1]
        action = self.teacher.layers[-1](context)
        target = torch.zeros(self.N, 29, device=self.device)
        target[:, self.teacher_holo_indices] = torch.tensor(
            DEFAULT23, device=self.device
        ) + action * torch.tensor(SCALE23, device=self.device)
        return target

    def target_to_holo(self, target):
        return (target - self.offset) / self.scale

    def holo_to_target(self, action):
        return self.offset + self.scale * action

    def student_inputs(self):
        from tensordict import TensorDict

        from modumotion_g1.perceptive_observations import (
            PerceptiveActorObservationInput,
            build_perceptive_actor_terms,
        )

        q, qd, quat, _yaw, gyro, context = self.observe()
        times = (torch.arange(13, device=self.device) - 1) * 0.02
        command = torch.tensor(self.command, device=self.device)
        angle = command[:, 1:2].expand(-1, 13)
        half = 0.5 * angle
        refquat = torch.stack(
            (half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()), -1
        )
        velocity = command[:, :1] * torch.stack((command[:, 1].cos(), command[:, 1].sin()), -1)
        position = torch.tensor(
            np.stack([d.qpos[:3] for d in self.data]),
            dtype=torch.float32,
            device=self.device,
        )
        xy = position[:, None, :2] + velocity[:, None] * times[None, :, None]
        reference = torch.cat(
            (
                xy,
                torch.full((self.N, 13, 1), 0.78, device=self.device),
                refquat,
                self.init_tensor[None, None].expand(self.N, 13, -1),
            ),
            -1,
        )
        terms = build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=reference,
                robot_root_quat_wxyz=quat,
                robot_root_angvel_local=gyro,
                robot_dof_pos=torch.tensor(q, device=self.device),
                robot_dof_vel=torch.tensor(qd, device=self.device),
                last_action=self.last_action,
                default_dof_pos=self.default_holo,
                reference_sample_time=times,
            )
        )
        return TensorDict(
            {f"unified/{k}": v for k, v in terms.items()}, batch_size=[self.N]
        ), context

    def step_target(self, target, active=None):
        target_np = target.detach().cpu().numpy()
        active = np.ones(self.N, bool) if active is None else np.asarray(active, bool)
        for i, data in enumerate(self.data):
            if not active[i]:
                continue
            data.ctrl[self.actuator_array] = target_np[i, self.present]
            # All seven board controls remain their original zero targets.
            for _ in range(4):
                mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
        self.last_action[active] = self.target_to_holo(target)[active]
        self.executed_teacher_action[active] = (
            target_np[active][:, self.teacher_holo_indices] - DEFAULT23
        ) / SCALE23
        self.steps[active] += 1
        self._observed = None
        m = self.metrics()
        return (
            (m["pelvis_height"] < 0.45)
            | (m["tilt"] > 1.0)
            | (m["board_distance"] > 0.8)
            | ~m["finite"]
        )

    def metrics(self):
        heights = np.array([d.xpos[self.pelvis, 2] for d in self.data])
        tilt = np.arccos(
            np.clip([d.xmat[self.pelvis].reshape(3, 3)[2, 2] for d in self.data], -1, 1)
        )
        distance = np.array(
            [np.linalg.norm(d.xpos[self.pelvis, :2] - d.xpos[self.board, :2]) for d in self.data]
        )
        return {
            "pelvis_height": heights,
            "tilt": tilt,
            "board_distance": distance,
            "board_xy": np.stack([d.xpos[self.board, :2].copy() for d in self.data]),
            "finite": np.array(
                [np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() for d in self.data]
            ),
        }

    def contact_flags(self):
        result = []
        for data in self.data:
            board_foot = wheel_ground = False
            for contact in data.contact[: data.ncon]:
                body = [self.model.body(int(self.model.geom_bodyid[g])).name for g in contact.geom]
                board_foot |= any("robot/" in b and "ankle_roll" in b for b in body) and any(
                    "skateboard/" in b and "wheel" not in b for b in body
                )
                static_ground = any(
                    self.model.body_weldid[int(self.model.geom_bodyid[g])] == 0
                    for g in contact.geom
                )
                wheel_ground |= (
                    any("skateboard/" in b and "wheel" in b for b in body) and static_ground
                )
            result.append((board_foot, wheel_ground))
        return np.array(result, dtype=bool)
