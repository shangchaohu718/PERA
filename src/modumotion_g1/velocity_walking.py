"""Velocity-command teacher and inputs for a frozen-public walking expert.

The released ONNX embeds its observation normalizer. Load those exact tensors;
never replace them with an independently fitted normalizer. Physical coordinates
are converted by named joints, including the reversed Instinct waist chain.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pera.paths import ROOT

RELEASE = ROOT / "third_party/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0"
DOMAIN = "velocity_walk"
CONTEXT_SCHEMA = {
    "kind": "velocity_command_and_clock_v1",
    "fields": [
        "vx_body_m_s",
        "vy_body_m_s",
        "yaw_rate_rad_s",
        "phase_sin",
        "phase_cos",
    ],
    "period_s": 0.6,
    "phase_zero_when_command_norm_below": 0.1,
    "source": "requested_command_and_elapsed_control_steps_only",
    "uses_vision": False,
    "uses_teacher_hidden_states_or_future_actions": False,
}
CONTEXT_HASH = hashlib.sha256(json.dumps(CONTEXT_SCHEMA, sort_keys=True).encode()).hexdigest()


def context_contract():
    # Existing plugin files call this identity field encoder_sha256. Here it
    # fingerprints the stateless command/clock producer; there is no encoder.
    return {
        "context_dim": 5,
        "context_schema": CONTEXT_SCHEMA,
        "encoder_sha256": CONTEXT_HASH,
    }


def command_context(command, elapsed):
    phase = elapsed * (2 * torch.pi / 0.6)
    clock = torch.stack((phase.sin(), phase.cos()), -1)
    clock = torch.where(command.norm(dim=-1, keepdim=True) < 0.1, 0.0, clock)
    return torch.cat((command, clock), -1)


class ReleasedVelocityTeacher(nn.Module):
    """Exact Torch evaluation of the small released ONNX, with batch support."""

    def __init__(self, path=RELEASE / "exported/policy.onnx", device="cpu"):
        super().__init__()
        import onnx
        from onnx.numpy_helper import to_array

        self.path = Path(path)
        graph = onnx.load(str(path))
        self.metadata = {p.key: p.value for p in graph.metadata_props}
        tensors = {x.name: torch.from_numpy(to_array(x).copy()) for x in graph.graph.initializer}
        expected = ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
        if [n.op_type for n in graph.graph.node] != expected:
            raise ValueError("Released teacher graph changed; adapt it explicitly")
        nodes = graph.graph.node
        if nodes[0].input[0] != "obs" or nodes[-1].output[0] != "actions":
            raise ValueError("Unexpected teacher graph inputs/outputs")
        self.register_buffer("mean", tensors[nodes[0].input[1]])
        self.register_buffer("divisor", tensors[nodes[1].input[1]])
        self.layers = nn.ModuleList()
        previous = nodes[1].output[0]
        for index in (2, 4, 6, 8):
            n = nodes[index]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            if (
                attrs.get("transB", 0) != 1
                or attrs.get("alpha", 1) != 1
                or attrs.get("beta", 1) != 1
            ):
                raise ValueError("Unsupported Gemm attributes")
            if n.input[0] != previous:
                raise ValueError("Unexpected teacher graph connectivity")
            weight, bias = tensors[n.input[1]], tensors[n.input[2]]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
            self.layers.append(layer)
            if index < 8:
                elu = nodes[index + 1]
                if elu.input[0] != n.output[0]:
                    raise ValueError("Unexpected ELU connectivity")
                if any(
                    a.name == "alpha" and onnx.helper.get_attribute_value(a) != 1
                    for a in elu.attribute
                ):
                    raise ValueError("Unexpected ELU alpha")
                previous = elu.output[0]
        self.names = self.metadata["joint_names"].split(",")
        for key in (
            "default_joint_pos",
            "action_scale",
            "joint_stiffness",
            "joint_damping",
        ):
            self.register_buffer(
                key, torch.tensor([float(x) for x in self.metadata[key].split(",")])
            )
        self.to(device).requires_grad_(False).eval()

    def forward(self, observation):
        x = (observation - self.mean) / self.divisor
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x

    @torch.no_grad()
    def verify_onnx(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(197)
        observations = rng.normal(0, 0.4, (32, 98)).astype(np.float32)
        observations[:, 3:6] = [0, 0, -1]
        expected = np.concatenate([session.run(None, {"obs": x[None]})[0] for x in observations])
        actual = self(torch.as_tensor(observations, device=self.mean.device)).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        return {
            "passed": True,
            "samples": 32,
            "max_abs_error": float(abs(actual - expected).max()),
        }


def build_walk_env(num_envs, device, teacher, video=False):
    """Same native robot/actuators as scene experts, on a fixed flat plane.

    No terrain/reference tracking termination is appropriate to this velocity
    task. Retain the locomotion task's body-contact termination. Report this as
    a controlled-physics check, not domain-randomized robustness.
    """
    from instinct_mj.envs import InstinctRlEnv
    from instinct_mj.tasks.locomotion.config.g1.flat_env_cfg import (
        instinct_g1_locomotion_flat_env_cfg,
    )
    from instinct_mj.tasks.registry import load_env_cfg

    cfg = instinct_g1_locomotion_flat_env_cfg(play=True)
    scene_cfg = load_env_cfg("Instinct-Perceptive-Shadowing-G1-v0", play=True)
    cfg.scene.entities["robot"] = copy.deepcopy(scene_cfg.scene.entities["robot"])
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (0, 0, 0.82)
    robot.init_state.joint_pos = {
        name: float(value) * (-1 if name.startswith("waist_") else 1)
        for name, value in zip(teacher.names, teacher.default_joint_pos.cpu())
    }
    robot.init_state.joint_vel = {".*": 0.0}
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 12.0
    cfg.auto_reset = False
    cfg.seed = 197
    cfg.sim = copy.deepcopy(scene_cfg.sim)
    cfg.decimation = scene_cfg.decimation
    cfg.episode_length_s = 120
    cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints")}
    cfg.events["reset_base"].params["pose_range"] = {"yaw": (-0.15, 0.15)}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["reset_robot_joints"].params["position_range"] = (0.97, 1.03)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    cfg.observations["policy"].enable_corruption = False
    cfg.commands["base_velocity"].debug_vis = False
    cfg.viewer.width, cfg.viewer.height = 960, 720
    cfg.viewer.elevation, cfg.viewer.azimuth, cfg.viewer.distance = -20, 140, 3.0
    env = InstinctRlEnv(cfg, device=device, render_mode="rgb_array" if video else None)
    # The local Instinct plane importer has no terrain.debug_vis method. Use
    # mjlab's normal manager visualizers, without changing simulated geometry.
    if video:
        import types

        from mjlab.envs import ManagerBasedRlEnv

        env.update_visualizers = types.MethodType(ManagerBasedRlEnv.update_visualizers, env)
    return env


class VelocityWalkTask:
    def __init__(self, num_envs=8, device="cuda:0", video=False):
        from instinct_mj.rl.supermimic.kinematics import G1Kinematics

        from modumotion_g1.perceptive_observations import HoloMotionObservationAdapter

        self.N, self.device = num_envs, device
        self.teacher = ReleasedVelocityTeacher(device=device)
        self.env = build_walk_env(num_envs, device, self.teacher, video)
        self.robot = self.env.scene["robot"]
        self.action = self.env.action_manager.get_term("joint_pos")
        if list(self.action._target_names) != list(self.robot.joint_names):
            raise ValueError("Action order differs from robot joint order")
        self.adapter = HoloMotionObservationAdapter(
            self.robot.joint_names,
            G1Kinematics(device, self.robot.joint_names, ["pelvis"]),
        )
        self.teacher_indices = [self.robot.joint_names.index(n) for n in self.teacher.names]
        self.sign = torch.tensor(
            [-1 if n.startswith("waist_") else 1 for n in self.teacher.names],
            device=device,
        )
        self.native_to_teacher = [self.teacher.names.index(n) for n in self.robot.joint_names]
        self.native_sign = self.adapter.sim_joint_signs.to(device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.command = torch.zeros(num_envs, 3, device=device)
        self.last_action = torch.zeros(num_envs, 29, device=device)
        self.teacher_last_action = torch.zeros_like(self.last_action)

    def state(self):
        from instinct_mj.rl.supermimic.rotations import quat_to_mat

        idx, data = self.robot.indexing, self.env.sim.data
        q, v = data.qpos, data.qvel
        R = quat_to_mat(q[:, idx.free_joint_q_adr.long()[3:7]])
        fv = idx.free_joint_v_adr.long()
        return (
            q[:, idx.free_joint_q_adr.long()[:3]].clone(),
            R,
            v[:, fv[:3]].clone(),
            (R @ v[:, fv[3:6], None]).squeeze(-1),
            q[:, idx.joint_q_adr.long()].clone(),
            v[:, idx.joint_v_adr.long()].clone(),
        )

    def reset(self, seed=197, actor=None):
        self.env.reset(seed=seed)
        self.elapsed.zero_()
        self.last_action.zero_()
        self.teacher_last_action.zero_()
        if actor is not None:
            actor.reset_kv_cache(self.N, self.device)
            actor.clear_env_cache(None)

    def teacher_observation(self):
        from modumotion_g1.perceptive_observations import _gravity, _matquat

        state = self.state()
        _, r, _, w = self.adapter.pelvis_state(state)
        return torch.cat(
            (
                (r.transpose(-1, -2) @ w[..., None]).squeeze(-1),
                _gravity(_matquat(r)),
                command_context(self.command, self.elapsed),
                state[4][:, self.teacher_indices] * self.sign - self.teacher.default_joint_pos,
                state[5][:, self.teacher_indices] * self.sign,
                self.teacher_last_action,
            ),
            -1,
        )

    def teacher_targets(self):
        target = self.teacher.default_joint_pos + self.teacher.action_scale * self.teacher(
            self.teacher_observation()
        )
        return target[:, self.native_to_teacher] * self.native_sign

    def native_to_holo_target(self, target):
        a = self.adapter
        return (
            target[:, a.to_holo] * a.holo_joint_signs.to(target) - a.action_offset.to(target)
        ) / a.action_scale.to(target)

    def student_inputs(self):
        """604-D public contract plus explicit 5-D command/clock context.

        Reference joint posture is constant; no teacher rollout, action label,
        future physical state or handcrafted cyclic joint motion is an input.
        """
        from tensordict import TensorDict

        from modumotion_g1.perceptive_observations import _matquat, _yaw

        state = self.state()
        p, r, _, _ = self.adapter.pelvis_state(state)
        yaw = _yaw(_matquat(r))
        t = (torch.arange(13, device=self.device) - 1) * 0.02
        angle = yaw[:, None] + self.command[:, 2:3] * t
        half = angle * 0.5
        quat = torch.stack(
            (half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()), -1
        )
        vx, vy = self.command[:, 0], self.command[:, 1]
        vel = torch.stack((yaw.cos() * vx - yaw.sin() * vy, yaw.sin() * vx + yaw.cos() * vy), -1)
        xy = p[:, None, :2] + vel[:, None] * t[None, :, None]
        # Fixed released standing pelvis height; command is relative and has no
        # absolute-world position term in Holo's released observation contract.
        z = torch.full((self.N, 13, 1), 0.76, device=self.device)
        q = self.teacher.default_joint_pos[
            [self.teacher.names.index(n) for n in self.adapter.names]
        ]
        reference = torch.cat((xy, z, quat, q[None, None].expand(self.N, 13, -1)), -1)
        terms = self.adapter.terms(state, self.last_action, reference, reference_sample_time=t)
        return TensorDict(
            {f"unified/{k}": v for k, v in terms.items()}, batch_size=[self.N]
        ), command_context(self.command, self.elapsed)

    def step_target(self, native_target):
        native_action = (native_target - self.action.offset) / self.action.scale
        if not torch.isfinite(native_action).all():
            raise FloatingPointError("Nonfinite walk action")
        _, _, terminated, truncated, _ = self.env.step(native_action)
        self.last_action.copy_(self.native_to_holo_target(native_target))
        canonical = native_target[:, self.teacher_indices] * self.sign
        self.teacher_last_action.copy_(
            (canonical - self.teacher.default_joint_pos) / self.teacher.action_scale
        )
        self.elapsed += self.env.step_dt
        return terminated | truncated

    def motion_metrics(self):
        from modumotion_g1.perceptive_observations import _matquat, _yaw

        p, r, v, w = self.adapter.pelvis_state(self.state())
        yaw = _yaw(_matquat(r))
        local = torch.stack(
            (
                yaw.cos() * v[:, 0] + yaw.sin() * v[:, 1],
                -yaw.sin() * v[:, 0] + yaw.cos() * v[:, 1],
                w[:, 2],
            ),
            -1,
        )
        tilt = torch.acos(r[:, 2, 2].clamp(-1, 1))
        return p, local, tilt


# Each command segment has enough steady time to separately measure tracking.
EVAL_SEGMENTS = [
    ("stand", 2, (0, 0, 0)),
    ("forward", 5, (0.6, 0, 0)),
    ("stop", 3, (0, 0, 0)),
    ("turn_left", 4, (0.35, 0, 0.5)),
    ("turn_right", 4, (0.35, 0, -0.5)),
    ("side_left", 3, (0, 0.25, 0)),
    ("side_right", 3, (0, -0.25, 0)),
    ("backward", 3, (-0.25, 0, 0)),
    ("final_stop", 3, (0, 0, 0)),
]
