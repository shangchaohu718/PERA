"""Physical task regions, with one shared student and explicit expert dispatch.

Native auxiliary MjData objects are read-only FK/sensor mirrors. Only the arena
MjData advances physics. Native teacher action networks are never queried.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from torch import nn

from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import quat_to_mat
from pera.multi_robot import MultiRobotController, NonfinitePolicyAction
from pera.paths import ROOT
from pera.robot import get_robot
from pera.showcase import SharedWorld, path, check_resources
from pera.switching import file_digest


class DepthEncoder(nn.Module):
    """Exact released two-convolution/MLP depth32 computation and normalization."""

    def __init__(self, folder, device):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv = nn.Module()
        self.encoder.conv.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.encoder.head = nn.Module()
        self.encoder.head.model = nn.Sequential(
            nn.Linear(18432, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.register_buffer("mean", torch.zeros(1, 576))
        self.register_buffer("std", torch.ones(1, 576))
        self.load_state_dict(
            torch.load(folder / "depth_state.pt", map_location="cpu", weights_only=True)
        )
        self.eps = json.loads((folder / "contract.json").read_text())["depth_eps"]
        self.to(device).eval().requires_grad_(False)

    def forward(self, depth):
        x = ((depth.reshape(1, -1) - self.mean) / (self.std + self.eps)).reshape(1, 1, 18, 32)
        return self.encoder.head.model(self.encoder.conv.conv(x).flatten(1))


class VaultCamera:
    def __init__(self, folder, device):
        from instinct_mj.utils.noise import noise_cfg

        self.device = device
        self.encoder = DepthEncoder(folder, device)
        self.geometry = dict(np.load(folder / "camera_geometry.npz"))
        self.offset_R = quat_to_mat(torch.tensor(self.geometry["offset_quat"])).numpy()
        self.pipeline = [
            getattr(noise_cfg, row["cls"])(**{**row["kwargs"], "device": device})
            for row in json.loads((folder / "noise.json").read_text())
        ]
        self.last_image = None

    @torch.no_grad()
    def observe(self, model, data, torso):
        g = self.geometry
        body_R = data.xmat[torso].reshape(3, 3)
        R = body_R @ self.offset_R
        point = data.xpos[torso] + body_R @ g["offset_pos"]
        assert np.max(abs(g["starts"] - g["starts"][0])) < 1e-6
        point = point + R @ g["starts"][0]
        directions = np.ascontiguousarray(g["directions"] @ R.T, dtype=np.float64)
        ids = np.empty(len(directions), np.int32)
        distances = np.empty(len(directions), np.float64)
        # The original camera sees geom groups0,2, does not exclude its body,
        # and has no mesh-path restriction in this released task.
        ray_args = [
            model,
            data,
            point,
            directions.ravel(),
            np.array([1, 0, 1, 0, 0, 0], np.uint8),
            True,
            -1,
            ids,
            distances,
        ]
        ray_args.append(None)
        mujoco.mj_multiRay(*ray_args, len(directions), 1e6)
        # Native rays continue past near self-geometry instead of whitening it.
        group = np.array([1, 0, 1, 0, 0, 0], np.uint8)
        near = getattr(self, "min_distance", 0.05)
        for i in np.flatnonzero((distances >= 0) & (distances < near)):
            distance = mujoco.mj_ray(
                model,
                data,
                point + directions[i] * (near + 0.0001),
                directions[i],
                group,
                True,
                -1,
                ids[i : i + 1],
            )
            distances[i] = distance + near + 0.0001 if distance >= 0 else -1
        depth = np.where(distances >= near, distances * g["directions"][:, 0], 1e6)
        height, width = getattr(self, "raw_shape", (27, 48))
        x = torch.tensor(depth, device=self.device, dtype=torch.float32).reshape(
            1, height, width, 1
        )
        for cfg in self.pipeline:
            x = cfg.func(x, cfg, torch.tensor([0], device=self.device))
        self.last_image = x[0, :, :, 0].cpu().numpy()
        return self.encode(x)

    def encode(self, x):
        return self.encoder(x).flatten()


class RoughCamera(VaultCamera):
    """Released parkour encoder with the original eight delayed depth frames."""

    def __init__(self, folder, device):
        from instinct_mj.utils.noise import noise_cfg

        self.device, self.min_distance, self.raw_shape = device, 0.1, (36, 64)
        self.encoder = torch.jit.load(str(folder / "depth_encoder.ts"), map_location=device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.geometry = dict(np.load(folder / "camera_geometry.npz"))
        self.offset_R = quat_to_mat(torch.tensor(self.geometry["offset_quat"])).numpy()
        self.pipeline = [
            getattr(noise_cfg, row["cls"])(**{**row["kwargs"], "device": device})
            for row in json.loads((folder / "noise.json").read_text())
        ]
        self.last_image = None
        self.history = None

    def reset(self):
        self.history = None

    def encode(self, x):
        current = x[0, :, :, 0]
        if self.history is None:
            self.history = current[None].repeat(37, 1, 1)
        else:
            self.history = torch.cat((self.history[1:], current[None]))
        # The native uniform reset delay is fractional in [0,1]; indexing to
        # long selects the preceding frame, except for its zero endpoint.
        ids = torch.arange(0, 36, 5, device=x.device)
        depth = self.history[ids][None]
        return self.encoder(depth).flatten()


class Mirror:
    """Named address maps between one native observation mirror and shared physics."""

    def __init__(self, slot, model):
        native = slot.model
        self.qmap, self.vmap = [], []
        self.free = []
        for j in range(native.njnt):
            target = model.joint(slot.name + "/" + native.joint(j).name).id
            kind = native.jnt_type[j]
            nq, nv = (
                (7, 6)
                if kind == mujoco.mjtJoint.mjJNT_FREE
                else ((4, 3) if kind == mujoco.mjtJoint.mjJNT_BALL else (1, 1))
            )
            a, b = int(native.jnt_qposadr[j]), int(model.jnt_qposadr[target])
            self.qmap.extend(range(b, b + nq))
            self.vmap.extend(
                range(int(model.jnt_dofadr[target]), int(model.jnt_dofadr[target]) + nv)
            )
            if nq == 7:
                self.free.append(a)
        self.qmap, self.vmap = np.array(self.qmap), np.array(self.vmap)
        if len(self.qmap) != native.nq or len(self.vmap) != native.nv:
            raise ValueError("Incomplete native joint map")
        self.umap = np.array(
            [model.actuator(slot.name + "/" + native.actuator(i).name).id for i in range(native.nu)]
        )
        self.bmap = np.array(
            [0]
            + [model.body(slot.name + "/" + native.body(i).name).id for i in range(1, native.nbody)]
        )
        self.geomap = np.array(
            [
                model.geom(slot.name + "/" + (native.geom(i).name or f"_arena_geom_{i}")).id
                for i in range(native.ngeom)
            ]
        )
        self.slot = slot

    def push_reset(self, arena):
        d = self.slot.data
        q = d.qpos.copy()
        for a in self.free:
            q[a : a + 3] += self.slot.origin
        arena.data.qpos[self.qmap] = q
        arena.data.qvel[self.vmap] = d.qvel
        arena.data.ctrl[self.umap] = d.ctrl
        arena.data.qacc_warmstart[self.vmap] = 0

    def pull(self, arena, *, forward=True):
        d = self.slot.data
        d.qpos[:] = arena.data.qpos[self.qmap]
        for a in self.free:
            d.qpos[a : a + 3] -= self.slot.origin
        d.qvel[:] = arena.data.qvel[self.vmap]
        d.ctrl[:] = arena.data.ctrl[self.umap]
        d.time = arena.data.time
        if forward:
            mujoco.mj_forward(self.slot.model, d)

    def push_ball(self, arena):
        sim = self.slot.native
        a, b = sim.ball_qpos_adr, sim.ball_dof_adr
        value = sim.data.qpos[a : a + 7].copy()
        value[:3] += self.slot.origin
        arena.data.qpos[self.qmap[a : a + 7]] = value
        arena.data.qvel[self.vmap[b : b + 6]] = sim.data.qvel[b : b + 6]
        arena.data.xfrc_applied[self.bmap[sim.ball_body_id]] = sim.data.xfrc_applied[
            sim.ball_body_id
        ]


class Slot:
    def __init__(self, row, actor, config):
        self.row, self.name, self.origin = row, row["id"], np.array(row["origin"], float)
        self.actor, self.device = actor, next(actor.parameters()).device
        self.adapter = actor.reference_actor()
        self.official_holomotion = getattr(actor, "official_holomotion", False)
        self.sdk_names = get_robot("g1").joint_names
        self.signs = np.ones(29)
        self.last_action = torch.zeros(1, 29, device=self.device)
        self.episode_step, self.episode = 0, 1
        self.records, self.completed, self.resets = [], [], []
        self.lost = False
        self.faulted = False
        self.reference = None
        folder = path(row["resources"]) if row.get("resources") else None
        if row["kind"] in ("descent", "motion"):
            cfg = json.loads(
                path(
                    config["motion_config"] if row["kind"] == "motion" else config["descent_config"]
                ).read_text()
            )
            task = copy.deepcopy(next(t for t in cfg["tasks"] if t["id"] == row["id"]))
            task["origin"] = [0.0, 0.0, 0.0]
            native = SharedWorld(cfg, [task], actor)
            self.model, self.data, self.spec = (
                native.model,
                native.data,
                mujoco.MjSpec.from_string(native.xml),
            )
            self.native = native
            self.reference, self.mdp = native.motion, native.mdp
            with np.load(path(task["motion"])) as ref:
                self.official_reference = (
                    torch.tensor(
                        np.concatenate(
                            (
                                ref["body_pos_w"][:, 0],
                                ref["body_quat_w"][:, 0],
                                ref["joint_pos"][:, self.adapter.holo_from_sdk.cpu().numpy()],
                            ),
                            -1,
                        ),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if self.official_holomotion
                    else None
                )
            self.horizon = task["horizon"]
            if row["expert"] == "mimicx_kungfu":
                normalizer_path = path(task["motion"]).parent / "frozen_context_normalizer.pt"
                expected = actor.policy["experts"][row["expert"]]["context"]["encoder_sha256"]
                if file_digest(normalizer_path) != expected:
                    raise ValueError("Martial-arts normalizer does not match its expert")
                normalizer = torch.load(
                    normalizer_path, map_location=self.device, weights_only=True
                )
                self.context_mean = normalizer["state_dict"]["_mean"]
                self.context_std = normalizer["state_dict"]["_std"]
                self.context_eps = normalizer["eps"]
                with np.load(path(task["motion"])) as raw:
                    self.context_reference = {
                        k: torch.tensor(raw[k], device=self.device)
                        for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
                    }
            self.joint_names = [task["id"] + "/robot/" + n for n in self.sdk_names]
            self.pelvis_name = task["id"] + "/robot/pelvis"
        elif row["kind"] == "box":
            from pera.box_showcase import BoxRuntime

            self.model = mujoco.MjModel.from_xml_path(str(folder / "scene/scene.xml"))
            self.spec = mujoco.MjSpec.from_file(str(folder / "scene/scene.xml"))
            self.data = mujoco.MjData(self.model)
            self.initial = dict(np.load(folder / "initial_state.npz"))
            self.data.qpos[:] = self.initial["qpos"]
            self.data.qvel[:] = self.initial["qvel"]
            mujoco.mj_forward(self.model, self.data)
            self.native = BoxRuntime(self, folder)
            self.joint_names = list(self.sdk_names)
            self.pelvis_name = "pelvis"
            self.horizon = 600
        elif row["kind"] == "skate":
            import modumotion_g1.husky_skateboarding as hs

            def scene(wrists=True):
                return mujoco.MjModel.from_xml_path(
                    str(folder / ("g1_29" if wrists else "g1_23") / "scene.xml")
                )

            with patch.object(hs, "HUSKY", folder), patch.object(hs, "scene_model", scene):
                native = hs.HuskySkateTask(1, str(self.device))
            if self.official_holomotion:
                expected = actor.policy["experts"][row["expert"]]["context"]["encoder_sha256"]
                if file_digest(folder / "ckpts/test.onnx") != expected:
                    raise ValueError("Skate encoder does not match official expert")
            native.reset(7)
            self.board_start = native.metrics()["board_xy"][0].copy()
            self.native = native
            self.model, self.data = native.model, native.data[0]
            self.spec = mujoco.MjSpec.from_file(str(folder / "g1_29/scene.xml"))
            self.joint_names = ["robot/" + n for n in self.sdk_names]
            self.pelvis_name = "robot/pelvis"
            self.horizon = 600
        elif row["kind"] in ("vault", "rough"):
            scene_name = "scene_compact.xml" if row["kind"] == "rough" else "scene.xml"
            self.model = mujoco.MjModel.from_xml_path(str(folder / "scene" / scene_name))
            self.data = mujoco.MjData(self.model)
            self.spec = mujoco.MjSpec.from_file(str(folder / "scene" / scene_name))
            self.initial = dict(np.load(folder / "initial_state.npz"))
            if self.official_holomotion and row["kind"] == "vault":
                ref = np.load(folder / "actor_reference.npz")
                if float(ref["fps"]) != 50:
                    raise ValueError("Explicitly resample the native reference before playback")
                self.official_reference = torch.tensor(ref["qpos_holo"], device=self.device)
            self.contract = json.loads((folder / "contract.json").read_text())
            self.joint_names = ["robot/" + n for n in self.sdk_names]
            signs = dict(zip(self.contract["joint_names"], self.contract["sim_joint_signs"]))
            self.signs = np.array([signs[n] for n in self.sdk_names])
            self.pelvis_name = "robot/pelvis"
            kin = Kinematics(get_robot("g1"), self.device)

            def load(filename):
                raw = np.load(filename)
                q = raw["qpos_holo"]
                sdk = self.adapter.sdk_from_holo.cpu().numpy()
                return q[:, :3], q[:, 3:7], q[:, 7:][:, sdk], float(raw["fps"])

            if row["kind"] == "vault":
                self.reference = MotionLib(
                    [str(folder / "actor_reference.npz")], kin, str(self.device), load
                )
                self.mdp = MDP(
                    kin,
                    self.reference,
                    actor.pi.ref_q0,
                    get_robot("g1").xi_dim,
                    get_robot("g1").sensing,
                )
            self.camera = (RoughCamera if row["kind"] == "rough" else VaultCamera)(
                folder, str(self.device)
            )
            if self.official_holomotion:
                from modumotion_g1.depth_features import tensor_module_digest

                expected = actor.policy["experts"][row["expert"]]["context"]["encoder_sha256"]
                if tensor_module_digest(self.camera.encoder) != expected:
                    raise ValueError("Vault depth encoder does not match official expert")
            self.horizon = int(self.reference.clip_end[0]) if self.reference is not None else 600
            self.native = None
        elif row["kind"] == "tennis":
            from experiments.athlete_tennis_v1 import evaluate_official as official

            official.POLICY = folder / "model_45000.onnx"
            from athlete.scripts.play_student_onnx import _StandaloneStudentSimulation
            from experiments.athlete_tennis_v1.distill import Bridge, BallAudit

            sim = _StandaloneStudentSimulation(official.cfg(97001))
            self.native = sim
            self.bridge = Bridge(list(sim.joint_names), self.device)
            self.audit = BallAudit(sim)
            self.model, self.data = sim.model, sim.data
            self.spec = mujoco.MjSpec.from_file(str(folder / "scene/scene.xml"))
            norm = json.loads((folder / "context_normalization.json").read_text())
            if self.official_holomotion:
                expected = actor.policy["experts"][row["expert"]]["context"]["encoder_sha256"]
                if file_digest(folder / "context_normalization.json") != expected:
                    raise ValueError("Tennis normalization does not match official expert")
            self.mean = torch.tensor(norm["mean"], device=self.device)
            self.std = torch.tensor(norm["std"], device=self.device)
            self.joint_names = list(self.sdk_names)
            self.pelvis_name = "pelvis"
            self.horizon = 600
        else:
            raise ValueError("Unknown arena task")
        self.pelvis = self.model.body(self.pelvis_name).id
        joints = [self.model.joint(n).id for n in self.joint_names]
        self.qids = self.model.jnt_qposadr[joints]
        self.vids = self.model.jnt_dofadr[joints]
        # Native action/actuator orders are resolved by joint transmission.
        self.uids = np.array(
            [int(np.flatnonzero(self.model.actuator_trnid[:, 0] == j)[0]) for j in joints]
        )
        self.initial_previous = self.adapter.old_to_sdk(torch.zeros(1, 29, device=self.device))
        self.reset_native(initial=True)

    def reset_native(self, initial=False):
        if self.row["kind"] in ("descent", "motion"):
            self.native.reset()
        elif self.row["kind"] == "box":
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.initial["qpos"]
            self.data.qvel[:] = self.initial["qvel"]
            self.data.ctrl[:] = self.initial["ctrl"]
            mujoco.mj_forward(self.model, self.data)
            self.native.reset()
        elif self.row["kind"] == "skate":
            self.native.reset(7)
        elif self.row["kind"] in ("vault", "rough"):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.initial["qpos"]
            self.data.qvel[:] = self.initial["qvel"]
            mujoco.mj_forward(self.model, self.data)
            if self.row["kind"] == "rough":
                self.camera.reset()
        elif not initial:
            self.native.reset()
        if self.row["kind"] == "tennis":
            sim = self.native
            sim._set_ball_state(sim.incoming_ball_plan.launch_position_w)
            sim.ball_released = False
            self.cycle_start = float(self.data.time)
        self.last_action.copy_(self.initial_previous)
        self.episode_step, self.lost = 0, False
        self.goal_dwell_steps, self.goal_ready = 0, False

    def state(self):
        d = self.data
        R = torch.tensor(
            d.xmat[self.pelvis].reshape(3, 3), device=self.device, dtype=torch.float32
        )[None]
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 0)
        return (
            torch.tensor(d.xpos[self.pelvis].copy(), device=self.device, dtype=torch.float32)[None],
            R,
            torch.tensor(velocity[3:].copy(), device=self.device, dtype=torch.float32)[None],
            torch.tensor(velocity[:3].copy(), device=self.device, dtype=torch.float32)[None],
            torch.tensor(
                (d.qpos[self.qids] * self.signs).copy(), device=self.device, dtype=torch.float32
            )[None],
            torch.tensor(
                (d.qvel[self.vids] * self.signs).copy(), device=self.device, dtype=torch.float32
            )[None],
        )

    @torch.no_grad()
    def inputs(self, arena):
        if self.official_holomotion:
            return self.official_inputs(arena)
        from experiments.g1_competitive_suite31_v1.model import CAPTURE

        kind = self.row["kind"]
        if kind in ("descent", "vault"):
            f = torch.minimum(
                self.reference.clip_start + self.episode_step, self.reference.clip_end
            )
            state = self.state()
            obs = self.mdp.obs(
                state,
                self.last_action,
                torch.zeros(
                    1, self.mdp.xi_dim if hasattr(self.mdp, "xi_dim") else 0, device=self.device
                ),
                f,
            )
            goal = self.reference.goal(f, 10)
            context = (
                self.camera.observe(
                    self.model,
                    self.data,
                    self.model.body("robot/torso_link").id,
                )
                if kind == "vault"
                else None
            )
            return obs[0], goal[0], context
        if kind == "skate":
            self.native.last_action.copy_(self.adapter.sdk_to_old(self.last_action))
            self.native._observed = None
            _, ctx = self.native.student_inputs()
            obs, goal = self.adapter.inputs(CAPTURE["inputs"])
            return obs[0], goal[0], ctx[0]
        sim = self.native
        sim.elapsed_s = arena.data.time - self.cycle_start
        old = self.adapter.sdk_to_old(self.last_action)[0].cpu().numpy()
        context = (torch.tensor(sim.observation(), device=self.device)[None] - self.mean) / self.std
        self.bridge.terms(sim, old)
        obs, goal = self.adapter.inputs(CAPTURE["inputs"])
        return obs[0], goal[0], context[0]

    def official_inputs(self, arena):
        from tensordict import TensorDict
        from blinkmimic.rotations import mat_to_quat
        from modumotion_g1.perceptive_observations import (
            PerceptiveActorObservationInput,
            build_perceptive_actor_terms,
        )

        kind = self.row["kind"]
        old = self.adapter.sdk_to_old(self.last_action)
        context = None
        if kind == "box":
            td, context = self.native.inputs(old)
        elif kind == "skate":
            self.native.last_action.copy_(old)
            self.native._observed = None
            td, context = self.native.student_inputs()
        elif kind == "tennis":
            self.native.elapsed_s = arena.data.time - self.cycle_start
            context = (
                torch.tensor(self.native.observation(), device=self.device)[None] - self.mean
            ) / self.std
            td = self.bridge.terms(self.native, old[0].cpu().numpy())
        elif kind == "rough":
            td, context = self.rough_inputs(arena, old)
        else:
            ids = (self.episode_step + torch.arange(-1, 12, device=self.device)).clamp(
                0, len(self.official_reference) - 1
            )
            state = self.state()
            gather = self.adapter.holo_from_sdk
            terms = build_perceptive_actor_terms(
                PerceptiveActorObservationInput(
                    reference_qpos=self.official_reference[ids][None],
                    robot_root_quat_wxyz=mat_to_quat(state[1]),
                    robot_root_angvel_local=(state[1].transpose(-1, -2) @ state[3][..., None])[
                        ..., 0
                    ],
                    robot_dof_pos=state[4][:, gather],
                    robot_dof_vel=state[5][:, gather],
                    last_action=old,
                    default_dof_pos=self.adapter.default,
                )
            )
            td = TensorDict({f"unified/{k}": v for k, v in terms.items()}, batch_size=[1])
            if kind == "vault":
                context = self.camera.observe(
                    self.model,
                    self.data,
                    self.model.body("robot/torso_link").id,
                )[None]
            if self.row["expert"] == "mimicx_kungfu":
                context = self.martial_context()
        return (
            td[0],
            torch.zeros(730, device=self.device),
            context[0] if context is not None else None,
        )

    def rough_inputs(self, arena, old):
        from tensordict import TensorDict
        from blinkmimic.rotations import mat_to_quat
        from modumotion_g1.perceptive_observations import (
            PerceptiveActorObservationInput,
            build_perceptive_actor_terms,
            _yaw,
        )

        state = self.state()
        R, p = state[1], state[0]
        # The released goal command is measured in its torso-root frame.
        torso = self.model.body("robot/torso_link").id
        rootp = torch.tensor(self.data.xpos[torso].copy(), device=self.device)[None].float()
        rootquat = torch.tensor(self.data.xquat[torso].copy(), device=self.device)[None].float()
        yaw = _yaw(rootquat)
        delta = torch.tensor(self.initial["route_goal"], device=self.device)[None] - rootp
        cfg = self.contract["command_cfg"]
        vx = (delta[:, 0] * yaw.cos() + delta[:, 1] * yaw.sin()) * cfg["velocity_control_stiffness"]
        vy = (-delta[:, 0] * yaw.sin() + delta[:, 1] * yaw.cos()) * cfg[
            "velocity_control_stiffness"
        ]
        angle = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
        angle = torch.atan2(angle.sin(), angle.cos())
        command = torch.stack(
            (
                vx.clamp(0, self.contract["max_command"][0]),
                vy * 0,
                (angle * cfg["heading_control_stiffness"]).clamp(*cfg["ang_vel_z"]),
            ),
            -1,
        )
        command *= (delta[:, :2].norm(dim=-1) > cfg["target_dis_threshold"])[:, None]
        pelvisyaw = _yaw(mat_to_quat(R))
        t = (torch.arange(13, device=self.device) - 1) * 0.02
        angle = pelvisyaw[:, None] + command[:, 2:3] * t
        quat = torch.stack(((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()), -1)
        worldv = torch.stack(
            (
                pelvisyaw.cos() * command[:, 0] - pelvisyaw.sin() * command[:, 1],
                pelvisyaw.sin() * command[:, 0] + pelvisyaw.cos() * command[:, 1],
            ),
            -1,
        )
        xy = p[:, None, :2] + worldv[:, None] * t[None, :, None]
        standing = torch.tensor(self.initial["standing"], device=self.device)[
            self.contract["to_holo"]
        ]
        standing *= torch.tensor(self.contract["sim_joint_signs"], device=self.device)[
            self.contract["to_holo"]
        ]
        reference = torch.cat(
            (xy, p[:, None, 2:3].expand(-1, 13, -1), quat, standing[None, None].expand(1, 13, -1)),
            -1,
        ).float()
        gather = self.adapter.holo_from_sdk
        terms = build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=reference,
                robot_root_quat_wxyz=mat_to_quat(R),
                robot_root_angvel_local=(R.transpose(-1, -2) @ state[3][..., None])[..., 0],
                robot_dof_pos=state[4][:, gather],
                robot_dof_vel=state[5][:, gather],
                last_action=old,
                default_dof_pos=self.adapter.default,
                reference_sample_time=t,
            )
        )
        depth = self.camera.observe(self.model, self.data, torso)
        phase = self.episode_step * 0.02 * (2 * math.pi / 0.6)
        context = torch.cat(
            (depth, command[0], depth.new_tensor([math.sin(phase), math.cos(phase)]))
        )[None]
        return TensorDict({f"unified/{k}": v for k, v in terms.items()}, batch_size=[1]), context

    def martial_context(self):
        frame = min(self.episode_step, len(self.context_reference["joint_pos"]) - 1)
        ref = self.context_reference
        torso = self.model.body(self.name + "/robot/torso_link").id
        anchor_index = torso - self.pelvis
        position = torch.tensor(self.data.xpos[torso].copy(), device=self.device).float()
        rotation = torch.tensor(
            self.data.xmat[torso].reshape(3, 3).copy(), device=self.device
        ).float()
        anchor_position = rotation.T @ (ref["body_pos_w"][frame, anchor_index] - position)
        anchor_rotation = rotation.T @ quat_to_mat(ref["body_quat_w"][frame, anchor_index])
        imu = self.model.site(self.name + "/robot/imu_in_pelvis").id
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, imu, velocity, 1)
        state = self.state()
        value = torch.cat(
            (
                ref["joint_pos"][frame],
                ref["joint_vel"][frame],
                anchor_position,
                anchor_rotation[:, :2].flatten(),
                torch.tensor(velocity[3:], device=self.device).float(),
                torch.tensor(velocity[:3], device=self.device).float(),
                state[4][0] - self.actor.pi.ref_q0,
                state[5][0],
                # mjlab resolves joint action targets in natural joint order,
                # independently of the exported XML actuator control order.
                self.last_action[0],
            )
        )[None]
        return (value - self.context_mean) / (self.context_std + self.context_eps)

    def execute(self, action, arena, raw_action=None):
        if self.row["kind"] == "box":
            executed = self.native.execute(action, raw_action)
            self.last_action.copy_(executed[None])
            return
        target = (self.actor.pi.ref_q0 + self.actor.pi.ref_alpha * action).cpu().numpy()
        self.data.ctrl[self.uids] = target * self.signs
        arena.data.ctrl[self.map.umap] = self.data.ctrl
        self.last_action.copy_(action[None])
        if self.row["kind"] == "tennis":
            old = self.adapter.sdk_to_old(action[None])[0].cpu().numpy()
            self.native.last_action[:] = self.bridge.native(self.native, old)
        if self.row["kind"] == "skate":
            from modumotion_g1.husky_skateboarding import DEFAULT23, SCALE23

            holo_target = (
                self.native.holo_to_target(self.adapter.sdk_to_old(action[None]))[0].cpu().numpy()
            )
            self.native.executed_teacher_action[0] = (
                holo_target[self.native.teacher_holo_indices] - DEFAULT23
            ) / SCALE23
            self.native.steps[0] += 1

    def after_control(self, arena):
        self.episode_step += 1
        state = self.state()
        if self.reference is not None:
            f = torch.minimum(
                self.reference.clip_start + self.episode_step, self.reference.clip_end
            )
            root = float((state[0] - self.reference.root_p[f]).norm())
            height = float(abs(state[0][0, 2] - self.reference.root_p[f, 2]))
            tilt = float(abs(state[1][0, 2, 2] - self.reference.root_R[f, 2, 2]))
            self.lost |= height > 0.25 or tilt > 0.8
        else:
            root = None
            self.lost |= float(state[0][0, 2]) < 0.35 or float(state[1][0, 2, 2]) < 0.3
        if self.row["kind"] == "skate":
            self.lost |= bool(self.native.metrics()["board_distance"][0] > 0.8)
        record = dict(
            time_s=float(arena.data.time),
            episode=self.episode,
            frame=self.episode_step,
            root_error_m=root,
            tracking_lost=bool(self.lost),
            pelvis_height_m=float(state[0][0, 2]),
        )
        if self.row["kind"] == "skate":
            metrics = self.native.metrics()
            record["board_displacement_m"] = float(
                np.linalg.norm(metrics["board_xy"][0] - self.board_start)
            )
            record["board_distance_m"] = float(metrics["board_distance"][0])
            foot, wheel = self.native.contact_flags()[0]
            record["native_mirror_foot_board_contact"] = bool(foot)
            record["native_mirror_wheel_ground_contact"] = bool(wheel)
        if self.row["kind"] == "box":
            record["box_metrics"] = self.native.audit(arena)
            self.lost |= self.native.world.fallen
            record["tracking_lost"] = bool(self.lost)
        if self.row["kind"] == "vault":
            record["depth_pixel_std"] = float(self.camera.last_image.std())
        if self.row["kind"] == "rough":
            record["depth_pixel_std"] = float(self.camera.last_image.std())
            record["goal_distance_m"] = float(
                np.linalg.norm(self.data.qpos[:2] - self.initial["route_goal"][:2])
            )
            record["displacement_m"] = float(
                np.linalg.norm(self.data.qpos[:2] - self.initial["qpos"][:2])
            )
            near = (
                record["goal_distance_m"] < 0.4
                and record["pelvis_height_m"] > 0.65
                and float(state[1][0, 2, 2]) > 0.7
                and not self.lost
                and record["displacement_m"] >= 2.0
            )
            self.goal_dwell_steps = self.goal_dwell_steps + 1 if near else 0
            self.goal_ready = self.goal_dwell_steps >= 50
            record["goal_dwell_s"] = self.goal_dwell_steps * 0.02
            record["goal_complete"] = bool(self.goal_ready)
        self.records.append(record)
        if self.episode_step == self.horizon and self.reference is not None:
            self.completed.append(
                dict(episode=self.episode, tracking_lost=bool(self.lost), endpoint_error_m=root)
            )


class Arena:
    def __init__(self, config, actor):
        self.config = config
        self.actor = actor
        self.slots = [Slot(t, actor, config) for t in config["tasks"]]
        spec = mujoco.MjSpec()
        spec.option.timestep = config["physics_dt"]
        spec.option.integrator = getattr(
            mujoco.mjtIntegrator, config.get("physics_integrator", "mjINT_IMPLICITFAST")
        )
        spec.option.solver = getattr(mujoco.mjtSolver, config.get("physics_solver", "mjSOL_NEWTON"))
        spec.option.iterations, spec.option.ls_iterations = config.get("solver_iterations", 10), 20
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1000
        spec.visual.quality.shadowsize = 2048
        spec.worldbody.add_light(
            name="arena_sun",
            pos=[15, 8, 20],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.6, 0.6, 0.6],
            ambient=[0.2, 0.2, 0.2],
        )
        # Display-only backdrop: excluded from contacts and the depth camera.
        spec.worldbody.add_geom(
            name="arena_backdrop",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[15, 8, -0.15],
            size=[70, 40, 0.1],
            rgba=[0.19, 0.24, 0.29, 1],
            contype=0,
            conaffinity=0,
            group=1,
        )
        collision_bit_offset = 0
        for i, slot in enumerate(self.slots):
            child = slot.spec.copy()
            for light in list(child.lights):
                child.delete(light)
            for key in list(child.keys):
                child.delete(key)
            child.compile()
            for geom_index, geom in enumerate(list(child.geoms)):
                if not geom.name:
                    geom.name = f"_arena_geom_{geom_index}"
                if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                    hx, hy, cx, *rest = slot.row["floor"]
                    if slot.row["kind"] == "box":
                        # Retain native plane contacts for grasp-sensitive box dynamics.
                        # A separate finite patch supplies only its visual appearance.
                        geom.rgba = [0, 0, 0, 0]
                        child.worldbody.add_geom(
                            name="box_display_floor",
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[hx, hy, 0.05],
                            pos=[cx, rest[0] if rest else 0, float(geom.pos[2]) - 0.05],
                            material=geom.material,
                            rgba=[0.3, 0.36, 0.42, 1],
                            contype=0,
                            conaffinity=0,
                            group=1,
                            density=0,
                        )
                    else:
                        geom.type = mujoco.mjtGeom.mjGEOM_BOX
                        geom.size = [hx, hy, 0.05]
                        geom.density = 0
                        geom.pos = [cx, rest[0] if rest else 0, float(geom.pos[2]) - 0.05]
            if config.get("compact"):
                from pera.compact_showcase import compact_child_spec

                collision_bit_offset += compact_child_spec(
                    child, slot.row, i, bit_offset=collision_bit_offset
                )
            frame = spec.worldbody.add_frame(pos=slot.origin)
            spec.attach(child, prefix=slot.name + "/", frame=frame)
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        if getattr(actor, "official_holomotion", False):
            from pera.holomotion_showcase import OfficialMultiRobotController

            self.controller = OfficialMultiRobotController(
                actor, [s.row["expert"] for s in self.slots]
            )
        else:
            self.controller = MultiRobotController(actor, [s.row["expert"] for s in self.slots])
        self.tick, self.ball_launches, self.ball_rebounds = 0, 1, 0
        for slot in self.slots:
            slot.map = Mirror(slot, self.model)
            slot.map.push_reset(self)
        self.sync_markers()
        mujoco.mj_forward(self.model, self.data)
        self.sync(initialize=True)
        self.decimation = round(config["control_dt"] / config["physics_dt"])
        self.reset_events = []
        self.fault_events = []

    def sync(self, *, initialize=False):
        for slot in self.slots:
            # ATHLETE observes mj_step's pre-integration derived fields alongside
            # the integrated qpos/qvel. Recomputing its FK/IMU here changes the
            # observable history relative to its original training interface.
            slot.map.pull(self, forward=initialize or slot.row["kind"] not in ("tennis", "box"))

    def sync_markers(self):
        tennis = next(s for s in self.slots if s.row["kind"] == "tennis")
        for name in ("landing_target_marker", "landing_target_center"):
            local = tennis.model.geom(name).id
            target = tennis.map.geomap[local]
            self.model.geom_pos[target] = tennis.model.geom_pos[local] + tennis.origin
            self.model.geom_rgba[target] = tennis.model.geom_rgba[local]

    @torch.no_grad()
    def step(self, repeat=True):
        for i, slot in enumerate(self.slots):
            boundary = slot.horizon if slot.reference is not None else slot.row.get("repeat_steps")
            if repeat and boundary is not None and slot.episode_step >= boundary:
                self.reset_events.append(
                    dict(
                        task=slot.name,
                        time_s=float(self.data.time),
                        tracking_lost=bool(slot.lost),
                        reason="reference_boundary"
                        if slot.reference is not None
                        else "timed_skill_loop",
                    )
                )
                slot.reset_native()
                slot.episode += 1
                slot.map.push_reset(self)
                self.controller.reset([i])
                mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.sync()
        inputs = [slot.inputs(self) for slot in self.slots]
        obs, goal = torch.stack([x[0] for x in inputs]), torch.stack([x[1] for x in inputs])
        contexts = [x[2] if not s.faulted else None for x, s in zip(inputs, self.slots)]
        # Isolate an invalid expert without ever sending its action to physics.
        # The fallback is explicitly logged and displayed as public-only control.
        for _ in range(len(self.slots) + 1):
            try:
                actions = self.controller.infer(obs, goal, contexts)
                break
            except NonfinitePolicyAction as exc:
                if getattr(self.actor, "official_holomotion", False):
                    raise
                experts = list(self.controller.experts)
                for i in exc.robot_ids:
                    if self.slots[i].faulted:
                        raise
                    self.slots[i].faulted = True
                    experts[i], contexts[i] = None, None
                    self.fault_events.append(
                        dict(
                            task=self.slots[i].name,
                            time_s=float(self.data.time),
                            reason="nonfinite_expert_action",
                            fallback="public_only",
                        )
                    )
                self.controller.experts = tuple(experts)
        else:
            raise RuntimeError("No finite public fallback")
        for slot, action in zip(self.slots, actions):
            slot.execute(action, self)
        tennis = next(s for s in self.slots if s.row["kind"] == "tennis")
        if not tennis.native.ball_released:
            tennis.native._set_ball_state(
                tennis.native.incoming_ball_plan.launch_position_w,
                tennis.native._planned_launch_velocity_w,
            )
            tennis.native.ball_released = True
            tennis.map.push_ball(self)
        for _ in range(self.decimation):
            tennis.map.pull(self)
            tennis.native._apply_ball_aerodynamics()
            self.data.xfrc_applied[tennis.map.bmap[tennis.native.ball_body_id]] = (
                tennis.data.xfrc_applied[tennis.native.ball_body_id]
            )
            mujoco.mj_step(self.model, self.data)
            tennis.map.pull(self, forward=False)
            pending = tennis.native._ground_impact_pending
            tennis.native._apply_explicit_ground_rebound()
            self.ball_rebounds += int(pending and not tennis.native._ground_impact_pending)
            tennis.native._ball_racket_contact()
            tennis.map.push_ball(self)
        self.tick += 1
        self.sync()
        for i, slot in enumerate(self.slots):
            slot.after_control(self)
            slot.records[-1]["expert_contribution"] = self.controller.last_dispatch[i][
                "contribution"
            ][0]
        tennis.audit.check(float(self.data.time))
        sim = tennis.native
        sim.elapsed_s = self.data.time - tennis.cycle_start
        if sim.elapsed_s >= sim.episode_duration_s:
            tennis.audit.end_cycle(True)
            tennis.audit.begin()
            before = self.data.qpos[tennis.map.qmap[:36]].copy()
            sim.start_next_strike_cycle()
            self.sync_markers()
            sim._set_ball_state(sim.incoming_ball_plan.launch_position_w)
            sim.ball_released = False
            tennis.map.push_ball(self)
            tennis.cycle_start = float(self.data.time)
            self.ball_launches += 1
            assert np.array_equal(before, self.data.qpos[tennis.map.qmap[:36]])
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite shared physical state")


def render_frame(renderer, arena):
    if arena.config.get("compact"):
        from pera.compact_showcase import render_frame as compact_frame

        return compact_frame(renderer, arena)
    from PIL import Image, ImageDraw, ImageFont

    frame = Image.new("RGB", (1920, 1008), (16, 24, 34))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    draw.text(
        (18, 12),
        "PERA  |  HOLOMOTION OFFICIAL BASE + TRAINED EXPERTS  |  ONE WORLD",
        font=font,
        fill="white",
    )
    draw.text(
        (18, 40),
        f"Student actions only  |  t={arena.data.time:.2f}s  |  {len(arena.reset_events)} labeled reference-boundary resets",
        font=small,
        fill="#bdd0df",
    )
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance = [15, 8, 0.5], 42
    cam.azimuth, cam.elevation = 90, -60
    renderer.update_scene(arena.data, camera=cam)
    frame.paste(Image.fromarray(renderer.render().copy()), (0, 72))
    for i, slot in enumerate(arena.slots):
        p = arena.data.xpos[slot.map.bmap[slot.pelvis]].copy()
        if slot.row["kind"] == "tennis":
            cam.lookat[:], cam.distance = slot.origin + [1.8, 0, 0.7], 6.8
        elif slot.row["kind"] == "vault":
            cam.lookat[:] = slot.origin + slot.reference.root_p.mean(0).cpu().numpy()
            cam.distance = 7
        elif slot.row["kind"] == "descent":
            cam.lookat[:], cam.distance = slot.origin + [1.4, 0, 0.5], 6.2
        else:
            cam.lookat[:], cam.distance = p, 4
        cam.azimuth, cam.elevation = 120, -28
        renderer.update_scene(arena.data, camera=cam)
        panel = Image.fromarray(renderer.render().copy()).resize((480, 360))
        x, y = 960 + (i % 2) * 480, 72 + (i // 2) * 360
        frame.paste(panel, (x, y))
        draw.rectangle((x, y, x + 480, y + 28), fill=(15, 22, 30))
        draw.text((x + 8, y + 4), slot.row["label"], font=small, fill="white")
    vault = next(s for s in arena.slots if s.row["kind"] == "vault")
    if vault.camera.last_image is not None:
        pixels = np.clip(vault.camera.last_image * 255, 0, 255).astype(np.uint8)
        frame.paste(Image.fromarray(pixels).resize((320, 180), Image.Resampling.NEAREST), (15, 812))
        draw.text((350, 817), "LIVE VAULT DEPTH INPUT", font=small, fill="white")
        draw.text(
            (350, 848), "18 x 32 -> frozen encoder -> expert context32", font=small, fill="#bdd0df"
        )
        draw.text(
            (350, 882), "Native sensor noise / normalization retained", font=small, fill="#bdd0df"
        )
    for i, slot in enumerate(arena.slots):
        text = f"{slot.row['label']}: episode {slot.episode}, step {slot.episode_step}"
        if slot.lost:
            text += "  TRACKING LOST"
        if slot.faulted:
            text += "  EXPERT FAULT: PUBLIC ONLY"
        if slot.episode > 1 and slot.episode_step <= 25:
            text += "  RESET TO START"
        draw.text((980, 815 + 39 * i), text, font=small, fill="#ff8a80" if slot.lost else "#b8e4c5")
    return np.asarray(frame)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/showcase.json")
    p.add_argument("--native-source-root", type=Path, default=ROOT)
    p.add_argument(
        "--bank", type=Path, help="One official HoloMotion checkpoint containing all task experts"
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        help="Directory containing official config.yaml and model_14000; may be outside the checkout",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--video", action="store_true")
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--check-isolation", action="store_true")
    args = p.parse_args(argv)
    if not 0 < args.seconds <= 120:
        p.error("Choose seconds in (0,120]")
    from pera.demo_policy import enforce_demo_policy

    config = json.loads(args.config.read_text())
    try:
        enforce_demo_policy(config)
    except ValueError as exc:
        p.error(str(exc))
    native = args.native_source_root.resolve()
    runtime = ROOT / "vendor" if native == ROOT else native / "third_party"
    sys.path[:0] = [
        str(ROOT / "reproduction"),
        str(ROOT / "integrations/HoloMotion"),
        str(runtime / "ATHLETE/athlete/src"),
        str(ROOT / "reproduction/experiments/ultra_box_v1"),
        str(runtime / "ULTRA/ultra"),
        str(runtime / "InstinctMJ/src"),
        str(runtime / "mjlab/src"),
        str(runtime / "instinct_rl"),
    ]
    # Import mjlab's actuator namespace before its spec helper (upstream cycle).
    import mjlab.actuator  # noqa: F401 -- initialize upstream import cycle
    from experiments.athlete_tennis_v1.compat import setup

    setup()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    for manifest in config.get("resource_manifests", []):
        check_resources({"resource_manifest": manifest})
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_hashes = {
        filename: file_digest(ROOT / filename)
        for filename in (
            "src/pera/interaction_showcase.py",
            "src/pera/holomotion_showcase.py",
            "src/pera/compact_showcase.py",
            "src/pera/box_showcase.py",
            "src/pera/multirate_showcase.py",
            "configs/demo_policy.json",
            "configs/showcase.json",
            "configs/showcase_motion_regions.json",
        )
    }
    from pera.holomotion_showcase import OfficialShowcaseActor

    actor = OfficialShowcaseActor(native, args.device, base_dir=args.base_dir, bank_path=args.bank)
    digest = actor.public_sha256
    if any(t["kind"] == "box" for t in config["tasks"]):
        from pera.multirate_showcase import MultiRateArena

        arena = MultiRateArena(config, actor)
    else:
        arena = Arena(config, actor)
    arena.controller.validate_on_next_infer = args.check_isolation
    (out / "world.xml").write_text(arena.spec.to_xml())
    renderer = writer = viewer = None
    error = None
    try:
        if args.video:
            import imageio.v2 as imageio

            binary = shutil.which("ffmpeg")
            if not binary:
                raise RuntimeError("Local ffmpeg is required")
            os.environ["IMAGEIO_FFMPEG_EXE"] = binary
            renderer = mujoco.Renderer(
                arena.model,
                height=900 if config.get("compact") else 720,
                width=1440 if config.get("compact") else 960,
            )
            writer = imageio.get_writer(out / "overview.mp4", fps=25, codec="libx264", quality=8)
        if args.viewer:
            import mujoco.viewer as mjviewer

            viewer = mjviewer.launch_passive(arena.model, arena.data)
            viewer.cam.lookat[:], viewer.cam.distance = [15, 8, 0.5], 42
            viewer.cam.azimuth, viewer.cam.elevation = 90, -60
            if config.get("compact"):
                cam = config["overview_camera"]
                viewer.cam.lookat[:], viewer.cam.distance = cam["lookat"], cam["distance"]
                viewer.cam.azimuth, viewer.cam.elevation = cam["azimuth"], cam["elevation"]
        for tick in range(math.ceil(args.seconds / config["control_dt"])):
            arena.step(args.repeat)
            if writer is not None and tick % 2 == 0:
                frame = render_frame(renderer, arena)
                writer.append_data(frame)
                if tick == 0:
                    from PIL import Image

                    Image.fromarray(frame).save(out / "preview.png")
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
        if viewer:
            viewer.close()
        tennis = next(s for s in arena.slots if s.row["kind"] == "tennis")
        assert actor.verify() == digest
        from pera.runtime_audit import imported_sources, versions

        runtime_sources = imported_sources(strict=native == ROOT)
        result = dict(
            schema="pera.interaction_arena.v1",
            status="failed" if error else "completed",
            error=error,
            runtime_sources=runtime_sources,
            environment=versions(),
            actor_instances=1,
            physics_worlds=1,
            physics_mjdata_instances=1,
            auxiliary_native_mirrors=len(arena.slots),
            auxiliary_role="FK, sensor and native tennis planner bookkeeping only; never physics stepped",
            auxiliary_physics_steps=0,
            student_control_steps=arena.tick,
            robot_control_steps={s.name: len(s.records) for s in arena.slots},
            robot_control_hz={s.name: 60 if s.row["kind"] == "box" else 50 for s in arena.slots},
            physics_steps=getattr(arena, "actual_physics_steps", arena.tick * arena.decimation),
            simulated_seconds=float(arena.data.time),
            physics_dt=config["physics_dt"],
            control_dt=config["control_dt"],
            solver_iterations=int(arena.model.opt.iterations),
            solver_ls_iterations=int(arena.model.opt.ls_iterations),
            physics_integrator=int(arena.model.opt.integrator),
            physics_solver=int(arena.model.opt.solver),
            fifty_hz_control_max_clock_quantization_s=config["physics_dt"]
            if hasattr(arena, "next_due")
            else 0,
            mujoco_version=mujoco.__version__,
            teacher_action_queries=0,
            base_family="holomotion_official_g1_v1.4",
            checkpoint="model_14000",
            checkpoint_sha256=actor.policy["base_files"]["model_14000/actor/model.safetensors"][
                "sha256"
            ],
            public_sha256=digest,
            source_sha256=source_hashes,
            expert_checkpoints=actor.policy["experts"],
            expert_bank_sha256=file_digest(args.bank) if args.bank else None,
            resident_expert_count=len(actor.expert_names),
            routing="original_checkpoint_continuous_anchor_group_mixture_with_native_public_Top2",
            independent_temporal_caches=len(arena.controller.caches),
            isolation_check=arena.controller.isolation_report,
            reset_events=arena.reset_events,
            expert_fault_events=arena.fault_events,
            robot_reset_count=len(arena.reset_events),
            ball_launches=arena.ball_launches,
            ball_rebounds=arena.ball_rebounds,
            tennis_cycles=tennis.audit.records,
            tennis_current_cycle=tennis.audit.current,
            tennis_first_unstable_s=tennis.audit.first_unstable,
            max_active_experts_per_inference=1,
            tasks=[
                dict(config=s.row, records=s.records, completed_reference_intervals=s.completed)
                for s in arena.slots
            ],
            elapsed_s=time.monotonic() - started,
            torch_cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_initialized()
            else 0,
            limitations=[
                "Shared-world scene port, not original native acceptance gates or a success-rate estimate",
                "Shared physical timestep is recorded explicitly; this scene port is not identical to each original native simulator",
                "Vault and parkour crossing use live depth internally; HUSKY uses proprioceptive128D, box uses native normalized4052D, and tennis uses ball-state631D context",
                "Labeled episode repeats reset selected robots at reference boundaries or configured loop duration; no learned continuous transitions",
                "All robot actions come from one unchanged official HoloMotion Actor with matching trained FFN/router experts; teacher action heads are not evaluated",
                "Compact skill regions use separate collision masks; no cross-region object interaction is claimed",
            ],
        )
        (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if k not in ("tasks",)}, indent=2),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
