"""ULTRA's released MuJoCo teacher, immutable physics and independent worlds.

Only startup compatibility is patched, inside a scoped context: CPU storage
loading, rl_games 1.1.x constructor defaults, and optional renderer suppression.
The native network, normalization, reset, assets, PD and 17 physics substeps are
retained. No target-object state writes occur during an episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from torch import nn

from pera.paths import ROOT

SOURCE = ROOT / "third_party/ULTRA"
CHECKPOINT = SOURCE / "ultra/weights/teacher_ultra_inference.pth"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


class NullRenderer:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def load_native(motion, device="cpu", renderer=False):
    """Construct the unchanged official class against existing dependencies."""
    import sim2sim_teacher as native
    from learning import ultra_models
    from rl_games.algos_torch.models import ModelA2CContinuousLogStd

    original_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    def compatible_network_init(self, network):
        ModelA2CContinuousLogStd.Network.__init__(
            self,
            network,
            obs_shape=(4052,),
            normalize_value=False,
            normalize_input=False,
            value_size=1,
        )

    with (
        patch.object(torch, "load", cpu_load),
        patch.object(
            ultra_models.ModelUltraContinuous.Network, "__init__", compatible_network_init
        ),
    ):
        if renderer:
            env = native.HumanoidEnv(
                str(CHECKPOINT), device=device, record_video=True, motion_path=str(motion)
            )
        else:
            with patch.object(mujoco, "Renderer", NullRenderer):
                env = native.HumanoidEnv(
                    str(CHECKPOINT), device=device, record_video=True, motion_path=str(motion)
                )
    env.policy.eval().requires_grad_(False)
    return env


class FrozenTeacher(nn.Module):
    def __init__(self, env, device="cpu"):
        super().__init__()
        self.policy = env.policy
        self.register_buffer("mean", env.running_mean.detach().float().clone())
        self.register_buffer("variance", env.running_var.detach().float().clone())
        self.to(device).eval().requires_grad_(False)

    def normalize(self, obs):
        return ((obs - self.mean) / (self.variance + 1e-5).sqrt()).clamp(-5, 5)

    def forward(self, obs):
        normalized = self.normalize(obs)
        # Native eval_actor is the exact deterministic actor; skip unused
        # critic and stochastic sample generation in the inference wrapper.
        return self.policy.a2c_network.eval_actor(normalized)[0].clamp(-1, 1)

    @torch.no_grad()
    def verify(self, observations):
        normalized = self.normalize(observations)
        expected = self.policy(
            {"is_train": False, "prev_actions": None, "obs": normalized.clone(), "rnn_states": None}
        )["mus"].clamp(-1, 1)
        actual = self(observations)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        return {
            "samples": len(observations),
            "max_abs_error": float((actual - expected).abs().max()),
            "teacher_updates": 0,
            "native_inference_wrapper_parity": True,
        }


class World:
    def __init__(self, template):
        import sim2sim_teacher as native

        self.env = copy.copy(template)
        self.model = template.model
        self.data = mujoco.MjData(self.model)
        self.env.data = self.data
        # Reset has an in-place +1cm root offset. Restore a pristine reference
        # for each reset so it always matches a fresh native invocation.
        self.reference = template.hoi_ref.detach().cpu().clone()
        self.env.hoi_ref = self.reference.clone()
        self.builder = native.MujocoObs(
            self.model,
            template.object_name,
            template.max_episode_length,
            template.hoi_data,
            template.object_points,
            history_step=10,
        )
        self.joint_qpos = self.builder.dof_qpos_ids
        self.joint_qvel = self.builder.dof_qvel_ids
        self.joint_names = [
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, int(self.model.actuator_trnid[i, 0])
            )
            for i in range(self.model.nu)
        ]
        if self.joint_names != native.JOINT_NAMES_29:
            raise ValueError("Unadapted native actuator/joint order")
        self.object_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, template.object_name
        )
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.hand_bids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_rubber_hand", "right_rubber_hand")
        }
        self.object_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, template.object_name + "_geom"
        )
        if (
            self.model.body_mocapid[self.object_bid] >= 0
            or self.model.body_mass[self.object_bid] <= 0
        ):
            raise ValueError("Expected the native dynamic free object")
        self.original_frames = template.hoi_data.shape[0] // 2
        self.end_frame = min(600, template.hoi_data.shape[0] - 1)
        self.reset()

    def reset(self, frame=0):
        self.env.hoi_ref.copy_(self.reference)
        self.env.reset_human_and_object_from_ref(int(frame))
        self.frame = int(frame)
        self.last_action = torch.zeros(1, 29)
        self.last_torque = torch.zeros(1, 29)
        self.builder.last_dof_pos = self.builder.last_dof_vel = None
        self.start_position = self.data.xpos[self.object_bid].copy()
        self.minimum_pelvis = float("inf")
        self.max_lift = self.max_displacement = 0.0
        self.max_lifted_displacement = 0.0
        self.hand_contact_frames = self.lifted_contact_frames = 0
        self.fallen = False
        self.object_state_writes_after_reset = 0

    def observation(self):
        return self.builder._compute_observations(
            self.data,
            self.frame,
            self.last_action,
            self.last_torque,
            student_obs=False,
            episode_length=self.frame,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Invalid native joint action")
        # Native first multiplies float32 action by 3, then assigns to the
        # float64 step_actions buffer. Preserve that rounding exactly.
        pd_target = (action * np.float32(3.0)).astype(np.float64)
        self.last_action = torch.from_numpy(action.astype(np.float32))[None]
        for _ in range(self.env.sim_decimation):
            # Match native float32 q/dq extraction and float64 PD parameters.
            q = self.data.qpos[self.joint_qpos].astype(np.float32)
            dq = self.data.qvel[self.joint_qvel].astype(np.float32)
            torque = np.clip(
                (pd_target - q) * self.env.stiffness - dq * self.env.damping,
                -self.env.torque_limits,
                self.env.torque_limits,
            )
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
        self.last_torque = torch.from_numpy(torque.astype(np.float32))[None]
        self.frame += 1
        self.audit()

    def audit(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite native MuJoCo state")
        height = float(self.data.xpos[self.pelvis_bid, 2])
        upright = float(self.data.xmat[self.pelvis_bid].reshape(3, 3)[2, 2])
        self.minimum_pelvis = min(self.minimum_pelvis, height)
        self.fallen |= height < 0.30 or upright < -0.2
        obj = self.data.xpos[self.object_bid]
        lift = float(obj[2] - self.start_position[2])
        displacement = float(np.linalg.norm(obj[:2] - self.start_position[:2]))
        self.max_lift = max(self.max_lift, lift)
        self.max_displacement = max(self.max_displacement, displacement)
        contact = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]
            if (b1 == self.object_bid and b2 in self.hand_bids) or (
                b2 == self.object_bid and b1 in self.hand_bids
            ):
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                contact |= np.linalg.norm(force[:3]) > 0.1
        self.hand_contact_frames += int(contact)
        self.lifted_contact_frames += int(contact and lift > 0.10)
        if contact and lift > 0.10:
            self.max_lifted_displacement = max(
                getattr(self, "max_lifted_displacement", 0.0), displacement
            )

    def summary(self):
        return {
            "frame": self.frame,
            "reference_frames": self.original_frames,
            "reference_completed": self.frame >= self.original_frames,
            "physical_survival": not self.fallen,
            "minimum_pelvis_height_m": self.minimum_pelvis,
            "max_object_lift_m": self.max_lift,
            "max_object_xy_displacement_m": self.max_displacement,
            "hand_object_contact_seconds": self.hand_contact_frames * self.env.control_dt,
            "lifted_contact_seconds": self.lifted_contact_frames * self.env.control_dt,
            "max_lifted_xy_displacement_m": getattr(self, "max_lifted_displacement", 0.0),
            "object_state_writes_after_reset": self.object_state_writes_after_reset,
            "physical_carry_demonstrated": (
                not self.fallen
                and self.max_lift > 0.10
                and self.lifted_contact_frames * self.env.control_dt >= 0.3
                and getattr(self, "max_lifted_displacement", 0.0) >= 0.30
            ),
        }
