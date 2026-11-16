"""Named-joint native MimicX -> frozen Blink-G1608 observation/29-action bridge."""

import json
from pathlib import Path
import numpy as np, torch
from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import quat_to_mat


class Bridge:
    def __init__(self, wrapped, pi, robot, motion_path, out):
        self.wrapped = wrapped
        self.env = wrapped.unwrapped
        self.pi = pi
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("motion")
        self.action = self.env.action_manager.get_term("joint_pos")
        self.device = self.env.device
        self.N = self.env.num_envs
        sdk = list(robot.joint_names)
        native = list(self.robot.joint_names)
        assert len(native) == 29 and set(native) == set(sdk)
        self.from_native = torch.tensor([native.index(n) for n in sdk], device=self.device)
        action_names = [native[i] for i in self.action._target_ids]
        self.to_action = torch.tensor([sdk.index(n) for n in action_names], device=self.device)
        self.from_action = torch.tensor([action_names.index(n) for n in sdk], device=self.device)
        self.q0 = pi.ref_q0
        self.alpha = pi.ref_alpha
        self.last_action = torch.zeros(self.N, 29, device=self.device)
        self.hist = None
        self.reset_pending = torch.ones(self.N, dtype=torch.bool, device=self.device)
        with np.load(motion_path) as d:
            assert abs(float(d["fps"][0]) - 50) < 1e-8
            q = d["joint_pos"][:, self.from_native.cpu().numpy()]
            p = d["body_pos_w"][:, 0]
            quat = d["body_quat_w"][:, 0]
        reference = Path(out) / "reference"
        reference.mkdir(exist_ok=True)
        np.savez_compressed(
            reference / "mimicx_stairs.npz",
            root_pos=p,
            root_quat_wxyz=quat,
            joint_pos=q,
            fps=np.float64(50),
        )
        kin = Kinematics(robot, self.device)
        self.motion = MotionLib.from_dir(str(reference), robot, kin, self.device)
        self.mdp = MDP(kin, self.motion, self.q0, XiLayout(robot, self.device).dim, robot.sensing)
        self.xi = torch.zeros(self.N, self.mdp.xi_dim, device=self.device)
        self.contract = dict(
            student_joint_names=sdk,
            native_joint_names=native,
            native_to_sdk_indices=self.from_native.cpu().tolist(),
            target_joint_names=action_names,
            action_offset_student=self.q0.tolist(),
            action_scale_student=self.alpha.tolist(),
            motion_arrays_order="native simulator robot joint_names; MotionCommand uses joint_pos directly",
            root="pelvis",
            root_source="motion body index0 / free joint",
            observation_dim=608,
            goal_dim=730,
            hist=10,
            dt=0.02,
            world_origin_max_abs=float(self.env.scene.env_origins.abs().max()),
        )
        assert self.contract["world_origin_max_abs"] < 1e-6, (
            "Native shared-coordinate scene must remain at origin"
        )
        (Path(out) / "bridge_contract.json").write_text(json.dumps(self.contract, indent=2))
        probe = torch.randn(self.N, 29, device=self.device)
        torch.testing.assert_close(
            self.to_student(self.to_native(probe, clip=False)), probe, atol=3e-6, rtol=3e-6
        )

    def state(self):
        idx = self.robot.indexing
        root = self.env.sim.data.qpos[:, idx.free_joint_q_adr]
        vel = self.env.sim.data.qvel[:, idx.free_joint_v_adr]
        R = quat_to_mat(root[:, 3:7])
        return (
            root[:, :3] - self.env.scene.env_origins,
            R,
            vel[:, :3].clone(),
            (R @ vel[:, 3:6, None])[..., 0],
            self.robot.data.joint_pos[:, self.from_native].clone(),
            self.robot.data.joint_vel[:, self.from_native].clone(),
        )

    def observe(self):
        f = self.command.time_steps.clamp(max=self.motion.num_frames - 2)
        o = self.mdp.obs(self.state(), self.last_action, self.xi, f)
        if self.hist is None:
            self.hist = o[:, None].repeat(1, 10, 1)
        else:
            self.hist = torch.cat((self.hist[:, 1:], o[:, None]), 1)
        self.hist = torch.where(
            self.reset_pending[:, None, None], o[:, None].expand(-1, 10, -1), self.hist
        )
        self.reset_pending.zero_()
        return self.hist, self.motion.goal(f, 10)

    def to_student(self, a):
        return (
            (a * self.action.scale + self.action.offset)[:, self.from_action] - self.q0
        ) / self.alpha

    def to_native(self, a, clip=True):
        x = ((self.q0 + self.alpha * a)[:, self.to_action] - self.action.offset) / self.action.scale
        return (
            x.clamp(-self.wrapped.clip_actions, self.wrapped.clip_actions)
            if clip and self.wrapped.clip_actions is not None
            else x
        )

    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        self.last_action.zero_()
        self.hist = None
        self.reset_pending.fill_(True)
        return obs

    def stepped(self, native, done):
        self.last_action.copy_(self.to_student(native))
        self.last_action[done.bool()] = 0
        self.reset_pending |= done.bool()
