"""Named-joint and canonical observation bridge; no trainable encoders."""

import hashlib
import json

import numpy as np
import torch
from holomotion.src.motion_tracking.actor_observation import (
    MotionActorObservationInput,
    derive_motion_actor_terms_torch,
)
from tensordict import TensorDict
from pera.paths import ROOT


class MimicXBridge:
    def __init__(self, env, config, motion_path):
        self.env = env
        self.robot = env.scene["robot"]
        self.command = env.command_manager.get_term("motion")
        self.action = env.action_manager.get_term("joint_pos")
        self.device = env.device
        contract_path = ROOT / "configs/g1_actor_io_contract.json"
        contract = json.loads(contract_path.read_text())
        self.contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        self.names = contract["joint_names"]
        assert contract["action_joint_ids"] == "slice(None, None, None)"
        assert set(self.names) == set(config.robot.dof_names)
        assert contract["step_dt"] == 0.02
        self.reference_indices = torch.tensor(
            [list(config.robot.dof_names).index(n) for n in self.names],
            dtype=torch.int32,
            device=self.device,
        )
        # The released motion's order was independently FK-verified.
        self.joints = [self.robot.joint_names.index(n) for n in self.names]
        target_names = [self.robot.joint_names[i] for i in self.action._target_ids]
        self.to_action = [self.names.index(n) for n in target_names]
        self.from_action = [target_names.index(n) for n in self.names]
        self.offset = torch.tensor(contract["action_offset"], device=self.device)
        self.default_pos = torch.tensor(contract["default_joint_pos"], device=self.device)
        self.scale = torch.tensor(contract["action_scale"], device=self.device)
        with np.load(motion_path, allow_pickle=False) as data:
            qpos = np.concatenate(
                (
                    data["body_pos_w"][:, 0],
                    data["body_quat_w"][:, 0],
                    data["joint_pos"],
                ),
                axis=-1,
            )
        self.qpos = torch.tensor(qpos, device=self.device)
        self.last_action = torch.zeros(env.num_envs, 29, device=self.device)
        self.num_future = int(config.obs.n_fut_frames)

    def observations(self):
        frames = (
            self.command.time_steps[:, None]
            + torch.arange(-1, self.num_future + 2, device=self.device)[None, :]
        )
        reference = self.qpos[frames.clamp(0, len(self.qpos) - 1)]
        terms = derive_motion_actor_terms_torch(
            MotionActorObservationInput(
                reference_qpos=reference.contiguous(),
                robot_root_quat_wxyz=self.robot.data.root_link_quat_w.contiguous(),
                robot_root_angvel_local=self.robot.data.root_link_ang_vel_b.contiguous(),
                robot_dof_pos=self.robot.data.joint_pos[:, self.joints].contiguous(),
                robot_dof_vel=self.robot.data.joint_vel[:, self.joints].contiguous(),
                last_action=self.last_action,
                default_dof_pos=self.default_pos.expand(self.env.num_envs, -1).contiguous(),
                reference_dof_indices=self.reference_indices,
            ),
            fps=50,
            current_index=1,
            num_future_frames=self.num_future,
        )
        return TensorDict(
            {f"unified/{name}": value for name, value in terms.items()},
            batch_size=[self.env.num_envs],
        )

    def teacher_to_student(self, actions):
        target = actions * self.action.scale + self.action.offset
        return (target[:, self.from_action] - self.offset) / self.scale

    def student_to_teacher(self, actions):
        target = self.offset + self.scale * actions
        return (target[:, self.to_action] - self.action.offset) / self.action.scale
