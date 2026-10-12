"""Play a reference motion directly in the native MJLab/MuJoCo viewer.

This is a kinematic data inspection tool: it does not load or execute Actor,
Dynamics, Critic, or Teacher networks, and it does not advance physics.  Every
displayed state is written directly from the reference motion file.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from unified_motion.api import MotionLibrary, MotionSampler, get_robot_contract
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.envs.mjlab_physics import make_k1_physics_cfg


def load_training_arguments(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arguments = payload.get("arguments")
    if payload.get("schema") != 1 or not isinstance(arguments, dict):
        raise ValueError("not a unified_motion reference-training checkpoint")
    return arguments


class ZeroPolicy:
    """Viewer protocol shim; actions are ignored by ReferencePlayback.step."""

    def __init__(self, action_dim: int, device: torch.device):
        self.action = torch.zeros((1, action_dim), device=device)

    def __call__(self, _: torch.Tensor) -> torch.Tensor:
        return self.action


class ReferencePlayback:
    """Expose direct reference-state playback through MJLab's viewer protocol."""

    def __init__(self, env: Any, bridge: K1MJLabBridge, sampler: MotionSampler):
        self._env = env
        self.bridge = bridge
        self.sampler = sampler
        self.robot = bridge.robot
        self.ids = torch.arange(1, device=bridge.device, dtype=torch.long)

    @property
    def unwrapped(self):
        return self._env

    @property
    def cfg(self):
        return self._env.cfg

    @property
    def device(self):
        return self._env.device

    @property
    def num_envs(self):
        return self._env.num_envs

    def get_observations(self) -> torch.Tensor:
        return torch.empty((1, 0), device=self.device)

    @torch.no_grad()
    def _write_current_frame(self) -> None:
        reference = self.sampler.reference_window()
        root_pose = torch.cat(
            (reference.root_pos_w[:, 0], reference.root_quat_wxyz[:, 0]), dim=-1)
        root_velocity = torch.cat(
            (reference.root_lin_vel_w[:, 0], reference.root_ang_vel_w[:, 0]), dim=-1)
        joint_pos = self.robot.to_simulator_order(
            reference.joint_pos[:, 0], self.bridge.joint_names)
        joint_vel = self.robot.to_simulator_order(
            reference.joint_vel[:, 0], self.bridge.joint_names)
        entity = self.bridge.entity
        entity.write_root_link_pose_to_sim(root_pose, env_ids=self.ids)
        entity.write_root_link_velocity_to_sim(root_velocity, env_ids=self.ids)
        entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=self.ids)
        self._env.sim.forward()
        self._env.sim.sense()

    def reset(self):
        self._env.reset()
        # Deliberately use the file's world pose.  No initial-state alignment or
        # random start is applied, so frame 0 is exactly the imported reference.
        self.sampler.reset(clip_ids=torch.zeros(1, device=self.device, dtype=torch.long))
        self._write_current_frame()
        return self.get_observations()

    def step(self, _: torch.Tensor):
        self.sampler.advance(1)
        self._write_current_frame()
        time_s = float(self.sampler.current_times_s()[0].item())
        return self.get_observations(), None, None, None, {"reference_time_s": time_s}

    def close(self) -> None:
        self._env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path,
                        help="checkpoint used only to locate the motion and K1 assets")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the interactive MJLab viewer requires CUDA")

    device = torch.device(args.device)
    train_args = load_training_arguments(args.checkpoint)
    robot = get_robot_contract("k1_22dof")
    motion_path = Path(train_args["motions"])
    library = (MotionLibrary.from_named_mjlab_file(
        motion_path, robot_id="k1_22dof", device=device)
        if motion_path.is_file() else MotionLibrary.from_project_folder(
            motion_path, robot_id="k1_22dof", device=device))
    if len(library.clip_names) != 1:
        raise ValueError(
            f"reference-only viewer requires one clip, found {len(library.clip_names)}")

    cfg = make_k1_physics_cfg(train_args["asset_root"], num_envs=1)
    cfg.seed = int(train_args["seed"])

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        bridge = K1MJLabBridge(env)
        sampler = MotionSampler(
            library, num_envs=1, step_dt=env.step_dt,
            future_offsets_s=(0.0, 0.04, 0.10, 0.20),
        )
        playback = ReferencePlayback(env, bridge, sampler)
        playback.reset()
        duration = float(library.clip_durations_s[0].item())
        steps = math.ceil(duration / float(env.step_dt))
        print(
            f"REFERENCE ONLY: {library.clip_names[0]} | {duration:.2f}s | "
            f"{steps} steps. No policy, no Dynamics, no physics rollout. "
            "SPACE pause, RIGHT single-step, close window to exit.",
            flush=True,
        )
        NativeMujocoViewer(
            playback, ZeroPolicy(len(robot.joint_names), device), frame_rate=60.0,
            enable_perturbations=False,
        ).run(num_steps=steps)
        print("REFERENCE_PLAYBACK_COMPLETE", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
