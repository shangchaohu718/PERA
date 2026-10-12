"""Interactively compare a trained policy with its reference-motion ghost.

The simulated robot uses the frozen deterministic Actor.  A translucent green
robot is drawn from the exact reference pose at the same reference time.  On a
fall the final state is held briefly, then the episode restarts from frame 0.
No weights, training metrics, or checkpoints are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unified_motion.api import (
    DynamicsModel, MotionLibrary, MotionSampler, MotionTrackingReward,
    MotionTrackingRewardConfig, MotionTrackingTask, build_policy,
    get_robot_contract,
)
from unified_motion.algorithms.sequence_training import PolicyInput
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.envs.mjlab_physics import K1PhysicsBridge, make_k1_physics_cfg
from unified_motion.models.physics import DynamicsMemory, PhysicsMLPActorCritic


class FrozenPolicy:
    """Deterministic checkpoint policy with explicit Dynamics memory."""

    def __init__(self, task: MotionTrackingTask, dynamics: DynamicsModel, policy: Any):
        self.task = task
        self.dynamics = dynamics
        self.policy = policy
        self.memory: DynamicsMemory | None = None
        self.cache = None
        self.next_memory: DynamicsMemory | None = None
        self.prediction = None

    def reset(self) -> None:
        self.memory = None
        self.cache = None
        self.next_memory = None
        self.prediction = None
        self.dynamics.invalidate_memories()
        if hasattr(self.policy, "invalidate_caches"):
            self.policy.invalidate_caches()

    @torch.no_grad()
    def __call__(self, _: torch.Tensor) -> torch.Tensor:
        frame = self.task.read()
        starts = self.task.episode_start.clone()
        encoding, self.next_memory = self.dynamics.step(
            frame, episode_start=starts, env_ids=self.task.env_ids,
            memory=self.memory,
        )
        privileged = self.task.privileged()
        actor_encoding = (self.task.policy_dynamics(encoding, privileged)
                          if hasattr(self.task, "policy_dynamics") else encoding)
        inputs = PolicyInput(
            frame, self.task.goal(), actor_encoding.context,
            actor_encoding.latent, actor_encoding.prediction_error,
            starts, privileged,
        )
        if isinstance(self.policy, PhysicsMLPActorCritic):
            output = self.policy(
                inputs.physics, inputs.dynamics(), inputs.goal,
                privileged=inputs.privileged,
            )
            action = output.mean
        else:
            _, action, _, self.cache = self.policy.step(
                self.task.robot_id, inputs.physics, inputs.dynamics(), inputs.goal,
                env_ids=self.task.env_ids, episode_start=starts,
                cache=self.cache, privileged=privileged, deterministic=True,
            )
        self.prediction = self.dynamics.predict_next(
            frame, action, encoding.latent,
            physics_encoding=encoding.physics_encoding(),
        )
        return action

    def commit_step(self) -> None:
        if self.next_memory is None or self.prediction is None:
            raise RuntimeError("policy action must be computed before stepping")
        self.memory = self.dynamics.remember_prediction(
            self.next_memory, self.prediction)


class ViewerTaskAdapter:
    """Expose MotionTrackingTask through mjlab's interactive viewer protocol."""

    def __init__(self, env: Any, task: MotionTrackingTask, frozen: FrozenPolicy,
                 hold_steps: int, clip_index: int):
        self._env = env
        self.task = task
        self.frozen = frozen
        self.hold_steps = hold_steps
        self.clip_index = clip_index
        self.hold_remaining = 0
        self.episode_step = 0

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
        return torch.empty((self.num_envs, 0), device=self.device)

    def reset(self):
        self.task.bridge.reset()
        data = self.task.bridge.entity.data
        clip_ids = torch.full(
            (self.task.num_envs,), self.clip_index,
            device=self.task.device, dtype=torch.long)
        self.task.sampler.reset(
            clip_ids=clip_ids,
            current_root_pos_w=data.root_link_pos_w,
            current_root_quat_wxyz=data.root_link_quat_w,
        )
        if self.task.reference_state_init:
            self.task._initialize_from_reference(None)
        self.frozen.reset()
        self.hold_remaining = 0
        self.episode_step = 0
        return self.get_observations()

    def step(self, action: torch.Tensor):
        if self.hold_remaining:
            self.hold_remaining -= 1
            if self.hold_remaining == 0:
                self.reset()
            return self.get_observations(), None, None, None, {}
        transition = self.task.step(action)
        self.frozen.commit_step()
        self.episode_step += 1
        done = transition.terminated | transition.truncated
        if done.any():
            metrics = transition.metrics or {}
            summary = {
                name: round(float(value.mean().item()), 4)
                for name, value in metrics.items()
                if value.ndim == 1
            }
            reason = "fallen" if transition.terminated.any() else "timeout"
            print(
                f"VIEWER_TERMINAL reason={reason} step={self.episode_step} "
                f"time={self.episode_step*self.task.sampler.step_dt:.2f}s metrics={summary}",
                flush=True,
            )
            self.hold_remaining = self.hold_steps
        return self.get_observations(), transition.reward, done, {}, {}

    def close(self) -> None:
        self._env.close()


def install_reference_ghost(env: Any, task: MotionTrackingTask) -> None:
    """Attach a green reference robot to mjlab's native debug visualizer."""
    original = getattr(env, "update_visualizers", None)
    robot = task.robot
    bridge = task.bridge

    def update_visualizers(visualizer) -> None:
        if original is not None:
            original(visualizer)
        reference = task.sampler.reference_window()
        root = reference.root_pos_w[0, 0]
        quat = reference.root_quat_wxyz[0, 0]
        joint = robot.to_simulator_order(
            reference.joint_pos[0, 0], bridge.joint_names)
        qpos = torch.cat((root, quat, joint)).detach().cpu().numpy()
        model = visualizer.mj_model
        rgba = model.geom_rgba.copy()
        try:
            model.geom_rgba[:, :3] = np.array((0.15, 1.0, 0.25))
            model.geom_rgba[:, 3] = 0.32
            visualizer.add_ghost_mesh(qpos, model, label="REFERENCE")
        finally:
            model.geom_rgba[:] = rgba

    env.update_visualizers = update_visualizers


def load_checkpoint(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("schema") != 1 or not isinstance(payload.get("arguments"), dict):
        raise ValueError("not a unified_motion reference-training checkpoint")
    args = payload["arguments"]
    robot = get_robot_contract("k1_22dof")
    full = args["adaptation"] == "full"
    return payload, args, robot, full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--clip-index", type=int, default=0,
                        help="固定播放MotionLibrary中的某个参考片段")
    args = parser.parse_args()
    if args.hold_seconds < 0 or args.clip_index < 0:
        parser.error("hold-seconds和clip-index必须非负")
    if not torch.cuda.is_available():
        raise RuntimeError("the interactive MJLab viewer requires CUDA")

    device = torch.device(args.device)
    payload, train_args, robot, full = load_checkpoint(args.checkpoint, device)
    motion_path = Path(train_args["motions"])
    library = (MotionLibrary.from_named_mjlab_file(
        motion_path, robot_id="k1_22dof", device=device)
        if motion_path.is_file() else MotionLibrary.from_project_folder(
            motion_path, robot_id="k1_22dof", device=device))
    if args.clip_index >= len(library.clip_names):
        parser.error(
            f"clip-index越界：动作库有{len(library.clip_names)}段，"
            f"收到{args.clip_index}")
    cfg = make_k1_physics_cfg(train_args["asset_root"], num_envs=1)
    cfg.seed = int(train_args["seed"])
    cfg.episode_length_s = max(
        float(cfg.episode_length_s),
        float(library.clip_durations_s.max().item()) + .02,
    )

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        bridge = K1MJLabBridge(env)
        physics = K1PhysicsBridge(bridge)
        sampler = MotionSampler(
            library, num_envs=1, step_dt=env.step_dt,
            future_offsets_s=(0., .04, .10, .20),
        )
        reward_cfg = MotionTrackingRewardConfig(
            body_pos_weight=float(train_args["body_pos_weight"]),
            body_rot_weight=float(train_args["body_rot_weight"]),
            body_lin_vel_weight=float(train_args["body_lin_vel_weight"]),
        )
        task = MotionTrackingTask(
            physics, sampler, MotionTrackingReward(robot, reward_cfg),
            actor_observation="deployable" if full else "full",
            privileged_learning=full,
            extended_contact_supervision=bool(
                train_args.get("extended_contact_supervision", False)),
            reference_state_init=bool(train_args.get("reference_state_init", False)),
            random_reference_start=False,
        )
        latent_width = int(train_args.get("latent_width", 64))
        dynamics = DynamicsModel(
            robot, latent_width=latent_width,
            system_id_dim=task.privileged_dim if full else 0,
            teacher_student=full,
            prediction_horizons=tuple(train_args.get("prediction_horizons", (1,))),
        ).to(device)
        policy = build_policy(
            {"k1_22dof": robot}, backbone=train_args["backbone"],
            d_model=int(train_args["d_model"]),
            hidden_dim=int(train_args["hidden_dim"]),
            n_layers=int(train_args["layers"]),
            context_len=int(train_args["context_len"]),
            num_experts=int(train_args["experts"]),
            init_std=float(train_args["init_std"]),
            latent_width=latent_width,
            recurrent_hidden_dim=train_args["recurrent_hidden_dim"],
            max_replay_steps=int(train_args["max_replay_steps"]),
            privileged_dims=({"k1_22dof": task.privileged_dim} if full else None),
        ).to(device)
        dynamics.load_state_dict(payload["dynamics"]["model"], strict=True)
        policy.load_state_dict(payload["ppo"]["model"], strict=True)
        dynamics.eval()
        policy.eval()
        frozen = FrozenPolicy(task, dynamics, policy)
        adapter = ViewerTaskAdapter(
            env, task, frozen,
            hold_steps=round(args.hold_seconds / float(env.step_dt)),
            clip_index=args.clip_index,
        )
        adapter.reset()
        install_reference_ghost(env, task)
        print(
            "Viewer: normal robot = checkpoint policy; translucent GREEN = reference. "
            f"Clip={library.clip_names[args.clip_index]}. "
            "SPACE pause, RIGHT single-step, ENTER reset, close window to exit.",
            flush=True,
        )
        NativeMujocoViewer(adapter, frozen, frame_rate=60.0).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
