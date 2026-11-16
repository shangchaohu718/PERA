"""Four remaining official parkour configurations; independent scene plugins."""

import copy
from pathlib import Path

import numpy as np
import torch

from modumotion_g1.parkour_distillation import (
    DATA,
    ParkourTask,
    restore_published_physics,
)
from modumotion_g1.parkour_remaining import configure_hfield_capacity
from modumotion_g1.parkour_remaining_rollout import actor_action, collect, endings

TASKS = {
    "stairs_up_high": "pyramid_stairs_inv_high",
    "stairs_down_high": "pyramid_stairs_high",
    "mesh_boxes": "mesh_boxes",
    "rough_stand": "perlin_rough_stand",
}
DIFFICULTY = {
    "stairs_up_high": (0.5, 0.75),
    "stairs_down_high": (0.5, 0.75),
    "mesh_boxes": (0.2, 0.5),
    "rough_stand": (0.2, 0.5),
}


def build_cfg(task, num_envs):
    import instinct_mj.tasks.parkour.config.g1.g1_parkour_target_amp_cfg as source

    original = source.motion_reference_cfg
    local = copy.deepcopy(original)
    for motion in local.motion_buffers.values():
        motion.path = str(DATA)
        motion.filtered_motion_selection_filepath = str(DATA / "parkour_motion_without_run.yaml")
    source.motion_reference_cfg = local
    try:
        cfg = source.instinct_g1_parkour_amp_final_cfg(play=True, shoe=True)
    finally:
        source.motion_reference_cfg = original
    restore_published_physics(cfg)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = num_envs, 0, False
    generator = cfg.scene.terrain.terrain_generator
    name = TASKS[task]
    sub = copy.deepcopy(generator.sub_terrains[name])
    sub.proportion = 1.0
    generator.sub_terrains = {name: sub}
    generator.num_rows = generator.num_cols = 2
    generator.seed, generator.difficulty_range = 0, DIFFICULTY[task]
    cfg.curriculum = {}
    command = cfg.commands["base_velocity"]
    command.velocity_ranges = {name: command.velocity_ranges[name]}
    command.random_velocity_terrain = [name] if task == "rough_stand" else []
    command.resampling_time_range = (60, 60)
    command.rel_standing_envs, command.debug_vis = 0.0, False
    for sensor in cfg.scene.sensors:
        if hasattr(sensor, "debug_vis"):
            sensor.debug_vis = False
    return cfg


class ExtraParkourTask(ParkourTask):
    def __init__(self, task, num_envs=1, device="cuda:0"):
        self.difficulty = DIFFICULTY[task]
        self.collision_workspace = configure_hfield_capacity(256)
        super().__init__(task, num_envs, device, self.difficulty, cfg=build_cfg(task, num_envs))

    def reset(self, seed, actor=None, env_ids=None):
        ids = torch.arange(self.N, device=self.device) if env_ids is None else env_ids
        observations, _ = self.env.reset(seed=seed if env_ids is None else None, env_ids=env_ids)
        if self.task == "mesh_boxes":
            terrain = self.env.scene["terrain"]
            targets = self.command.valid_targets[
                terrain.terrain_levels[ids], terrain.terrain_types[ids]
            ]
            relative = targets[..., :2] - terrain.env_origins[ids, None, :2]
            valid = (relative[..., 0] >= 2.0) & (relative[..., 1].abs() <= 2.0)
            if not valid.any(-1).all():
                raise ValueError("No original valid forward target in the mesh-box field")
            score = (relative - relative.new_tensor([3.2, 0])).square().sum(-1)
            selected = score.masked_fill(~valid, float("inf")).argmin(-1)
            self.command.pos_command_w[ids] = targets[
                torch.arange(len(ids), device=self.device), selected
            ]
            self.command._update_command()
            self.env.observation_manager.reset(env_ids=ids)
            observations = self.env.observation_manager.compute(update_history=True, env_ids=ids)
        packed = self.vec._pack_observations(observations)["policy"]
        if env_ids is None or self.obs is None:
            self.obs = packed
        else:
            self.obs[ids] = packed[ids]
        self.last_action[ids], self.elapsed[ids] = 0, 0
        p = self.metrics()[0] if self.route_goal is not None else self.reader.state()[0]
        if self.route_start is None:
            self.route_start, self.route_goal = (
                p.clone(),
                self.command.pos_command_w.clone(),
            )
        else:
            self.route_start[ids], self.route_goal[ids] = (
                p[ids],
                self.command.pos_command_w[ids],
            )
        if actor is not None:
            if env_ids is None:
                actor.reset_kv_cache(self.N, self.device)
                actor.clear_env_cache(None)
            else:
                actor.clear_env_cache(ids)
        return self.obs

    def overflow_flags(self):
        return self.env.sim.data.overflow.clone().to(torch.int64)

    def step(self, action):
        done = super().step(action)
        self.route_goal.copy_(self.command.pos_command_w)
        return done


def teacher_decision(gate, *, task, checkpoint_sha256, authorization=None):
    if (
        gate.get("status") != "completed"
        or gate["arguments"]["task"] != task
        or gate["teacher_checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("Wrong teacher provenance")
    evaluation = gate["evaluation"]
    episodes = evaluation["episodes"]
    successes = sum(e["success"] for e in episodes)
    allowed = bool(
        authorization
        and authorization.get("user_request") == "把这些都训练了吧"
        and task in authorization.get("tasks", [])
    )
    if (
        len(episodes) < 8
        or not successes
        or not all(e["physics_valid"] for e in episodes)
        or (not evaluation["acceptance_passed"] and not allowed)
    ):
        raise ValueError(
            "Teacher has no verified successful episodes, invalid physics or no authorization"
        )
    return {
        "training_allowed": True,
        "teacher_acceptance_passed": evaluation["acceptance_passed"],
        "teacher_successes": successes,
        "teacher_episodes": len(episodes),
        "authorization_used": not evaluation["acceptance_passed"],
    }


@torch.no_grad()
def rollout(task, actor=None, *, seed, seconds=20, features=None, beta=0, recording=None):
    if features is not None:
        return collect(task, actor, features, seed, seconds, beta)
    if task.N != 1:
        raise ValueError("Independent evaluation requires one physical world")
    episodes, weights, poses, depths, commands = [], [], [], [], []
    total_steps = 0
    for index in range(4):
        task.reset(seed + index, actor)
        start = task.metrics()[0][0].clone()
        goal_ground = float(task.ground(task.route_goal[:, :2].cpu().numpy())[0])
        near, dwell, flags = 0, False, 0
        max_drift = max_tilt = 0.0
        for step in range(task.env.max_episode_length):
            if task.overflow_flags().any():
                raise RuntimeError("Overflow before a valid evaluation action")
            if actor is None:
                action = task.teacher_actions()
            else:
                mu, _ = actor_action(task, actor)
                action = task.holo_to_native(mu)
                block = actor.actor_module.layers[0]
                weights.append(
                    float(block.last_group_weights[..., -1].mean()) if block.active_domains else 0
                )
            if recording is not None and index == 0 and step % 2 == 0:
                poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
                depths.append(
                    task.obs[0, task.depth_slice]
                    .reshape(*task.depth_shape)[-1]
                    .cpu()
                    .numpy()
                    .copy()
                )
                commands.append(task.command.command[0].cpu().numpy().copy())
            done = task.step(action)
            p, tilt, distance, fired, failed, overflow = endings(task)
            flags |= int(overflow[0])
            max_drift = max(max_drift, float((p[0, :2] - start[:2]).norm()))
            max_tilt = max(max_tilt, float(tilt[0]))
            valid_height = goal_ground + 0.55 <= float(p[0, 2]) <= goal_ground + 1.1
            near = near + 1 if distance[0] <= 0.4 and valid_height and not failed[0] else 0
            dwell |= near >= round(1 / task.env.step_dt)
            if done[0] or failed[0] or overflow[0]:
                break
        total_steps += step + 1
        survived = bool(fired["time_out"][0] and not failed[0] and not flags)
        displacement = float((p[0, :2] - start[:2]).norm())
        height_change = float(p[0, 2] - start[2])
        if task.task == "rough_stand":
            ground = float(task.ground(p[:, :2].cpu().numpy())[0])
            success = (
                survived
                and max_drift <= 0.35
                and max_tilt <= 0.5
                and 0.55 <= float(p[0, 2]) - ground <= 1.1
            )
        else:
            success = (
                survived and dwell and displacement >= (1.5 if task.task == "mesh_boxes" else 2.0)
            )
            if task.task == "stairs_up_high":
                success &= height_change >= 0.15
            if task.task == "stairs_down_high":
                success &= height_change <= -0.15
        episodes.append(
            {
                "seed": seed + index,
                "success": bool(success),
                "physics_valid": flags == 0,
                "overflow_flags": flags,
                "physical_survival_to_horizon": survived,
                "physical_failure": bool(failed[0]),
                "elapsed_s": float(task.elapsed[0]),
                "horizontal_displacement_m": displacement,
                "height_change_m": height_change,
                "max_drift_m": max_drift,
                "max_pelvis_tilt_rad": max_tilt,
                "goal_dwell_1s": dwell,
                "fired": {k: bool(v[0]) for k, v in fired.items()},
            }
        )
    if recording is not None:
        import mujoco

        recording = Path(recording)
        recording.mkdir(parents=True, exist_ok=False)
        mujoco.mj_saveModel(task.cpu_model, str(recording / "scene.mjb"))
        np.savez_compressed(
            recording / "physics.npz",
            qpos=np.asarray(poses),
            depth=np.asarray(depths),
            commands=np.asarray(commands),
            fps=25,
        )
    successes = sum(e["success"] for e in episodes)
    return {
        "task": task.task,
        "terrain": TASKS[task.task],
        "difficulty": task.difficulty,
        "episodes": episodes,
        "successes": successes,
        "num_episodes": len(episodes),
        "acceptance_passed": successes / len(episodes) >= 0.9
        and all(e["physics_valid"] for e in episodes),
        "teacher_controlled_env_steps": total_steps if actor is None else 0,
        "mean_new_group_weight": float(np.mean(weights)) if weights else None,
        "collision_workspace": task.collision_workspace,
    }, None
