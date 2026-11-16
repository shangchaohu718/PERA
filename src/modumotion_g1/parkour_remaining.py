"""Teacher audit for the remaining tasks, with native and traversal contracts.

Native playback keeps the source reset/target sampling. Traversal explicitly
requests ascending stairs or crossing the whole field and retains that scope in
the report. Native termination, the former pelvis rule and numeric overflows are
reported separately; one cannot be silently substituted for another.
"""

from __future__ import annotations

import copy

import torch

from modumotion_g1.parkour_distillation import (
    DATA,
    TASKS,
    ParkourTask,
    restore_published_physics,
)

SKILL_DIFFICULTY = {
    "stairs_up": (0.0, 0.15),
    "gaps": (0.2, 0.5),
    "obstacles": (0.0, 0.15),
}


def configure_hfield_capacity(capacity=256):
    """Enlarge only this process's Warp hfield candidate workspace.

    The native kernel collects candidate contacts before reducing them. Its
    fixed 50-slot workspace can overflow at stair edges before the task ends.
    No dependency file, physical model, contact reduction, geometry or gain is
    changed. Must run before constructing the first simulation in this process.
    """
    import warp as wp
    from mujoco_warp._src import collision_convex

    if capacity < 50:
        raise ValueError("Do not reduce the native contact candidate capacity")
    collision_convex.MJ_MAXCONPAIR = capacity
    collision_convex.vec_maxconpair = wp.types.vector(length=capacity, dtype=float)
    collision_convex.mat_maxconpair = wp.types.matrix(shape=(capacity, 3), dtype=float)
    return {
        "native_capacity": 50,
        "effective_capacity": capacity,
        "scope": "process_local_runtime_constants_only",
        "physical_model_modified": False,
    }


def native_cfg(task, num_envs, difficulty=(0.2, 0.5), seed=0, ascent_from_valley=True):
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
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = num_envs, seed, False
    # The source resets at each terrain's center. A positive pyramid therefore
    # starts at the top (descent). Its official inverted variant starts in the
    # valley and makes native playback an actual ascent, without teleporting.
    name = "pyramid_stairs_inv" if task == "stairs_up" and ascent_from_valley else TASKS[task]
    generator = cfg.scene.terrain.terrain_generator
    sub = copy.deepcopy(generator.sub_terrains[name])
    sub.proportion = 1.0
    generator.sub_terrains = {name: sub}
    generator.num_rows, generator.num_cols = 2, 2
    generator.seed, generator.difficulty_range = seed, difficulty
    cfg.curriculum = {}
    command = cfg.commands["base_velocity"]
    command.velocity_ranges = {name: command.velocity_ranges[name]}
    command.random_velocity_terrain = []
    command.debug_vis = False
    for sensor in cfg.scene.sensors:
        if hasattr(sensor, "debug_vis"):
            sensor.debug_vis = False
    return cfg


class RemainingParkourTask(ParkourTask):
    def __init__(
        self,
        task,
        num_envs=1,
        device="cuda:0",
        difficulty=None,
        seed=0,
        route_mode="skill",
        hfield_capacity=256,
    ):
        if route_mode not in ("native", "skill", "traverse"):
            raise ValueError("Choose native Play, a valid-target skill, or traversal")
        self.route_mode = route_mode
        difficulty = SKILL_DIFFICULTY[task] if difficulty is None else difficulty
        self.difficulty = tuple(difficulty)
        self.collision_workspace = configure_hfield_capacity(hfield_capacity)
        cfg = native_cfg(
            task,
            num_envs,
            difficulty,
            seed,
            ascent_from_valley=route_mode != "traverse",
        )
        if route_mode == "skill":
            # One locomotion request per episode. Keep source root/joint reset,
            # source 10s horizon and source speed sampling (0.45--0.8m/s).
            cfg.commands["base_velocity"].resampling_time_range = (60.0, 60.0)
            cfg.commands["base_velocity"].rel_standing_envs = 0.0
        if route_mode == "traverse":
            cfg.episode_length_s = 20.0
            cfg.commands["base_velocity"].resampling_time_range = (60.0, 60.0)
            cfg.commands["base_velocity"].rel_standing_envs = 0.0
        super().__init__(task, num_envs, device, difficulty, seed, cfg=cfg)

    def reset(self, seed, actor=None, env_ids=None):
        if self.route_mode == "traverse":
            return super().reset(seed, actor, env_ids)
        ids = torch.arange(self.N, device=self.device) if env_ids is None else env_ids
        observations, _ = self.env.reset(seed=seed if env_ids is None else None, env_ids=env_ids)
        packed = self.vec._pack_observations(observations)["policy"]
        if self.route_mode == "skill" and self.task == "obstacles":
            terrain = self.env.scene["terrain"]
            targets = self.command.valid_targets[
                terrain.terrain_levels[ids], terrain.terrain_types[ids]
            ]
            origins = terrain.env_origins[ids]
            relative = targets[..., :2] - origins[:, None, :2]
            valid = (relative[..., 0] >= 2.0) & (relative[..., 1].abs() <= 2.0)
            if not valid.any(-1).all():
                raise ValueError("No source-sampled forward flat-patch goal in this field")
            ideal = relative.new_tensor([3.2, 0.0])
            score = (relative - ideal).square().sum(-1).masked_fill(~valid, float("inf"))
            selected = targets[torch.arange(len(ids), device=self.device), score.argmin(-1)]
            self.command.pos_command_w[ids] = selected
            self.command._update_command()
            self.env.observation_manager.reset(env_ids=ids)
            observations = self.env.observation_manager.compute(update_history=True, env_ids=ids)
            packed = self.vec._pack_observations(observations)["policy"]
        if env_ids is None or self.obs is None:
            self.obs = packed
        else:
            self.obs[ids] = packed[ids]
        self.last_action[ids], self.elapsed[ids] = 0, 0
        p = self.reader.state()[0]
        if self.route_start is None:
            self.route_start, self.route_goal = (
                p.clone(),
                self.command.pos_command_w.clone(),
            )
        else:
            self.route_start[ids] = p[ids]
            self.route_goal[ids] = self.command.pos_command_w[ids]
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
        if self.route_mode != "traverse":
            self.route_goal.copy_(self.command.pos_command_w)
        return done
