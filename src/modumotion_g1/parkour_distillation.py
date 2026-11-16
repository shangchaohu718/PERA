"""Released InstinctMJ parkour teacher in its native physical environment.

This is a separate task adapter; existing shadowing/course jobs are untouched.
Terrain geometry and robot physics come from the official parkour factory.
Only dataset paths, terrain selection, deterministic route and batch size change.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from mjlab.sim.sim import MujocoCfg
from tensordict import TensorDict

from pera.paths import ROOT

RELEASE = ROOT / "checkpoints/instinct_parkour_main"
DATA = ROOT / "data/instinct_parkour_reference"
PUBLIC_SHA = "898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d"
TASKS = {
    "stairs_up": "pyramid_stairs",
    "stairs_down": "pyramid_stairs",
    "gaps": "square_gaps",
    "slope": "hf_pyramid_slope_inv",
    "rough": "perlin_rough",
    "obstacles": "boxes",
}


@dataclass
class PublishedMujocoCfg(MujocoCfg):
    """The current mjlab flag API no longer exposes the release's boolean."""

    multiccd: bool = False

    def apply(self, model):
        import mujoco

        super().apply(model)
        if hasattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD"):
            bit = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
            if self.multiccd:
                model.opt.disableflags &= ~bit
            else:
                model.opt.disableflags |= bit
        else:
            bit = int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
            if self.multiccd:
                model.opt.enableflags |= bit
            else:
                model.opt.enableflags &= ~bit


def restore_published_physics(cfg):
    """Translate published scalar settings, never execute YAML Python tags."""
    import yaml

    published = yaml.load((RELEASE / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    physics = published["sim"]
    cfg.sim.mujoco = PublishedMujocoCfg(**asdict(cfg.sim.mujoco))
    for name, value in physics["mujoco"].items():
        old = getattr(cfg.sim.mujoco, name)
        if isinstance(old, bool):
            value = value == "true"
        elif isinstance(old, tuple):
            value = tuple(float(v) for v in value)
        else:
            value = type(old)(value)
        setattr(cfg.sim.mujoco, name, value)
    cfg.sim.ls_parallel = physics["ls_parallel"] == "true"
    for name in ("nconmax", "njmax", "contact_sensor_maxmatch"):
        setattr(cfg.sim, name, int(physics[name]))
    original = published["scene"]["entities"]["robot"]["articulation"]["actuators"]
    current = cfg.scene.entities["robot"].articulation.actuators
    if len(original) != len(current):
        raise ValueError("Published actuator grouping differs")
    for old, actuator in zip(original, current, strict=True):
        if tuple(old["target_names_expr"]) != tuple(actuator.target_names_expr):
            raise ValueError("Published actuator order differs")
        for name in (
            "armature",
            "frictionloss",
            "stiffness",
            "damping",
            "effort_limit",
        ):
            setattr(actuator, name, float(old["base_cfg"][name]))
        for name in ("delay_min_lag", "delay_max_lag", "delay_update_period"):
            setattr(actuator, name, int(old[name]))
        actuator.delay_hold_prob = float(old["delay_hold_prob"])
        actuator.delay_per_env_phase = old["delay_per_env_phase"] == "true"


class ParkourDepthFeatures(torch.nn.Module):
    """The released parkour encoder uses a different block name than shadowing."""

    def __init__(self, teacher, normalizer, obs_format):
        super().__init__()
        from instinct_rl.utils.utils import get_obs_slice

        section, self.shape = get_obs_slice(obs_format, "depth_image")
        self.encoder = teacher.encoders._parallel_blocks["depth_encoder"]
        self.register_buffer("mean", normalizer._mean[:, section].clone())
        self.register_buffer("std", normalizer._std[:, section].clone())
        self.eps = normalizer.eps
        self.requires_grad_(False).eval()

    def forward(self, depth):
        normalized = (depth.flatten(1) - self.mean) / (self.std + self.eps)
        return self.encoder(normalized.reshape(-1, *self.shape))


class NativeStateReader:
    """Read physical coordinates without constructing shadowing/WM contacts."""

    def __init__(self, env):
        self.env = env

    def state(self):
        from instinct_mj.rl.supermimic.rotations import quat_to_mat

        idx, data = self.env.scene["robot"].indexing, self.env.sim.data
        q, v = data.qpos, data.qvel
        r = quat_to_mat(q[:, idx.free_joint_q_adr.long()[3:7]])
        fv = idx.free_joint_v_adr.long()
        return (
            q[:, idx.free_joint_q_adr.long()[:3]].clone(),
            r,
            v[:, fv[:3]].clone(),
            (r @ v[:, fv[3:6], None]).squeeze(-1),
            q[:, idx.joint_q_adr.long()].clone(),
            v[:, idx.joint_v_adr.long()].clone(),
        )


def build_cfg(task, num_envs, difficulty=(0.2, 0.5), seed=0):
    import instinct_mj.tasks.parkour.config.g1.g1_parkour_target_amp_cfg as source

    # The factory deep-copies this sensor. Restore the module global immediately;
    # other callers in this interpreter must not inherit our local paths.
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
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = 20.0
    generator = cfg.scene.terrain.terrain_generator
    name = TASKS[task]
    selected = copy.deepcopy(generator.sub_terrains[name])
    selected.proportion = 1.0
    generator.sub_terrains = {name: selected}
    generator.num_rows, generator.num_cols = 2, 2
    generator.seed = seed
    generator.difficulty_range = difficulty
    generator.curriculum = True
    cfg.curriculum = {}
    command = cfg.commands["base_velocity"]
    command.velocity_ranges = {name: command.velocity_ranges[name]}
    command.random_velocity_terrain = []
    # No resampling while measuring one route. Same command generator converts
    # the target to a velocity, using its released stiffness and limits.
    command.resampling_time_range = (60.0, 60.0)
    command.rel_standing_envs = 0.0
    command.debug_vis = False
    for sensor in cfg.scene.sensors:
        if hasattr(sensor, "debug_vis"):
            sensor.debug_vis = False
    return cfg


class ParkourTask:
    def __init__(self, task, num_envs=8, device="cuda:0", difficulty=(0.2, 0.5), seed=0, cfg=None):
        from modumotion_g1.training_dependencies import require_instinct_teacher_sources

        require_instinct_teacher_sources()
        import mujoco
        import yaml
        from instinct_mj.rl.supermimic.kinematics import G1Kinematics
        from instinct_mj.tasks.registry import load_env_cls, load_vecenv_cls
        from instinct_rl.modules import build_actor_critic, build_normalizer
        from instinct_rl.utils.utils import get_obs_slice, get_subobs_size

        from modumotion_g1.perceptive_observations import HoloMotionObservationAdapter
        from modumotion_g1.depth_features import tensor_module_digest

        self.task, self.N, self.device = task, num_envs, device
        self.cfg = build_cfg(task, num_envs, difficulty, seed) if cfg is None else cfg
        task_id = "Instinct-Parkour-Target-Amp-G1-Play-v0"
        self.env = load_env_cls(task_id)(cfg=self.cfg, device=device)
        self.vec = load_vecenv_cls(task_id)(self.env, policy_group="policy", critic_group="critic")
        formats = self.vec.get_obs_format()
        cfg = yaml.safe_load((RELEASE / "params/agent.yaml").read_text())
        policy_cfg = copy.deepcopy(cfg["policy"])
        cls = policy_cfg.pop("class_name")
        # No runner/algorithm/optimizer/storage is constructed for this teacher.
        self.teacher = build_actor_critic(
            cls, policy_cfg, formats, self.vec.num_actions, self.vec.num_rewards
        ).to(device)
        norm_cfg = dict(cfg["normalizers"]["policy"])
        self.normalizer = build_normalizer(
            get_subobs_size(formats["policy"]), norm_cfg.pop("class_name"), norm_cfg
        ).to(device)
        released = torch.load(RELEASE / "model_57000.pt", map_location=device, weights_only=True)
        self.teacher.load_state_dict(released["model_state_dict"], strict=True)
        self.normalizer.load_state_dict(released["policy_normalizer_state_dict"], strict=True)
        self.teacher.requires_grad_(False).eval()
        self.normalizer.requires_grad_(False).eval()
        self.depth = ParkourDepthFeatures(self.teacher, self.normalizer, formats["policy"])
        self.depth_slice, self.depth_shape = get_obs_slice(formats["policy"], "depth_image")
        if tuple(self.depth_shape) != (8, 18, 32):
            raise ValueError(f"Unexpected published parkour depth shape: {self.depth_shape}")
        self.encoder_sha = tensor_module_digest(self.depth)
        self.teacher_sha = tensor_module_digest(self.teacher)
        self.normalizer_sha = tensor_module_digest(self.normalizer)
        self.formats = formats
        self.robot = self.env.scene["robot"]
        self.reader = NativeStateReader(self.env)
        self.action = self.env.action_manager.get_term("joint_pos")
        if (
            list(self.action._target_names) != list(self.robot.joint_names)
            or self.action.cfg.clip is not None
        ):
            raise ValueError("Native action order or clipping needs explicit adaptation")
        self.adapter = HoloMotionObservationAdapter(
            self.robot.joint_names,
            G1Kinematics(device, self.robot.joint_names, ["pelvis"]),
        )
        self.last_action = torch.zeros(num_envs, 29, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.command = self.env.command_manager.get_term("base_velocity")
        self.cpu_model = self.env.sim.mj_model
        self.cpu_data = mujoco.MjData(self.cpu_model)
        mujoco.mj_forward(self.cpu_model, self.cpu_data)
        self.route_start = self.route_goal = None
        self.obs = None

    def contract(self):
        return {
            "context_dim": 133,
            "context_schema": {
                "kind": "instinct_parkour_frozen_depth_command_clock_v1",
                "depth_shape": list(self.depth_shape),
                "depth_features": 128,
                "additional_fields": [
                    "requested_vx_body",
                    "requested_vy_body",
                    "requested_yaw_rate",
                    "phase_sin",
                    "phase_cos",
                ],
                "normalization": "released_policy_normalizer",
                "clock_period_s": 0.6,
                "uses_teacher_action_or_future_state": False,
                "reference": "constant native standing posture and measured pelvis pose integrated by current requested velocity; no motion-file future",
                "frozen": True,
            },
            "encoder_sha256": self.encoder_sha,
        }

    def ground(self, xy):
        import mujoco

        out = []
        for x, y in np.asarray(xy):
            point = np.array([x, y, 30.0], dtype=np.float64)
            direction = np.array([0.0, 0.0, -1.0])
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.cpu_model, self.cpu_data, point, direction, None, 1, -1, geom_id
            )
            if distance < 0:
                raise ValueError(f"No static ground at {(x, y)}")
            out.append(30.0 - distance)
        return torch.tensor(out, device=self.device)

    def reset(self, seed, actor=None, env_ids=None):
        ids = torch.arange(self.N, device=self.device) if env_ids is None else env_ids
        self.env.reset(seed=seed if env_ids is None else None, env_ids=env_ids)
        origins = self.env.scene["terrain"].env_origins[ids].clone()
        start, goal = origins.clone(), origins.clone()
        if self.task == "stairs_up":
            start[:, 0] -= 3.35
            goal[:, 0] -= 0.15
        elif self.task == "stairs_down":
            goal[:, 0] += 3.35
        else:
            start[:, 0] -= 3.35
            goal[:, 0] += 3.35
        start[:, 2] = self.ground(start[:, :2].cpu().numpy())
        goal[:, 2] = self.ground(goal[:, :2].cpu().numpy())
        state = torch.zeros(len(ids), 13, device=self.device)
        state[:, :3] = start
        state[:, 2] += 0.9  # Native parkour torso root's released reset height.
        state[:, 3:7] = state.new_tensor([1, 0, 0, 0])
        state[:, 7:] = 0
        self.robot.write_root_state_to_sim(state, env_ids=ids)
        self.env.sim.forward()
        self.env.scene["camera"].reset(ids)
        self.env.scene.update(dt=0)
        self.env.sim.sense()
        self.command.pos_command_w[ids] = goal
        self.command.max_command_b[ids] = state.new_tensor([0.65, 0.0, 1.0])
        self.command.is_standing_env[ids] = False
        self.command._update_command()
        self.last_action[ids] = 0
        self.elapsed[ids] = 0
        if self.route_start is None:
            self.route_start, self.route_goal = start.clone(), goal.clone()
        else:
            self.route_start[ids], self.route_goal[ids] = start, goal
        if actor is not None:
            if env_ids is None:
                actor.reset_kv_cache(self.N, self.device)
                actor.clear_env_cache(None)
            else:
                actor.clear_env_cache(ids)
        # Reset histories to the actual route entry, before the first action.
        self.env.observation_manager.reset(env_ids=ids)
        observations = self.env.observation_manager.compute(update_history=True, env_ids=ids)
        packed = self.vec._pack_observations(observations)["policy"]
        if env_ids is None or self.obs is None:
            self.obs = packed
        else:
            self.obs[ids] = packed[ids]
        return self.obs

    def teacher_actions(self):
        return self.teacher.act_inference(self.normalizer(self.obs))

    def native_to_holo(self, action):
        return self.adapter.action_from_sim(action, self.action.offset, self.action.scale)

    def holo_to_native(self, action):
        return self.adapter.action_to_sim(action, self.action.offset, self.action.scale)

    def student_inputs(self):
        from modumotion_g1.perceptive_observations import _matquat, _yaw
        from modumotion_g1.velocity_walking import command_context

        state = self.reader.state()
        p, r, _, _ = self.adapter.pelvis_state(state)
        yaw = _yaw(_matquat(r))
        command = self.command.command.clone()
        t = (torch.arange(13, device=self.device) - 1) * self.env.step_dt
        angle = yaw[:, None] + command[:, 2:3] * t
        half = angle * 0.5
        quat = torch.stack(
            (half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()), -1
        )
        vx, vy = command[:, 0], command[:, 1]
        velocity = torch.stack(
            (yaw.cos() * vx - yaw.sin() * vy, yaw.sin() * vx + yaw.cos() * vy), -1
        )
        xy = p[:, None, :2] + velocity[:, None] * t[None, :, None]
        z = p[:, None, 2:3].expand(-1, 13, -1)
        a = self.adapter
        standing = self.robot.data.default_joint_pos[:, a.to_holo] * a.holo_joint_signs.to(p)
        reference = torch.cat((xy, z, quat, standing[:, None].expand(-1, 13, -1)), -1)
        terms = a.terms(state, self.last_action, reference, reference_sample_time=t)
        context = torch.cat(
            (
                self.depth(self.obs[:, self.depth_slice]),
                command_context(command, self.elapsed),
            ),
            -1,
        )
        return TensorDict(
            {f"unified/{k}": v for k, v in terms.items()}, batch_size=[self.N]
        ), context

    def step(self, native_action):
        if not torch.isfinite(native_action).all():
            raise FloatingPointError("Nonfinite parkour action")
        self.obs, _, done, _ = self.vec.step(native_action)
        self.last_action.copy_(self.native_to_holo(native_action))
        self.elapsed += self.env.step_dt
        return done.bool()

    def metrics(self):
        p, r, _, _ = self.adapter.pelvis_state(self.reader.state())
        tilt = torch.acos(r[:, 2, 2].clamp(-1, 1))
        distance = (p[:, :2] - self.route_goal[:, :2]).norm(dim=-1)
        return p, tilt, distance

    def assert_frozen(self):
        from modumotion_g1.depth_features import tensor_module_digest

        if (
            tensor_module_digest(self.teacher) != self.teacher_sha
            or tensor_module_digest(self.normalizer) != self.normalizer_sha
            or tensor_module_digest(self.depth) != self.encoder_sha
        ):
            raise RuntimeError("Frozen parkour teacher/normalizer/encoder changed")
