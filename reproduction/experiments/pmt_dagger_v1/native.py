"""PMT native Isaac Lab task; import only after AppLauncher has started."""

import json

import torch
from tensordict import TensorDict

from pera.paths import ROOT

ASSETS = ROOT / "data/pmt_bigmap_official_v1"
STEM = "ppofinetune_vision_teacher_stepping_stone_latent_anchor"
CHECKPOINT = ASSETS / "checkpoints/pretrained/ppoft_vision_bigmap_walkdance.pt"


class NativePMTTask:
    def __init__(self, num_envs=4, video=False, full_reference=False):
        from experiments.pmt_dagger_v1.prepare_assets import digest

        manifest = json.loads((ASSETS / "manifest.json").read_text())
        if manifest["status"] != "verified":
            raise ValueError("Official assets have not passed checksum verification")
        for row in manifest["files"]:
            if digest(ASSETS / row["path"]) != row["sha256"]:
                raise ValueError(f"Official asset changed: {row['path']}")
        from isaaclab.envs import ManagerBasedRLEnv
        from motion_tracking_rl import registry
        from pmt_tasks.builder import build_agent_cfg, build_env_cfg

        data = ASSETS / "assets/motions/terrain_mocaphouse/walk_dance1sub2start"
        overrides = [
            f"motion.motion_files={data / 'optimized'}",
            f"motion.raw_motion_files={data / 'raw'}",
            "network.base_policy_ckpt=null",
        ]
        cfg = build_env_cfg(
            "distill_stepping_stone_latent_anchor",
            overrides=[*overrides[:2], "network.teacher_ckpt=null"],
        )
        if full_reference:
            cfg.episode_length_s = 31.0
        cfg.scene.num_envs = num_envs
        cfg.sim.device = "cuda:0"
        if video:
            cfg.viewer.origin_type = "asset_root"
            cfg.viewer.asset_name = "robot"
            cfg.viewer.env_index = 0
            cfg.viewer.eye = (2.8, 2.8, 2.2)
            cfg.viewer.lookat = (0, 0, 0.6)
            cfg.viewer.resolution = (960, 720)
        self.env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array" if video else None)
        self.N, self.device = num_envs, "cuda:0"
        self.robot = self.env.scene["robot"]
        self.motion = self.env.command_manager.get_term("motion")
        self.raw = self.env.command_manager.get_term("student_motion")
        self.action = self.env.action_manager.get_term("joint_pos")
        agent = build_agent_cfg(STEM, overrides=overrides)
        obs, _ = self.env.reset(seed=7)
        options = agent.policy.to_dict()
        name = options.pop("class_name")
        registry.autoload()
        checkpoint = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)
        self.vision_keys = list(checkpoint["policy_metadata"]["obs_schema"]["available_obs"])
        self.teacher = registry.get_network(name)(
            obs=self.vision_observations(obs),
            obs_groups=agent.obs_groups,
            num_actions=29,
            **options,
        ).to(self.device)
        expected = checkpoint["policy_metadata"]
        current = self.teacher.get_checkpoint_metadata()
        if expected["obs_schema"] != current["obs_schema"]:
            raise ValueError("Native PMT observation schema differs from release")
        for key, value in expected["signature"].items():
            if current["signature"].get(key) != value:
                raise ValueError(f"Native PMT signature differs: {key}")
        self.teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.teacher.eval().requires_grad_(False)
        self.metadata = expected
        blind_path = CHECKPOINT.with_name("walkdance_bigmap_teacher.pt")
        blind_checkpoint = torch.load(blind_path, map_location=self.device, weights_only=False)
        blind_agent = build_agent_cfg("walk_dance_bigmap")
        blind_options = blind_agent.policy.to_dict()
        blind_name = blind_options.pop("class_name")
        self.blind_keys = list(blind_checkpoint["policy_metadata"]["obs_schema"]["available_obs"])
        self.blind = registry.get_network(blind_name)(
            obs=self.blind_observations(obs),
            obs_groups=blind_agent.obs_groups,
            num_actions=29,
            **blind_options,
        ).to(self.device)
        self.blind.load_state_dict(blind_checkpoint["model_state_dict"], strict=True)
        if (
            self.blind.get_checkpoint_metadata()["obs_schema"]
            != blind_checkpoint["policy_metadata"]["obs_schema"]
        ):
            raise ValueError("Blind teacher observation schema differs")
        self.blind.eval().requires_grad_(False)
        self.contract = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())
        names = self.contract["joint_names"]
        actual_names = self.robot.joint_names
        if set(names) != set(actual_names) or len(actual_names) != 29:
            raise ValueError("PMT/HoloMotion joint names differ")
        if self.robot.body_names[0] != "pelvis":
            raise ValueError("PMT root must be pelvis")
        self.to_holo = [actual_names.index(n) for n in names]
        self.to_native = [names.index(n) for n in actual_names]
        if list(self.action._joint_names) != list(actual_names):
            raise ValueError("PMT action order differs from articulation")
        self.holo_offset = torch.tensor(self.contract["action_offset"], device=self.device)
        self.holo_scale = torch.tensor(self.contract["action_scale"], device=self.device)
        self.default_pos = torch.tensor(self.contract["default_joint_pos"], device=self.device)
        self.last_action = torch.zeros(num_envs, 29, device=self.device)
        self.alignment_history = torch.zeros(num_envs, 10, 29, device=self.device)
        self.raw_qpos, lengths = [], []
        store = self.raw.data_store
        if self.raw.cfg.body_names[0] != "pelvis":
            raise ValueError("Raw reference body zero must be pelvis")
        for i, length in enumerate(store.motion_lengths.tolist()):
            self.raw_qpos.append(
                torch.cat(
                    [
                        store._stacked_body_pos_w[i, :length, 0],
                        store._stacked_body_quat_w[i, :length, 0],
                        store._stacked_joint_pos[i, :length, self.to_holo],
                    ],
                    dim=-1,
                ).float()
            )
            lengths.append(length)
        self.raw_qpos = torch.cat(self.raw_qpos).to(self.device)
        lengths = torch.tensor(lengths, device=self.device)
        self.starts = lengths.cumsum(0) - lengths
        self.ends = lengths.cumsum(0) - 1
        self.resampled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.completions = torch.zeros_like(self.resampled)
        self.final_positions = self.motion.robot_anchor_pos_w.clone()
        original = self.motion._resample_command

        def resample(ids):
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
            self.resampled[ids] = True
            sizes = self.motion.data_store.motion_lengths[self.motion.motion_ids[ids]]
            self.completions[ids] |= self.motion.frame_ids[ids] >= sizes
            self.final_positions[ids] = self.motion.robot_anchor_pos_w[ids]
            return original(ids)

        self.motion._resample_command = resample
        self.assets_manifest_sha256 = digest(ASSETS / "manifest.json")

    def vision_observations(self, obs):
        return TensorDict(
            {key: obs[key] for key in self.vision_keys},
            batch_size=[self.N],
        )

    def blind_observations(self, obs):
        aliases = {
            "policy": "teacher",
            "command_window": "teacher_command_window",
            "motion_anchor_delta_window": "teacher_motion_anchor_delta_window",
            "anchor_body_pose": "teacher_anchor_body_pose",
        }
        result = TensorDict(
            {key: obs[aliases.get(key, key)] for key in self.blind_keys},
            batch_size=[self.N],
        )
        if hasattr(self, "alignment_history"):
            # The blind teacher was trained with actions relative to q_opt.
            # Native paired env action histories are relative to q_raw.
            # Convert only the blind teacher's three action-history views.
            result["policy"] = result["policy"].clone()
            result["policy"][:, -290:] -= self.alignment_history.flatten(1)
            result["proprio"] = result["proprio"].clone()
            result["proprio"][:, -29:] -= self.alignment_history[:, -1]
            result["proprio_history"] = result["proprio_history"].clone()
            result["proprio_history"][:, :, -29:] -= self.alignment_history
        return result

    @torch.no_grad()
    def teacher_action(self, obs, controller="blind"):
        if controller == "vision":
            return self.teacher.act_inference(self.vision_observations(obs))
        action = self.blind.act_inference(self.blind_observations(obs))
        if not torch.all(torch.as_tensor(self.action._scale) == 1):
            raise ValueError("Official teacher alignment expects scale 1")
        # Same residual-reference compensation as PMT align_teacher_actions:
        # q_opt + a_teacher == q_raw + a_aligned.
        return action + self.motion.joint_pos - self.raw.joint_pos

    @torch.no_grad()
    def context(self, obs):
        # Height-map attention is conditioned on frozen raw-command and
        # proprioception encoders. No optimized reference/action head is used.
        return self.teacher._compute_actor_context(
            self.vision_observations(obs), store_aux_outputs=False
        )[-1].detach()

    def student_inputs(self, obs):
        from modumotion_g1.perceptive_observations import (
            PerceptiveActorObservationInput,
            build_perceptive_actor_terms,
            sample_holomotion_reference,
        )

        ids = self.raw.motion_ids
        frame = self.starts[ids] + self.raw.frame_ids
        reference, times = sample_holomotion_reference(
            self.raw_qpos,
            frame,
            torch.zeros_like(frame, dtype=torch.float32),
            self.starts[ids],
            self.ends[ids],
            reference_fps=50,
        )
        state = self.robot.data
        terms = build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=reference,
                robot_root_quat_wxyz=state.root_quat_w,
                robot_root_angvel_local=state.root_ang_vel_b,
                robot_dof_pos=state.joint_pos[:, self.to_holo],
                robot_dof_vel=state.joint_vel[:, self.to_holo],
                last_action=self.last_action,
                default_dof_pos=self.default_pos,
                reference_sample_time=times,
            )
        )
        return TensorDict(
            {f"unified/{k}": v for k, v in terms.items()},
            batch_size=[self.N],
        ), self.context(obs)

    def action_to_holo(self, value):
        absolute = self.action._offset + self.action._scale * value
        return (absolute[:, self.to_holo] - self.holo_offset) / self.holo_scale

    def action_to_native(self, value):
        absolute = self.holo_offset + self.holo_scale * value
        return (absolute[:, self.to_native] - self.action._offset) / self.action._scale

    def reset(self, seed, actor=None, from_start=False):
        # Only initial sampling differs between training and phase-zero eval.
        self.motion.cfg.force_start_frame_zero = from_start
        self.motion.cfg.default_reset_joint_prob = 0.0 if from_start else 0.2
        self.resampled.zero_()
        self.completions.zero_()
        draw = self.motion._draw_samples
        if from_start:
            from pmt_tasks.mdp.commands.multi_motion_command import (
                SamplingResult,
            )

            def sample_from_start(ids):
                return SamplingResult(
                    motion_ids=ids % len(self.starts),
                    frame_ids=torch.zeros_like(ids),
                )

            self.motion._draw_samples = sample_from_start
        try:
            obs, _ = self.env.reset(seed=seed)
        finally:
            self.motion._draw_samples = draw
        self.last_action.zero_()
        self.alignment_history.zero_()
        if actor is not None:
            actor.reset_kv_cache(self.N, self.device)
            actor.clear_env_cache(None)
        return obs

    def step(self, action):
        self.resampled.zero_()
        self.completions.zero_()
        # Offset can change during command.compute(), so convert beforehand.
        holo_action = self.action_to_holo(action)
        delta = (self.motion.joint_pos - self.raw.joint_pos).clone()
        obs, reward, terminated, truncated, extras = self.env.step(action)
        self.alignment_history = torch.cat(
            [
                self.alignment_history[:, 1:],
                delta[:, None],
            ],
            dim=1,
        )
        self.alignment_history[self.resampled] = 0
        self.last_action.copy_(holo_action)
        self.last_action[self.resampled] = 0
        return obs, reward, terminated, truncated, extras

    def close(self):
        self.env.close()
