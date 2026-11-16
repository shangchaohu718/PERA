"""Native InstinctMJ teacher and exact frozen-public feature DAgger.

This experiment uses the official task reference for both teacher and student.
It does not use the separate Raw/TCRS reference contract of the WM experiment.
"""

import torch
from instinct_mj.rl.supermimic.kinematics import G1Kinematics
from instinct_mj.rl.supermimic.motion_lib import MotionLib
from instinct_mj.rl.supermimic.sim import SimReader
from instinct_mj.scripts.supermimic.evaluate import (
    FAILURE_TERMS,
    PLAY_TASK,
    SUCCESS_TERM,
    EvalArgs,
    build_eval_cfg,
)
from instinct_mj.tasks.registry import (
    load_env_cls,
    load_instinct_rl_cfg,
    load_runner_cls,
    load_vecenv_cls,
)
from instinct_rl.utils.utils import get_obs_slice
from tensordict import TensorDict

from modumotion_g1.domain_moe import set_domain_context
from modumotion_g1.perceptive_observations import (
    HoloMotionObservationAdapter,
    flatten_actor_terms,
)
from modumotion_g1.depth_features import FrozenDepthFeatures


class NativeTeacherTask:
    """Official Play physics, noise, reference, and strict endings."""

    def __init__(self, folder, checkpoint, num_envs, device, video=False):
        args = EvalArgs(
            motion_folder=str(folder),
            checkpoint=str(checkpoint),
            num_envs=num_envs,
            min_episodes_per_motion=1,
            device=device,
            seed=7,
        )
        cfg, self.motion_files = build_eval_cfg(args, num_envs, render=video)
        if len(self.motion_files) != 1:
            raise ValueError("Each task expert requires one official motion")
        cfg.auto_reset = False
        self.device, self.N = device, num_envs
        self.env = load_env_cls(PLAY_TASK)(
            cfg=cfg,
            device=device,
            render_mode="rgb_array" if video else None,
        )
        self.vec = load_vecenv_cls(PLAY_TASK)(
            self.env, policy_group="policy", critic_group="critic"
        )
        agent_cfg = load_instinct_rl_cfg(PLAY_TASK)
        runner = load_runner_cls(PLAY_TASK)(
            self.vec, agent_cfg.to_dict(), log_dir=None, device=device
        )
        released = torch.load(checkpoint, map_location=device, weights_only=True)
        runner.ac.load_state_dict(released["model_state_dict"], strict=True)
        for group, normalizer in runner.normalizers.items():
            normalizer.load_state_dict(released[f"{group}_normalizer_state_dict"], strict=True)
        self.teacher_actor, self.teacher_norm = runner.ac, runner.pol_norm
        self.teacher_actor.requires_grad_(False).eval()
        self.teacher_norm.requires_grad_(False).eval()
        self.depth = FrozenDepthFeatures(
            self.teacher_actor,
            self.teacher_norm,
            self.vec.get_obs_format()["policy"],
        )
        self.depth_slice, _ = get_obs_slice(self.vec.get_obs_format()["policy"], "depth_image")
        self.reader = SimReader(self.env)
        self.manager = self.env.scene["motion_reference"]
        self.motion = MotionLib(self.manager, device)
        self.robot = self.env.scene["robot"]
        self.action = self.env.action_manager.get_term("joint_pos")
        if list(self.action._target_names) != list(self.robot.joint_names):
            raise ValueError("Native action and robot joint orders differ")
        if self.action.cfg.clip is not None:
            raise ValueError("Explicitly adapt clipped teacher targets first")
        self.adapter = HoloMotionObservationAdapter(
            self.robot.joint_names,
            G1Kinematics(device, self.robot.joint_names, ["pelvis"]),
        )
        self.last_action = torch.zeros(num_envs, 29, device=device)
        self.last_observation = None
        self.steps = int(self.motion.clip_len.max()) + 5
        self.context_contract = None
        # No WM, critic fitting, optimizer, or teacher training is constructed.
        del runner, released

    def teacher(self, obs):
        return self.teacher_actor.act_inference(self.teacher_norm(obs))

    def to_holo(self, action):
        return self.adapter.action_from_sim(action, self.action.offset, self.action.scale)

    def to_native(self, action):
        return self.adapter.action_to_sim(action, self.action.offset, self.action.scale)

    def reset(self, seed, actor=None):
        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        self.last_action.zero_()
        if actor is not None:
            actor.reset_kv_cache(self.N, self.device)
            actor.clear_env_cache(None)
        return self.vec.get_observations()[0]

    def student_inputs(self, obs):
        frame, phase, origin = self.motion.current_frame()
        reference, times = self.adapter.reference_window(self.motion, frame, phase, origin)
        terms = self.adapter.terms(
            self.reader.state(),
            self.last_action,
            reference,
            reference_sample_time=times,
        )
        td = TensorDict(
            {f"unified/{k}": v for k, v in terms.items()},
            batch_size=[self.N],
        )
        depth = self.depth(obs[:, self.depth_slice])
        return td, depth, flatten_actor_terms(terms)

    @torch.no_grad()
    def rollout(
        self,
        actor=None,
        features=None,
        *,
        seed=7,
        teacher_probability=0.0,
        controller="student",
        video=False,
        max_steps=None,
    ):
        if controller not in ("teacher", "student", "mixed"):
            raise ValueError("Unknown controller")
        if controller != "teacher" and actor is None:
            raise ValueError("Student controller requires its Actor")
        obs = self.reset(seed, actor)
        active = torch.ones(self.N, device=self.device, dtype=torch.bool)
        steps = torch.zeros(self.N, device=self.device, dtype=torch.long)
        success = torch.zeros_like(active)
        fired = {n: torch.zeros_like(active) for n in (*FAILURE_TERMS, SUCCESS_TERM, "time_out")}
        errors, weights, rows, frames = [], [], [], []
        student_steps, queries, completed = 0, 0, 0
        durations = self.manager.complete_motion_lengths.clone()
        endpoint = (
            self.motion.root_p[self.motion.clip_end[0]][None]
            + self.motion.buffer._assigned_env_origins
        ).clone()
        final_position = self.reader.state()[0].clone()
        maximum_height = final_position[:, 2].clone()
        for _step in range(min(self.steps, max_steps or self.steps)):
            valid = active.clone()
            teacher_action = None
            if controller != "student" or features is not None:
                teacher_action = self.teacher(obs)
                queries += int(valid.sum())
            student_action = None
            if actor is not None:
                td, z, flat = self.student_inputs(obs)
                set_domain_context(actor, z)
                student_action = actor(td, mode="inference")["actions"]
                set_domain_context(actor, None)
                w = actor.actor_module.layers[0].last_group_weights
                weights.append(
                    w[valid, ..., -1].float().mean().cpu()
                    if actor.actor_module.layers[0].active_domains
                    else torch.tensor(0.0)
                )
                if features is not None:
                    labels = self.to_holo(teacher_action)
                    row = features.snapshot(labels, student_action, context=z)
                    row.update(
                        observation=flat.detach().cpu(),
                        teacher_observation=obs.detach().cpu().clone(),
                        teacher_native_action=teacher_action.detach().cpu(),
                    )
                    rows.append({k: v[valid.cpu()] for k, v in row.items()})
            if controller == "teacher":
                action = teacher_action
            elif controller == "student":
                action = self.to_native(student_action)
                student_steps += int(valid.sum())
            else:
                choose = torch.rand(self.N, 1, device=self.device)
                choose = choose < teacher_probability
                action = torch.where(choose, teacher_action, self.to_native(student_action))
                student_steps += int((valid & ~choose[:, 0]).sum())
            if not torch.isfinite(action).all():
                raise FloatingPointError("Nonfinite executed action")
            if video:
                frames.append(self.env.render())
            obs, _, done, _ = self.vec.step(action)
            self.last_action.copy_(self.to_holo(action))
            steps += valid
            ended = done.bool() & valid
            tm = self.env.termination_manager
            for name in fired:
                fired[name] |= tm.get_term(name) & ended
            failed = torch.stack([fired[n] for n in FAILURE_TERMS]).any(0)
            success |= ended & fired[SUCCESS_TERM] & ~failed
            state = self.reader.state()
            final_position[valid] = state[0][valid]
            maximum_height[valid] = torch.maximum(maximum_height[valid], state[0][valid, 2])
            ref = self.manager.reference_frame
            error = (state[0] - ref.base_pos_w[:, 0]).norm(dim=-1)
            errors.append(error[valid].mean().cpu())
            active &= ~ended
            completed += int(ended.sum())
            if not active.any():
                break
            ids = done.nonzero().flatten()
            if len(ids):
                self.env.reset(env_ids=ids)
                self.last_action[ids] = 0
                if actor is not None:
                    actor.clear_env_cache(ids)
                obs = self.vec.get_observations()[0]
        result = {
            "controller": controller,
            "seed": seed,
            "teacher_execution_probability": (
                1.0
                if controller == "teacher"
                else 0.0
                if controller == "student"
                else teacher_probability
            ),
            "episodes": self.N,
            "successful_episodes": int(success.sum()),
            "success_rate": float(success.float().mean()),
            "episode_seconds": (steps * self.env.step_dt).cpu().tolist(),
            "fraction_completed": torch.where(
                success,
                torch.ones_like(durations),
                (steps * self.env.step_dt / durations).clamp(max=1),
            )
            .cpu()
            .tolist(),
            "termination_counts": {k: int(v.sum()) for k, v in fired.items()},
            "teacher_queries": queries,
            "student_executed_transitions": student_steps,
            "new_labels": sum(len(x["target"]) for x in rows),
            "root_tracking_error_m": float(torch.stack(errors).mean()),
            "mean_domain_weight": float(torch.stack(weights).mean()) if weights else None,
            "episodes_terminated": completed,
            "actual_final_torso_position_w": final_position.cpu().tolist(),
            "reference_endpoint_torso_position_w": endpoint.cpu().tolist(),
            "endpoint_distance_m": (final_position - endpoint).norm(dim=-1).cpu().tolist(),
            "maximum_torso_z_w_m": maximum_height.cpu().tolist(),
            "success_definition": (
                "Official dataset_exhausted without any native failure; "
                "actual endpoint distances reported separately"
            ),
        }
        data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None
        return result, data, frames

    def close(self):
        self.vec.close()


def append_replay(bank, new, capacity):
    if not new or len(new["target"]) == 0:
        raise ValueError("DAgger collection produced no labels")
    if bank is None:
        return {k: v[-capacity:] for k, v in new.items()}
    if set(bank) != set(new):
        raise ValueError("Replay schema changed")
    return {k: torch.cat((bank[k], new[k]))[-capacity:] for k in bank}
