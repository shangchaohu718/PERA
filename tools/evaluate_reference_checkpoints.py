"""冻结评测正式参考动作 checkpoint；不更新网络、不修改训练目录。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from unified_motion.api import (
    DynamicsModel, MotionLibrary, MotionSampler, MotionTrackingReward,
    MotionTrackingRewardConfig, MotionTrackingTask, build_policy, get_robot_contract,
)
from unified_motion.algorithms.sequence_training import PolicyInput
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.envs.mjlab_physics import K1PhysicsBridge, make_k1_physics_cfg
from unified_motion.models.physics import DynamicsMemory, PhysicsMLPActorCritic


@dataclass
class CandidateResult:
    checkpoint: str
    iteration: int
    completed_reference: bool
    terminated_early: bool
    truncated: bool
    steps: int
    duration_s: float
    reward_mean: float
    reward_terms: dict[str, float]
    tracking_metrics: dict[str, float]
    final_tracking_metrics: dict[str, float]
    action_abs_mean: float
    action_abs_max: float


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_payload(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != 1 or not isinstance(payload.get("arguments"), dict):
        raise ValueError(f"{path} 不是正式参考动作训练 checkpoint")
    return payload


@torch.no_grad()
def _reset_fixed_clip(task: MotionTrackingTask, clip_index: int) -> None:
    """Reset physics and select one explicit clip for a fair comparison."""
    task.bridge.reset()
    data = task.bridge.entity.data
    clip_ids = torch.full(
        (task.num_envs,), clip_index, device=task.device, dtype=torch.long)
    task.sampler.reset(
        clip_ids=clip_ids,
        current_root_pos_w=data.root_link_pos_w,
        current_root_quat_wxyz=data.root_link_quat_w,
    )
    if task.reference_state_init:
        task._initialize_from_reference(None)


def _same_training_contract(reference: dict, candidate: dict, path: Path) -> None:
    keys = (
        "asset_root", "motions", "backbone", "d_model", "hidden_dim", "layers",
        "context_len", "recurrent_hidden_dim", "max_replay_steps", "experts",
        "init_std", "adaptation", "body_pos_weight", "body_rot_weight",
        "body_lin_vel_weight", "latent_width", "prediction_horizons",
        "extended_contact_supervision",
    )
    changed = [key for key in keys if candidate.get(key) != reference.get(key)]
    if changed:
        raise ValueError(f"{path} 与首个 checkpoint 训练合同不同：{changed}")


@torch.no_grad()
def _evaluate_one(task: MotionTrackingTask, dynamics: DynamicsModel, policy,
                  checkpoint: Path, payload: dict, max_steps: int,
                  clip_index: int) -> CandidateResult:
    dynamics.load_state_dict(payload["dynamics"]["model"], strict=True)
    policy.load_state_dict(payload["ppo"]["model"], strict=True)
    dynamics.eval(); policy.eval()
    dynamics.invalidate_memories()
    if hasattr(policy, "invalidate_caches"):
        policy.invalidate_caches()

    _reset_fixed_clip(task, clip_index)
    memory: DynamicsMemory | None = None
    cache = None
    rewards: list[float] = []
    terms: dict[str, list[float]] = {}
    metrics: dict[str, list[float]] = {}
    final_metrics: dict[str, float] = {}
    action_abs_mean: list[float] = []
    action_abs_max = 0.0
    completed = early = cut = False

    for _ in range(max_steps):
        frame = task.read()
        starts = task.episode_start.clone()
        encoding, next_memory = dynamics.step(
            frame, episode_start=starts, env_ids=task.env_ids, memory=memory)
        privileged = task.privileged()
        actor_encoding = (task.policy_dynamics(encoding, privileged)
                          if hasattr(task, "policy_dynamics") else encoding)
        inp = PolicyInput(frame, task.goal(), actor_encoding.context,
                          actor_encoding.latent, actor_encoding.prediction_error,
                          starts, privileged)
        if isinstance(policy, PhysicsMLPActorCritic):
            output = policy(inp.physics, inp.dynamics(), inp.goal,
                            privileged=inp.privileged)
            action = output.mean
        else:
            _, action, _, cache = policy.step(
                task.robot_id, inp.physics, inp.dynamics(), inp.goal,
                env_ids=task.env_ids, episode_start=starts, cache=cache,
                privileged=privileged, deterministic=True)
        prediction = dynamics.predict_next(frame, action, encoding.latent)
        memory = dynamics.remember_prediction(next_memory, prediction)
        transition = task.step(action)

        rewards.append(float(transition.reward.mean().item()))
        action_abs_mean.append(float(action.abs().mean().item()))
        action_abs_max = max(action_abs_max, float(action.abs().max().item()))
        for name, value in (transition.terms or {}).items():
            terms.setdefault(name, []).append(float(value.mean().item()))
        current_metrics = transition.metrics or {}
        for name, value in current_metrics.items():
            if value.ndim == 1:
                number = float(value.mean().item())
                metrics.setdefault(name, []).append(number)
                final_metrics[name] = number

        reference_done = bool(task.sampler.finished().all().item())
        env_terminated = bool(transition.terminated.any().item())
        env_truncated = bool(transition.truncated.any().item())
        if reference_done or env_terminated or env_truncated:
            completed = reference_done
            early = env_terminated and not reference_done
            cut = env_truncated
            break

    result = CandidateResult(
        checkpoint=checkpoint.name, iteration=int(payload["iteration"]),
        completed_reference=completed, terminated_early=early, truncated=cut,
        steps=len(rewards), duration_s=len(rewards) * float(task.sampler.step_dt),
        reward_mean=_mean(rewards),
        reward_terms={name: _mean(value) for name, value in terms.items()},
        tracking_metrics={name: _mean(value) for name, value in metrics.items()},
        final_tracking_metrics=final_metrics,
        action_abs_mean=_mean(action_abs_mean), action_abs_max=action_abs_max,
    )
    numbers = [result.reward_mean, result.action_abs_mean, result.action_abs_max,
               *result.reward_terms.values(), *result.tracking_metrics.values()]
    if not all(math.isfinite(value) for value in numbers):
        raise FloatingPointError(f"{checkpoint} 冻结评测出现 NaN/Inf")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="确定性评测默认单环境，避免复制相同轨迹浪费显存")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--clip-index", type=int, default=0,
                        help="固定参考片段编号；所有checkpoint使用同一片段")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_steps < 1 or args.clip_index < 0:
        parser.error("num-envs/max-steps必须为正整数，clip-index必须非负")
    if not torch.cuda.is_available():
        raise RuntimeError("冻结参考动作评测需要真实 MJLab CUDA 环境")

    device = torch.device(args.device)
    first = _load_payload(args.checkpoints[0], device)
    train_args = first["arguments"]
    for path in args.checkpoints[1:]:
        probe = _load_payload(path, device)
        _same_training_contract(train_args, probe["arguments"], path)

    seed = int(train_args["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    robot = get_robot_contract("k1_22dof")
    motion_path = Path(train_args["motions"])
    library = (MotionLibrary.from_named_mjlab_file(
        motion_path, robot_id="k1_22dof", device=device)
        if motion_path.is_file() else MotionLibrary.from_project_folder(
            motion_path, robot_id="k1_22dof", device=device))
    if args.clip_index >= len(library.clip_names):
        parser.error(
            f"clip-index越界：动作库有{len(library.clip_names)}段，"
            f"收到{args.clip_index}")
    cfg = make_k1_physics_cfg(train_args["asset_root"], num_envs=args.num_envs)
    cfg.seed = seed
    cfg.episode_length_s = max(float(cfg.episode_length_s),
                               float(library.clip_durations_s.max().item()) + .02)

    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        bridge = K1MJLabBridge(env)
        physics = K1PhysicsBridge(bridge)
        sampler = MotionSampler(library, num_envs=args.num_envs, step_dt=env.step_dt,
                                future_offsets_s=(0., .04, .10, .20))
        reward_cfg = MotionTrackingRewardConfig(
            body_pos_weight=float(train_args["body_pos_weight"]),
            body_rot_weight=float(train_args["body_rot_weight"]),
            body_lin_vel_weight=float(train_args["body_lin_vel_weight"]))
        full = train_args["adaptation"] == "full"
        task = MotionTrackingTask(
            physics, sampler, MotionTrackingReward(robot, reward_cfg),
            actor_observation="deployable" if full else "full",
            privileged_learning=full,
            extended_contact_supervision=bool(
                train_args.get("extended_contact_supervision", False)),
            reference_state_init=bool(train_args.get("reference_state_init", False)),
            random_reference_start=False)
        latent_width = int(train_args.get("latent_width", 64))
        prediction_horizons = tuple(train_args.get("prediction_horizons", (1,)))
        dynamics = DynamicsModel(
            robot, latent_width=latent_width,
            system_id_dim=task.privileged_dim if full else 0,
            teacher_student=full,
            prediction_horizons=prediction_horizons).to(device)
        policy = build_policy(
            {"k1_22dof": robot}, backbone=train_args["backbone"],
            d_model=int(train_args["d_model"]), hidden_dim=int(train_args["hidden_dim"]),
            n_layers=int(train_args["layers"]), context_len=int(train_args["context_len"]),
            num_experts=int(train_args["experts"]), init_std=float(train_args["init_std"]),
            latent_width=latent_width,
            recurrent_hidden_dim=train_args["recurrent_hidden_dim"],
            max_replay_steps=int(train_args["max_replay_steps"]),
            privileged_dims=({"k1_22dof": task.privileged_dim} if full else None)).to(device)

        results = []
        for path in args.checkpoints:
            payload = _load_payload(path, device)
            _same_training_contract(train_args, payload["arguments"], path)
            result = _evaluate_one(
                task, dynamics, policy, path, payload, args.max_steps,
                args.clip_index)
            results.append(result.__dict__)
            print(json.dumps(results[-1], ensure_ascii=False, allow_nan=False), flush=True)
        ranking = sorted(results, key=lambda item: (
            not item["completed_reference"], item["terminated_early"],
            -item["duration_s"], -item["reward_mean"]))
        report = {
            "schema": 1, "deterministic_actor": True,
            "motion": str(motion_path), "num_envs": args.num_envs,
            "clip_index": args.clip_index,
            "clip_name": library.clip_names[args.clip_index],
            "max_steps": args.max_steps, "results": results,
            "ranking": [item["checkpoint"] for item in ranking],
            "recommended": ranking[0]["checkpoint"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
        temporary.replace(args.output)
        print(f"评测完成：{args.output}，推荐 {report['recommended']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
