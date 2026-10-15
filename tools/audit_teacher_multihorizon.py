"""Frozen multi-horizon audit for a privileged Teacher checkpoint.

This tool never updates model weights.  It collects fresh closed-loop transitions
from motion windows that can be excluded by filename, then compares the EMA
Teacher with the deployable Student, state persistence, and a kinematic
constant-velocity baseline at 1/4/8/16 control steps.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from unified_motion.algorithms.sequence_training import parameter_stamp
from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.holomotion_mjlab import (
    HoloMotionK1MJLabAdapter,
    HoloMotionOnnxContract,
    make_holomotion_k1_mjlab_cfg,
)
from unified_motion.holomotion_mjlab_task import (
    HoloMotionK1MJLabTask,
    HoloMotionPhysicsRandomizationConfig,
)
from unified_motion.k1_policy_runtime import resume_k1_trainer, runtime_configuration
from unified_motion.physics_data import slice_physics
from unified_motion.privileged_teacher import core_groups, core_scale, normalized_core

DEFAULT_HOLOMOTION_ROOT = PROJECT_ROOT / "third_party" / "HoloMotion" / "source"
GROUP_NAMES = (
    "root_linear_velocity",
    "root_angular_velocity",
    "gravity_direction",
    "joint_position",
    "joint_velocity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-motion-dir",
        type=Path,
        help="exclude NPZ basenames found here (normally the Teacher training subset)",
    )
    parser.add_argument(
        "--include-motion-prefix",
        action="append",
        default=[],
        help="restrict evaluation to complete original motions by filename prefix; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 4, 8, 16))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--holomotion-root", type=Path, default=DEFAULT_HOLOMOTION_ROOT)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 2:
        parser.error("--num-envs must be positive and --steps must be at least 2")
    horizons = tuple(args.horizons)
    if any(type(value) is not int or value < 1 for value in horizons):
        parser.error("horizons must be positive integers")
    if tuple(sorted(set(horizons))) != horizons:
        parser.error("horizons must be unique and increasing")
    if max(horizons) >= args.steps:
        parser.error("the largest horizon must be smaller than --steps")
    return args


def _motion_paths(folder: Path, excluded: Path | None) -> tuple[list[Path], list[str]]:
    paths = sorted(folder.expanduser().resolve().rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"{folder} contains no NPZ motion windows")
    excluded_names = set()
    if excluded is not None:
        excluded_names = {path.name for path in excluded.expanduser().resolve().rglob("*.npz")}
        paths = [path for path in paths if path.name not in excluded_names]
    if not paths:
        raise ValueError("motion exclusion removed every candidate window")
    return paths, sorted(excluded_names)


def _valid_horizon(starts: torch.Tensor, dones: torch.Tensor, horizon: int) -> torch.Tensor:
    """Return [B,T-h+1] transitions that never cross an episode boundary."""
    length = starts.shape[1] - horizon + 1
    valid = torch.ones_like(starts[:, :length])
    for offset in range(horizon):
        valid &= ~dones[:, offset:length + offset]
    for offset in range(1, horizon):
        valid &= ~starts[:, offset:length + offset]
    return valid


def _rotate(vector: torch.Tensor, rotation_vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation with a stable small-angle fallback."""
    angle = rotation_vector.norm(dim=-1, keepdim=True)
    axis = rotation_vector / angle.clamp_min(1e-8)
    cosine, sine = angle.cos(), angle.sin()
    rotated = (vector * cosine + torch.linalg.cross(axis, vector) * sine
               + axis * (axis * vector).sum(-1, keepdim=True) * (1.0 - cosine))
    return torch.where(angle < 1e-7, vector + torch.linalg.cross(rotation_vector, vector), rotated)


def constant_velocity_core(current: torch.Tensor, horizon: int, dt: float, dof: int) -> torch.Tensor:
    """Kinematic baseline in the normalized Teacher state convention."""
    result = current.clone()
    omega = current[..., 3:6] * 10.0
    result[..., 6:9] = _rotate(current[..., 6:9], -omega * (horizon * dt))
    q = current[..., 9:9 + dof]
    qdot = current[..., 9 + dof:9 + 2 * dof]
    result[..., 9:9 + dof] = q + qdot * (10.0 / math.pi) * (horizon * dt)
    return result


def recursive_teacher(teacher, latent, actions, current, horizon: int):
    """Roll the checkpoint's one-step decoder without reading future states."""
    predicted = current
    for offset in range(horizon):
        predicted = teacher.decode(latent, actions[..., offset, :], predicted)["next_core"]
    return predicted


def _group_metrics(predicted, truth, core_valid, transition_valid, robot):
    mask = core_valid & transition_valid.unsqueeze(-1)
    scales = core_scale(robot, truth)
    rows = {}
    for name, section in zip(GROUP_NAMES, core_groups(truth.shape[-1]), strict=True):
        selected = mask[..., section]
        count = int(selected.sum())
        if not count:
            rows[name] = {"normalized_mse": None, "physical_rmse": None, "values": 0}
            continue
        error = predicted[..., section] - truth[..., section]
        normalized_mse = torch.where(selected, error.square(), 0.0).sum() / selected.sum()
        physical = error * scales[section]
        physical_mse = torch.where(selected, physical.square(), 0.0).sum() / selected.sum()
        rows[name] = {
            "normalized_mse": float(normalized_mse),
            "physical_rmse": float(physical_mse.sqrt()),
            "values": count,
        }
    return rows


def _contact_metrics(logits, truth, mask, transition_valid, current=None):
    import torch.nn.functional as F
    selected = mask & transition_valid.unsqueeze(-1)
    count = int(selected.sum())
    if not count:
        return {"bce": None, "brier": None, "accuracy": None, "f1": None,
                "switch_bce": None, "switch_accuracy": None, "switch_values": 0,
                "calibration": [], "values": 0}
    loss = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    probability = logits.sigmoid()
    prediction = logits >= 0
    label = truth >= 0.5
    tp = int((prediction & label & selected).sum())
    fp = int((prediction & ~label & selected).sum())
    fn = int((~prediction & label & selected).sum())
    result = {
        "bce": float(torch.where(selected, loss, 0.0).sum() / selected.sum()),
        "brier": float(torch.where(selected, (probability-truth).square(), 0.0).sum()
                       / selected.sum()),
        "accuracy": float(torch.where(selected, prediction == label, False).sum() / selected.sum()),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "values": count,
    }
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        in_bin = selected & (probability >= lower) & (
            probability <= upper if index == 9 else probability < upper)
        bin_count = int(in_bin.sum())
        bins.append({
            "lower": lower, "upper": upper, "values": bin_count,
            "mean_probability": (None if not bin_count else float(probability[in_bin].mean())),
            "empirical_contact_rate": (None if not bin_count else float(truth[in_bin].mean())),
        })
    result["calibration"] = bins
    switch = (torch.zeros_like(selected) if current is None else
              selected & ((truth >= .5) != (current >= .5)))
    switch_count = int(switch.sum())
    result.update(
        switch_bce=(None if not switch_count else
                    float(torch.where(switch, loss, 0.0).sum() / switch.sum())),
        switch_accuracy=(None if not switch_count else
                         float(torch.where(switch, prediction == label, False).sum() / switch.sum())),
        switch_values=switch_count)
    return result


@torch.no_grad()
def audit(trainer, rollout, horizons: tuple[int, ...], dt: float) -> dict:
    teacher = trainer.privileged_teacher_ema
    if teacher is None:
        raise ValueError("checkpoint does not contain a privileged Teacher")
    teacher.eval()
    ids = rollout.privileged_teacher_env_ids
    if ids is None or len(ids) != rollout.actions.shape[0]:
        raise ValueError("audit requires Teacher inputs for every environment")
    inputs = rollout.privileged_teacher_input
    current = rollout.privileged_teacher_core
    if inputs is None or current is None:
        raise ValueError("rollout does not contain privileged Teacher observations")
    starts = rollout.physics_episode_start.index_select(0, ids)
    dones = rollout.dones.squeeze(-1).index_select(0, ids)
    collected_actions = (rollout.actions if rollout.executed_actions is None
                         else rollout.executed_actions)
    actions = collected_actions.index_select(0, ids)
    teacher_output = teacher(inputs, starts, actions, current)
    next_obs = rollout.next_physics_observation
    truth_all = normalized_core(next_obs, trainer.dynamics.robot).index_select(0, ids)
    valid_all = next_obs.core_valid.index_select(0, ids)

    # Recompute the frozen deployable Student on exactly the same sequence.
    dynamics = trainer.actor_dynamics.eval()
    encoding = dynamics.forward_sequence(
        rollout.physics_observation,
        episode_start=rollout.physics_episode_start,
        initial_hidden=rollout.physics_initial_hidden,
        previous_predicted_core=rollout.previous_predicted_core,
        previous_prediction_valid=rollout.previous_prediction_valid,
    )
    one_step_student = dynamics.predict_next(
        rollout.physics_observation,
        collected_actions,
        encoding.latent,
        physics_encoding=encoding.physics_encoding(),
    )
    scale = core_scale(dynamics.robot, actions)
    report = {}
    for horizon in horizons:
        length = actions.shape[1] - horizon + 1
        transition_valid = _valid_horizon(starts, dones, horizon)
        truth = truth_all[:, horizon - 1:length + horizon - 1]
        target_valid = valid_all[:, horizon - 1:length + horizon - 1]
        now = current[:, :length]
        action_window = torch.stack(
            [actions[:, offset:length + offset] for offset in range(horizon)], dim=-2)
        latent = teacher_output["latent"][:, :length]
        predictions = {
            "persistence": now,
            "constant_velocity": constant_velocity_core(
                now, horizon, dt, trainer.dynamics.robot.dof),
            "teacher_recursive": recursive_teacher(teacher, latent, action_window, now, horizon),
        }
        native = horizon == 1 or horizon in teacher.config.prediction_horizons
        if horizon == 1:
            predictions["teacher_direct"] = teacher_output["next_core"][:, :length]
            predictions["student_direct"] = one_step_student.next_core.index_select(0, ids)[:, :length] / scale
        elif horizon in teacher.config.prediction_horizons:
            predictions["teacher_direct"] = teacher.predict_future_core(
                latent, action_window, now, horizon=horizon)
        if horizon in dynamics.prediction_horizons and horizon > 1:
            selected_encoding = slice_physics(rollout.physics_observation, 0, length)
            physics_encoding = encoding.physics_encoding()
            from unified_motion.models.physics import PhysicsEncoding
            physics_encoding = PhysicsEncoding(
                physics_encoding.context[:, :length],
                physics_encoding.contact_tokens[:, :length],
                physics_encoding.object_tokens[:, :length],
            )
            student = dynamics.predict_future_core(
                selected_encoding, action_window, encoding.latent[:, :length],
                horizon=horizon, physics_encoding=physics_encoding)
            predictions["student_direct"] = student.index_select(0, ids) / scale
        report[str(horizon)] = {
            "valid_transitions": int(transition_valid.sum()),
            "teacher_has_native_direct_head": native,
            "state": {
                name: _group_metrics(value, truth, target_valid, transition_valid, dynamics.robot)
                for name, value in predictions.items()
            },
        }
        clip_ids = rollout.clip_ids.index_select(0, ids)[:, :length]
        clip_rows = {}
        for clip_id in clip_ids[transition_valid].unique().tolist():
            selected_transition = transition_valid & (clip_ids == clip_id)
            clip_rows[trainer.task.sampler.library.clip_names[clip_id]] = {
                "clip_id": clip_id,
                "valid_transitions": int(selected_transition.sum()),
                "state": {
                    name: _group_metrics(value, truth, target_valid,
                                         selected_transition, dynamics.robot)
                    for name, value in predictions.items()
                },
            }
        report[str(horizon)]["clips"] = clip_rows
        if horizon == 1:
            count = teacher.config.contact_slots
            contact_truth = next_obs.contact_features.index_select(0, ids)[:, :length, :count, 10]
            contact_mask = next_obs.contact_mask.index_select(0, ids)[:, :length, :count]
            current_contacts = rollout.privileged_teacher_contacts[:, :length, :count]
            persistence_logits = torch.logit(current_contacts.clamp(0.01, 0.99))
            report[str(horizon)]["contact"] = {
                "teacher": _contact_metrics(teacher_output["contact_logits"][:, :length],
                                             contact_truth, contact_mask, transition_valid,
                                             current_contacts),
                "student": _contact_metrics(one_step_student.next_contact_logits.index_select(0, ids)[:, :length, :count],
                                             contact_truth, contact_mask, transition_valid,
                                             current_contacts),
                "persistence": _contact_metrics(persistence_logits, contact_truth,
                                                 contact_mask, transition_valid,
                                                 current_contacts),
            }
    return report


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; frozen audit was not started")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runtime = runtime_configuration(payload)
    contract = HoloMotionOnnxContract(**runtime["task_contract"])
    paths, excluded = _motion_paths(args.motion_dir, args.exclude_motion_dir)
    if args.include_motion_prefix:
        paths = [path for path in paths if any(
            path.name.startswith(prefix + "__") for prefix in args.include_motion_prefix)]
        if not paths:
            raise ValueError("motion-prefix restriction removed every candidate window")
    library = MotionLibrary.from_named_mjlab_files(paths, robot_id="k1_22dof", device=device)
    offsets = tuple(index * 0.02 for index in range(contract.future_frames + 2))
    sampler = MotionSampler(library, num_envs=args.num_envs, step_dt=0.02,
                            future_offsets_s=offsets)
    cfg = make_holomotion_k1_mjlab_cfg(args.asset_root, contract, num_envs=args.num_envs)
    cfg.seed = args.seed
    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        randomization = runtime.get("physics_randomization")
        task = HoloMotionK1MJLabTask(
            HoloMotionK1MJLabAdapter(
                K1MJLabBridge(env, reference_steps=contract.future_frames + 2), contract),
            sampler,
            physics_randomization=HoloMotionPhysicsRandomizationConfig(**randomization),
        )
        trainer = resume_k1_trainer(args.checkpoint, task,
                                    holomotion_root=args.holomotion_root)
        before = (parameter_stamp(trainer.privileged_teacher_ema),
                  parameter_stamp(trainer.actor_dynamics))
        minimum_remaining = args.steps * sampler.step_dt + float(sampler.offsets_s[-1])
        trainer.reset(random_start=True, min_remaining_s=minimum_remaining)
        rollout = trainer.collect(args.steps)
        result = audit(trainer, rollout, tuple(args.horizons), sampler.step_dt)
        after = (parameter_stamp(trainer.privileged_teacher_ema),
                 parameter_stamp(trainer.actor_dynamics))
        if before != after:
            raise RuntimeError("frozen audit unexpectedly changed model state")
        document = {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_iteration": payload.get("iter"),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "motion_candidates": len(paths),
            "included_motion_prefixes": args.include_motion_prefix,
            "excluded_training_basenames": excluded,
            "excluded_training_count": len(excluded),
            "teacher_config": asdict(trainer.privileged_teacher.config),
            "student_prediction_horizons": list(trainer.actor_dynamics.prediction_horizons),
            "weights_unchanged": True,
            "results": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
