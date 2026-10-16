#!/usr/bin/env python3
"""Train and evaluate the grouped G1 Teacher on motion-isolated CPU replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified_motion.privileged_teacher import (  # noqa: E402
    GROUPED_PHYSICS_LATENT,
    PrivilegedTeacherConfig,
    PrivilegedTransformerTeacher,
    core_groups,
    teacher_supervision,
)
from unified_motion.robot_contract import get_robot_contract  # noqa: E402


def load_replay(path: Path) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") not in (1, 2) or saved.get("robot") != "g1_29dof":
        raise ValueError(f"unsupported G1 Teacher replay: {path}")
    frames = saved["frames"]
    required = {"critic", "core", "contacts", "found", "start", "action", "next_core"}
    optional = {"params", "next_contacts", "next_found", "next_contact_valid"}
    if not required.issubset(frames) or set(frames) - required - optional:
        raise ValueError(f"replay fields mismatch in {path}: {set(frames)}")
    if ("next_contacts" in frames) != ("next_found" in frames):
        raise ValueError(f"replay must provide both next contact tensors: {path}")
    if saved.get("schema") == 2 and not {
            "next_contacts", "next_found", "next_contact_valid"}.issubset(frames):
        raise ValueError(f"schema-2 replay requires explicit next contact validity: {path}")
    if "params" not in frames:
        # Backward-compatible nominal-physics replay. New randomized replays
        # always persist the exact normalized pair.
        frames["params"] = torch.zeros(frames["core"].shape[0], 2)
    lengths = {value.shape[0] for value in frames.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(f"invalid replay lengths in {path}: {lengths}")
    return {
        "path": path,
        "motion_name": saved["motion_name"],
        "source_motion_name": saved.get("source_motion_name", saved["motion_name"]),
        "frames": frames,
    }


def contact_sequences(frames: dict, start: int, end: int, device: torch.device):
    """Return strictly lagged inputs and next-frame contact supervision."""
    current = frames["contacts"][start:end]
    current_found = frames["found"][start:end]
    starts = frames["start"][start:end].bool()
    history = torch.zeros_like(current)
    history_found = torch.zeros_like(current_found)
    if start > 0:
        history[0] = frames["contacts"][start - 1]
        history_found[0] = frames["found"][start - 1]
    if end - start > 1:
        history[1:] = frames["contacts"][start:end - 1]
        history_found[1:] = frames["found"][start:end - 1]
    history[starts] = 0
    history_found[starts] = False

    valid = torch.ones(end - start, dtype=torch.bool)
    if "next_contacts" in frames:
        target = frames["next_contacts"][start:end]
        target_found = frames["next_found"][start:end]
        valid = frames.get(
            "next_contact_valid", torch.ones_like(frames["start"], dtype=torch.bool)
        )[start:end].bool().clone()
    else:
        # Schema-1 recovery: frame t+1 was recorded as the following frame's
        # pre-step contact. The final replay frame has no recoverable target.
        target = torch.zeros_like(current)
        target_found = torch.zeros_like(current_found)
        available = min(end, frames["contacts"].shape[0] - 1) - start
        if available > 0:
            target[:available] = frames["contacts"][start + 1:start + 1 + available]
            target_found[:available] = frames["found"][start + 1:start + 1 + available]
            valid[:available] &= ~frames["start"][start + 1:start + 1 + available].bool()
        if end == frames["contacts"].shape[0]:
            valid[-1] = False
    return tuple(value.unsqueeze(0).to(device) for value in (
        history, history_found, target, target_found, valid))


def make_window(replay: dict, start: int, sequence: int, device: torch.device) -> dict:
    frames = replay["frames"]
    end = min(start + sequence, frames["core"].shape[0])
    critic = frames["critic"][start:end].unsqueeze(0).to(device)
    contact_history, history_found, contact, found, contact_valid = contact_sequences(
        frames, start, end, device)
    length = end - start
    params = frames["params"][start:end].unsqueeze(0).to(device)
    privilege = torch.cat((
        contact_history.flatten(-2), torch.zeros(1, length, 68, device=device), params,
    ), -1)
    starts = frames["start"][start:end].unsqueeze(0).to(device).bool()
    # Every sampled window is an independent Transformer sequence.
    starts[:, 0] = True
    return {
        "inputs": torch.cat((critic, privilege), -1),
        "contact_history": contact_history,
        "history_found": history_found,
        "contact": contact,
        "found": found,
        "contact_valid": contact_valid,
        "params": params,
        "starts": starts,
        "actions": frames["action"][start:end].unsqueeze(0).to(device),
        "core": frames["core"][start:end].unsqueeze(0).to(device),
        "next_core": frames["next_core"][start:end].unsqueeze(0).to(device),
    }


def multistep_supervision(teacher, output: dict, batch: dict, config) -> torch.Tensor:
    """Match the production Teacher's direct 4/8-step executed-action loss."""
    terms = []
    steps = batch["core"].shape[1]
    for horizon in config.prediction_horizons:
        length = steps - horizon + 1
        if horizon <= 1 or length < 1:
            continue
        valid = torch.ones_like(batch["starts"][:, :length])
        for offset in range(1, horizon):
            valid &= ~batch["starts"][:, offset:length + offset]
        actions = torch.stack([
            batch["actions"][:, offset:length + offset]
            for offset in range(horizon)
        ], -2)
        predicted = teacher.predict_future_core(
            output["latent"][:, :length], actions, batch["core"][:, :length],
            horizon=horizon)
        truth = batch["next_core"][:, horizon - 1:length + horizon - 1]
        group_losses = []
        for section in core_groups(truth.shape[-1]):
            mask = valid.unsqueeze(-1).expand_as(truth[..., section])
            error = torch.where(
                mask, predicted[..., section] - truth[..., section].detach(), 0.0)
            group_losses.append(error.square().sum() / mask.sum().clamp_min(1))
        terms.append(torch.stack(group_losses).mean())
    return torch.stack(terms).mean() if terms else batch["core"].new_zeros(())


def future_contact_supervision(teacher, output: dict, batch: dict, config) -> torch.Tensor:
    """Supervise t+4/t+8 contacts using future actions and future-only labels."""
    terms = []
    steps = batch["core"].shape[1]
    for horizon in config.prediction_horizons:
        length = steps - horizon + 1
        if horizon <= 1 or length < 1:
            continue
        actions = torch.stack([
            batch["actions"][:, offset:length + offset]
            for offset in range(horizon)
        ], -2)
        prediction = teacher.predict_future_contacts(
            output["latent"][:, :length], actions, batch["core"][:, :length],
            horizon=horizon)
        found = batch["found"][:, horizon - 1:length + horizon - 1]
        contact = batch["contact"][:, horizon - 1:length + horizon - 1]
        valid = batch["contact_valid"][:, horizon - 1:length + horizon - 1]
        for offset in range(1, horizon):
            valid = valid & ~batch["starts"][:, offset:length + offset]
        mask = valid.unsqueeze(-1).expand_as(found)
        contact_loss = F.binary_cross_entropy_with_logits(
            prediction["contact_logits"], found.float(), reduction="none")
        physics_target = contact[..., (9, 11, 12, 13)]
        physics_mask = mask.unsqueeze(-1).expand_as(physics_target)
        physics_loss = torch.where(
            physics_mask,
            (prediction["contact_physics"] - physics_target.detach()).square(), 0.0,
        ).sum() / physics_mask.sum().clamp_min(1)
        modes = torch.where(found, torch.where(
            contact[..., 12] > config.contact_slip_threshold_m_s, 2, 1), 0).long()
        mode_loss = F.cross_entropy(
            prediction["contact_mode_logits"].flatten(0, -2), modes.flatten(),
            reduction="none").unflatten(0, modes.shape)
        terms.append(
            torch.where(mask, contact_loss, 0.0).sum() / mask.sum().clamp_min(1)
            + config.interaction_weight * physics_loss
            + config.contact_mode_weight * (
                torch.where(mask, mode_loss, 0.0).sum() / mask.sum().clamp_min(1)))
    return torch.stack(terms).mean() if terms else batch["core"].new_zeros(())


@torch.no_grad()
def evaluate(teacher, replay: dict, config, device, sequence: int) -> dict:
    teacher.eval()
    names = ("base_lin_vel", "base_ang_vel", "gravity", "joint_pos", "joint_vel")
    pred_sse = torch.zeros(5, device=device)
    persist_sse = torch.zeros(5, device=device)
    zero_parameter_sse = torch.zeros(5, device=device)
    counts = torch.zeros(5, device=device)
    parameter_sse = torch.zeros(2, device=device)
    parameter_count = 0
    future = {
        horizon: {
            "teacher": torch.zeros(5, device=device),
            "persistence": torch.zeros(5, device=device),
            "zero_parameter": torch.zeros(5, device=device),
            "count": torch.zeros(5, device=device),
        }
        for horizon in config.prediction_horizons if horizon > 1
    }
    tp = fp = fn = correct = total = 0.0
    frame_count = replay["frames"]["core"].shape[0]
    for start in range(0, frame_count, sequence):
        batch = make_window(replay, start, sequence, device)
        output = teacher(batch["inputs"], batch["starts"], batch["actions"], batch["core"])
        zero_parameter_inputs = batch["inputs"].clone()
        zero_parameter_inputs[..., -2:] = 0.0
        zero_parameter_output = teacher(
            zero_parameter_inputs, batch["starts"], batch["actions"], batch["core"])
        parameter_error = output["system_id"] - batch["params"]
        parameter_sse += parameter_error.square().sum((0, 1))
        parameter_count += parameter_error.shape[0] * parameter_error.shape[1]
        target_delta = batch["next_core"] - batch["core"]
        for index, section in enumerate(core_groups(67)):
            error = output["delta_core"][..., section] - target_delta[..., section]
            baseline = -target_delta[..., section]
            zero_parameter_error = (
                zero_parameter_output["delta_core"][..., section]
                - target_delta[..., section])
            pred_sse[index] += error.square().sum()
            persist_sse[index] += baseline.square().sum()
            zero_parameter_sse[index] += zero_parameter_error.square().sum()
            counts[index] += error.numel()
        for horizon, sums in future.items():
            length = batch["core"].shape[1] - horizon + 1
            if length < 1:
                continue
            actions = torch.stack([
                batch["actions"][:, offset:length + offset]
                for offset in range(horizon)
            ], -2)
            predicted_future = teacher.predict_future_core(
                output["latent"][:, :length], actions, batch["core"][:, :length],
                horizon=horizon)
            zero_parameter_future = teacher.predict_future_core(
                zero_parameter_output["latent"][:, :length], actions,
                batch["core"][:, :length], horizon=horizon)
            truth_future = batch["next_core"][:, horizon - 1:length + horizon - 1]
            for index, section in enumerate(core_groups(67)):
                error = predicted_future[..., section] - truth_future[..., section]
                persistence = (
                    batch["core"][:, :length, section] - truth_future[..., section])
                zero_error = (
                    zero_parameter_future[..., section] - truth_future[..., section])
                sums["teacher"][index] += error.square().sum()
                sums["persistence"][index] += persistence.square().sum()
                sums["zero_parameter"][index] += zero_error.square().sum()
                sums["count"][index] += error.numel()
        predicted = output["contact_logits"].sigmoid() >= 0.5
        truth = batch["found"]
        contact_mask = batch["contact_valid"].unsqueeze(-1).expand_as(truth)
        tp += float((predicted & truth & contact_mask).sum())
        fp += float((predicted & ~truth & contact_mask).sum())
        fn += float((~predicted & truth & contact_mask).sum())
        correct += float(((predicted == truth) & contact_mask).sum())
        total += float(contact_mask.sum())
    pred_mse = pred_sse / counts.clamp_min(1)
    persist_mse = persist_sse / counts.clamp_min(1)
    zero_parameter_mse = zero_parameter_sse / counts.clamp_min(1)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "motion_name": replay["motion_name"],
        "frames": frame_count,
        "transition": {
            name: {
                "teacher_rmse_normalized": float(pred_mse[i].sqrt()),
                "persistence_rmse_normalized": float(persist_mse[i].sqrt()),
                "zero_physics_parameter_rmse_normalized": float(
                    zero_parameter_mse[i].sqrt()),
                "mse_improvement_vs_persistence": float(
                    1.0 - pred_mse[i] / persist_mse[i].clamp_min(1.0e-12)),
                "mse_improvement_from_true_physics_parameters": float(
                    1.0 - pred_mse[i] /
                    zero_parameter_mse[i].clamp_min(1.0e-12)),
            } for i, name in enumerate(names)
        },
        "next_contact_prediction": {
            "accuracy": correct / max(total, 1.0),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
            "note": "targets are t+1; contact inputs are strictly t-1 or older",
        },
        "physics_parameter_reconstruction": {
            "friction_rmse_normalized": float(
                (parameter_sse[0] / max(parameter_count, 1)).sqrt()),
            "torso_mass_scale_rmse_normalized": float(
                (parameter_sse[1] / max(parameter_count, 1)).sqrt()),
        },
        "future_transition": {
            str(horizon): {
                name: {
                    "teacher_rmse_normalized": float(
                        (sums["teacher"][i] / sums["count"][i].clamp_min(1)).sqrt()),
                    "persistence_rmse_normalized": float(
                        (sums["persistence"][i] / sums["count"][i].clamp_min(1)).sqrt()),
                    "zero_physics_parameter_rmse_normalized": float(
                        (sums["zero_parameter"][i] /
                         sums["count"][i].clamp_min(1)).sqrt()),
                }
                for i, name in enumerate(names)
            }
            for horizon, sums in future.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--validation", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=15000)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates < 1 or args.sequence < 2:
        raise ValueError("updates>=1 and sequence>=2 are required")
    train_replays = [load_replay(path.resolve()) for path in args.train]
    validation_replays = [load_replay(path.resolve()) for path in args.validation]
    train_names = {item["motion_name"] for item in train_replays}
    validation_names = {item["motion_name"] for item in validation_replays}
    train_sources = {item["source_motion_name"] for item in train_replays}
    validation_sources = {item["source_motion_name"] for item in validation_replays}
    overlap = train_sources & validation_sources
    if overlap:
        raise ValueError(f"source-motion train/validation leakage: {sorted(overlap)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
        config = PrivilegedTeacherConfig(**saved["teacher_config"])
        teacher = PrivilegedTransformerTeacher(
            get_robot_contract("g1_29dof"), 64, 2, config,
            latent_groups=GROUPED_PHYSICS_LATENT,
        ).to(device)
        teacher.load_state_dict(saved["model"], strict=True)
        losses = []
        train_seconds = 0.0
    else:
        has_randomized_physics = any(
            bool(replay["frames"]["params"].abs().max() > 0)
            for replay in train_replays)
        config = PrivilegedTeacherConfig(
            objective_version=5, critic_dim=1046, contact_slots=4,
            object_slots=4, width=128, heads=8, layers=4,
            context_steps=64, train_envs=1, learning_rate=1.0e-4,
            parameter_reconstruction_weight=(0.1 if has_randomized_physics else 0.0),
            future_contact_weight=0.25, contact_input_lag=1,
            latent_groups=GROUPED_PHYSICS_LATENT,
        )
        teacher = PrivilegedTransformerTeacher(
            get_robot_contract("g1_29dof"), 64, 2, config,
            latent_groups=GROUPED_PHYSICS_LATENT,
        ).to(device).train()
        optimizer = torch.optim.Adam(teacher.parameters(), lr=config.learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        losses = []
        started = time.perf_counter()
        for _ in range(args.updates):
            replay_index = int(torch.randint(len(train_replays), (), generator=generator))
            replay = train_replays[replay_index]
            frame_count = replay["frames"]["core"].shape[0]
            maximum_start = max(0, frame_count - args.sequence)
            start = int(torch.randint(maximum_start + 1, (), generator=generator))
            batch = make_window(replay, start, args.sequence, device)
            contact_physics = batch["contact"][..., (9, 11, 12, 13)]
            modes = torch.where(batch["found"], torch.where(
                batch["contact"][..., 12] > config.contact_slip_threshold_m_s, 2, 1), 0).long()
            output = teacher(batch["inputs"], batch["starts"], batch["actions"], batch["core"])
            loss = teacher_supervision(
                output, batch["next_core"], torch.ones_like(batch["next_core"], dtype=torch.bool),
                batch["found"].float(), batch["contact_valid"].unsqueeze(-1).expand_as(
                    batch["found"]), batch["params"],
                torch.ones(1, batch["core"].shape[1], dtype=torch.bool, device=device),
                config, current_core=batch["core"], contact_physics=contact_physics,
                contact_modes=modes,
            )
            loss = loss + config.multistep_weight * multistep_supervision(
                teacher, output, batch, config)
            loss = loss + config.future_contact_weight * future_contact_supervision(
                teacher, output, batch, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_seconds = time.perf_counter() - started

    train_metrics = [evaluate(teacher, replay, config, device, args.sequence)
                     for replay in train_replays]
    validation_metrics = [evaluate(teacher, replay, config, device, args.sequence)
                          for replay in validation_replays]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (args.checkpoint.resolve() if args.checkpoint is not None
                  else args.output_dir / "teacher_grouped_z64_multimotion.pt")
    if args.checkpoint is None:
        torch.save({
            "schema": 1,
            "teacher_config": config.__dict__,
            "latent_groups": GROUPED_PHYSICS_LATENT,
            "model": teacher.state_dict(),
            "train_motion_names": sorted(train_names),
            "validation_motion_names": sorted(validation_names),
            "train_source_motion_names": sorted(train_sources),
            "validation_source_motion_names": sorted(validation_sources),
        }, checkpoint)
    report = {
        "status": "passed",
        "device": str(device),
        "updates": (0 if args.checkpoint is not None else args.updates),
        "sequence": args.sequence,
        "train_seconds": train_seconds,
        "updates_per_second": (0.0 if not train_seconds else args.updates / train_seconds),
        "initial_loss": (None if not losses else losses[0]),
        "final_loss": (None if not losses else losses[-1]),
        "train": train_metrics,
        "validation": validation_metrics,
        "checkpoint": str(checkpoint),
        "vram_allocated_mb": (torch.cuda.max_memory_allocated() / 2**20
                              if device.type == "cuda" else 0.0),
    }
    report_path = args.output_dir / "multimotion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("G1_TEACHER_MULTIMOTION=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
