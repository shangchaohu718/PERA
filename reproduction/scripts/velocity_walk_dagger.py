#!/usr/bin/env python3
"""Validate a released velocity teacher, then distill one new Holo FFN/router."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

from pera.paths import ROOT


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def evaluate(task, actor=None, output=None, video=False, seed=7002):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from modumotion_g1.domain_moe import set_domain_context
    from modumotion_g1.velocity_walking import DOMAIN, EVAL_SEGMENTS

    task.reset(seed, actor)
    active = torch.ones(task.N, dtype=torch.bool, device=task.device)
    failed_at = [None] * task.N
    rows = []
    writer = (
        imageio.get_writer(str(output / "rollout.mp4"), fps=25, codec="libx264") if video else None
    )
    step = 0
    try:
        with torch.no_grad():
            for name, seconds, command in EVAL_SEGMENTS:
                if not active.any():
                    break
                task.command[:] = torch.tensor(command, device=task.device)
                errors, observed, weights = [], [], []
                for i in range(round(seconds / task.env.step_dt)):
                    if actor is None:
                        target = task.teacher_targets()
                    else:
                        td, context = task.student_inputs()
                        set_domain_context(actor, {DOMAIN: context})
                        mu = actor(td, mode="inference")["actions"]
                        set_domain_context(actor, None)
                        target = (
                            task.action.offset
                            + task.action.scale
                            * task.adapter.action_to_sim(mu, task.action.offset, task.action.scale)
                        )
                        weights.append(
                            float(actor.actor_module.layers[0].last_group_weights[..., -1].mean())
                        )
                    if writer and step % 2 == 0:
                        frame = Image.fromarray(task.env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 960, 75), fill=(20, 25, 35))
                        draw.text(
                            (15, 12),
                            "OFFICIAL VELOCITY TEACHER"
                            if actor is None
                            else "HOLO STUDENT / FROZEN PUBLIC + WALK EXPERT",
                            fill="white",
                        )
                        draw.text(
                            (15, 36),
                            f"{name} | cmd vx,vy,wz={command} | t={float(task.elapsed[0]):.2f}s | live physics, no reset",
                            fill="white",
                        )
                        writer.append_data(np.asarray(frame))
                    done = task.step_target(target)
                    p, actual, tilt = task.motion_metrics()
                    finite = torch.isfinite(actual).all(-1) & torch.isfinite(p).all(-1)
                    bad = done | (p[:, 2] < 0.45) | (tilt > 1.0) | ~finite
                    ended = active & bad
                    for env_id in ended.nonzero().flatten().tolist():
                        failed_at[env_id] = {
                            "segment": name,
                            "time_s": float(task.elapsed[env_id]),
                            "height_m": float(p[env_id, 2]),
                            "tilt_rad": float(tilt[env_id]),
                        }
                    active &= ~bad
                    if i * task.env.step_dt >= 1.0 and active.any():
                        errors.append((actual[active] - task.command[active]).cpu())
                        observed.append(actual[active].cpu())
                    step += 1
                    if done.any() or not active.all():
                        # Evaluation never resets or continues stepping a failed
                        # environment; preserve an incomplete command sequence.
                        break
                error = torch.cat(errors) if errors else None
                values = torch.cat(observed) if observed else None
                rows.append(
                    {
                        "segment": name,
                        "command": command,
                        "segment_completed": i + 1 == round(seconds / task.env.step_dt)
                        and bool(active.all()),
                        "steady_rmse_vx_vy_wz": error.square().mean(0).sqrt().tolist()
                        if error is not None
                        else None,
                        "steady_mean_vx_vy_wz": values.mean(0).tolist()
                        if values is not None
                        else None,
                        "mean_new_group_weight": sum(weights) / len(weights) if weights else None,
                    }
                )
                if not active.all():
                    break
    finally:
        if writer:
            writer.close()
    completed = len(rows) == len(EVAL_SEGMENTS) and all(r["segment_completed"] for r in rows)
    tracking = completed and all(
        max(r["steady_rmse_vx_vy_wz"][:2]) < 0.25 and r["steady_rmse_vx_vy_wz"][2] < 0.4
        for r in rows
    )
    stops = completed and all(
        sum(v * v for v in r["steady_mean_vx_vy_wz"][:2]) ** 0.5 < 0.15
        for r in rows
        if r["segment"] in ("stop", "final_stop")
    )
    return {
        "controller": "released_velocity_teacher"
        if actor is None
        else "HoloMotion_public_plus_velocity_walk",
        "elapsed_simulation_s": float(task.elapsed.max()),
        "num_envs": task.N,
        "sequence_complete": completed,
        "tracking_passed": tracking,
        "stop_passed": stops,
        "acceptance_passed": bool(completed and tracking and stops),
        "failed_at": failed_at,
        "segments": rows,
        "resets_after_start": 0,
        "teacher_actions_executed": actor is None,
        "video": str(output / "rollout.mp4") if video else None,
        "scope": "fixed flat plane; no domain-randomized or ten-task-course success claim",
    }


def collect(task, actor, features, *, seed, beta, seconds=10.0, fixed=False):
    import torch

    from modumotion_g1.domain_moe import set_domain_context
    from modumotion_g1.velocity_walking import DOMAIN

    task.reset(seed, actor)
    generator = torch.Generator(device=task.device).manual_seed(seed)
    shards, student_steps, teacher_steps, episodes_failed = [], 0, 0, 0
    for step in range(round(seconds / task.env.step_dt)):
        if step % 100 == 0:
            if fixed:
                commands = torch.tensor(
                    [
                        [0, 0, 0],
                        [0.6, 0, 0],
                        [0.35, 0, 0.5],
                        [0.35, 0, -0.5],
                        [0, 0.25, 0],
                        [0, -0.25, 0],
                        [-0.25, 0, 0],
                        [0.1, 0.1, 0],
                    ],
                    device=task.device,
                )
                task.command[:] = commands[
                    (torch.arange(task.N, device=task.device) + step // 100) % len(commands)
                ]
            else:
                task.command[:] = torch.rand(
                    task.N, 3, generator=generator, device=task.device
                ) * torch.tensor([1.0, 0.6, 1.2], device=task.device) - torch.tensor(
                    [0.3, 0.3, 0.6], device=task.device
                )
                stand = torch.rand(task.N, generator=generator, device=task.device) < 0.25
                task.command[stand] = 0
        with torch.no_grad():
            teacher_target = task.teacher_targets()
            labels = task.native_to_holo_target(teacher_target)
            td, context = task.student_inputs()
            set_domain_context(actor, {DOMAIN: context})
            mu = actor(td, mode="inference")["actions"]
            set_domain_context(actor, None)
            shards.append(features.snapshot(labels, mu, context=context))
            use_teacher = torch.rand(task.N, 1, generator=generator, device=task.device) < beta
            student_target = task.action.offset + task.action.scale * task.adapter.action_to_sim(
                mu, task.action.offset, task.action.scale
            )
            target = torch.where(use_teacher, teacher_target, student_target)
            teacher_steps += int(use_teacher.sum())
            student_steps += int((~use_teacher).sum())
            done = task.step_target(target)
            p, _, tilt = task.motion_metrics()
            bad = done | (p[:, 2] < 0.45) | (tilt > 1.0)
            if bad.any():
                ids = bad.nonzero().flatten()
                episodes_failed += len(ids)
                task.env.reset(env_ids=ids)
                task.elapsed[ids] = 0
                task.last_action[ids] = 0
                task.teacher_last_action[ids] = 0
                actor.clear_env_cache(ids)
    bank = {key: torch.cat([x[key] for x in shards]) for key in shards[0]}
    return bank, {
        "samples": len(bank["target"]),
        "teacher_probability": beta,
        "teacher_steps": teacher_steps,
        "student_steps": student_steps,
        "failed_episodes": episodes_failed,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("teacher", "preflight", "train", "evaluate"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7002)
    parser.add_argument("--bc-updates", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.phase == "train" and args.teacher_report is None:
        parser.error("Training requires a passed --teacher-report")
    if args.phase == "evaluate" and args.plugin is None:
        parser.error("Evaluation requires --plugin")
    if min(args.num_envs, args.bc_updates, args.rounds, args.updates, args.batch_size) < 1:
        parser.error("Counts must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import torch

    from modumotion_g1.domain_moe import (
        add_domain,
        domain_parameters,
        load_domain,
        public_digest,
        save_domain,
        set_trainable_domain,
    )
    from modumotion_g1.feature_dagger import FrozenFeatureDagger
    from modumotion_g1.instinct_dagger import append_replay
    from modumotion_g1.official import load_official
    from modumotion_g1.depth_features import (
        file_digest,
        optimizer_step,
        tensor_module_digest,
    )
    from modumotion_g1.velocity_walking import (
        DOMAIN,
        RELEASE,
        VelocityWalkTask,
        context_contract,
    )

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.4)
    torch.manual_seed(args.seed)
    report = {
        "status": "initializing",
        "phase": args.phase,
        "arguments": vars(args),
        "actor_updates": 0,
        "uses_ppo": False,
        "uses_world_model": False,
        "public_frozen": True,
        "uses_vision_for_walk": False,
        "fixed_merge_bias": 0.0,
        "history": [],
        "teacher_onnx_sha256": file_digest(RELEASE / "exported/policy.onnx"),
        "teacher_deploy_sha256": file_digest(RELEASE / "params/deploy.yaml"),
        "context_contract": context_contract(),
    }
    sources = [
        "modumotion_g1/velocity_walking.py",
        "scripts/velocity_walk_dagger.py",
        "modumotion_g1/domain_moe.py",
        "modumotion_g1/feature_dagger.py",
        "modumotion_g1/perceptive_observations.py",
        "scripts/run_flex_supermimic.sh",
    ]
    report["source_sha256"] = {}
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        report["source_sha256"][name] = file_digest(ROOT / name)
    started = time.monotonic()
    task = features = actor = None

    def progress():
        report["elapsed_wall_s"] = time.monotonic() - started
        report["pytorch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        atomic_json(output / "report.json", report)

    try:
        progress()
        if args.phase == "train":
            gate = json.loads(args.teacher_report.read_text())
            if (
                gate["status"] != "completed"
                or not gate["evaluation"]["acceptance_passed"]
                or gate["teacher_onnx_sha256"] != report["teacher_onnx_sha256"]
            ):
                raise ValueError("Teacher closed-loop acceptance gate failed")
            if (
                gate["source_sha256"]["modumotion_g1/velocity_walking.py"]
                != report["source_sha256"]["modumotion_g1/velocity_walking.py"]
            ):
                raise ValueError("Teacher interface/physics changed since acceptance")
            report["teacher_report_sha256"] = file_digest(args.teacher_report)
        task = VelocityWalkTask(args.num_envs, args.device, args.video)
        teacher_hash = tensor_module_digest(task.teacher)
        report["teacher_tensor_sha256"] = teacher_hash
        report["onnx_torch_equivalence"] = task.teacher.verify_onnx()
        if args.phase == "teacher":
            report["status"] = "evaluating_teacher"
            progress()
            report["evaluation"] = evaluate(task, output=output, video=args.video, seed=args.seed)
            report["status"] = "completed"
            return
        actor, critic, _ = load_official(args.device)
        del critic
        if args.plugin:
            if load_domain(actor, args.plugin, **context_contract()) != DOMAIN:
                raise ValueError("Wrong walking plugin")
            set_trainable_domain(actor, DOMAIN)
        else:
            add_domain(actor, DOMAIN, initial_bias=0.0, **context_contract())
        report["public_sha256"] = actor.public_sha256
        initial = {
            name: p.detach().cpu().clone()
            for name, p in actor.named_parameters()
            if p.requires_grad
        }
        params = domain_parameters(actor)
        report["trainable_tensors"] = list(initial)
        report["trainable_parameters"] = sum(p.numel() for p in params)
        if len(params) != 4:
            raise ValueError("Expected exactly FFN gate_up/down and router weight/bias")
        if args.phase == "evaluate":
            set_trainable_domain(actor, None)
            report["evaluation"] = evaluate(task, actor, output, args.video, args.seed)
            report["status"] = "completed"
            return
        features = FrozenFeatureDagger(actor, DOMAIN)
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        bank = None
        for round_id in [-1] if args.phase == "preflight" else range(-1, args.rounds):
            for name, digest in report["source_sha256"].items():
                if file_digest(ROOT / name) != digest:
                    raise RuntimeError("Bound source changed while job was running")
            beta = 1.0 if round_id < 0 else max(0.0, 0.8 - 0.1 * round_id)
            report.update(status="collecting", round=round_id, teacher_probability=beta)
            progress()
            new, collection = collect(
                task,
                actor,
                features,
                seed=100 + round_id,
                beta=beta,
                seconds=0.5 if args.phase == "preflight" else 10,
                fixed=round_id < 0,
            )
            torch.save(new, output / f"collection_{round_id + 1:03d}.pt")
            bank = append_replay(bank, new, 60000)
            # The bounded feature bank fits alongside the frozen 409M Actor.
            # Transfer once per round, rather than doing many CPU gathers and
            # host/device copies on every small FFN update.
            training_bank = {k: v.to(args.device) for k, v in bank.items()}
            count = (
                64
                if args.phase == "preflight"
                else args.bc_updates
                if round_id < 0
                else args.updates
            )
            losses, update_started = [], time.monotonic()
            for step in range(count):
                indices = torch.randint(len(bank["target"]), (args.batch_size,), device=args.device)
                batch = {k: v[indices] for k, v in training_bank.items()}
                loss = (features.predict(batch) - batch["target"]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite imitation loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 100 == 0:
                    report.update(
                        status="training",
                        actor_updates=optimizer_step(optimizer),
                        last_loss=losses[-1],
                        updates_done_in_round=step + 1,
                    )
                    progress()
            del training_bank
            changed = {
                name: bool(not torch.equal(p.detach().cpu(), initial[name]))
                for name, p in actor.named_parameters()
                if name in initial
            }
            if not all(changed.values()) or any(not torch.isfinite(p).all() for p in params):
                raise RuntimeError("New tensor update/finite check failed")
            if (
                public_digest(actor) != actor.public_sha256
                or tensor_module_digest(task.teacher) != teacher_hash
            ):
                raise RuntimeError("Frozen public/teacher weights changed")
            if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):
                raise RuntimeError("Frozen parameters received gradients")
            plugin = output / f"round_{round_id + 1:03d}.domain.pt"
            save_domain(
                actor,
                DOMAIN,
                plugin,
                {
                    "method": "released_velocity_teacher_BC_then_online_DAgger",
                    "actual_adam_steps": optimizer_step(optimizer),
                    "teacher_onnx_sha256": report["teacher_onnx_sha256"],
                    "context_is_command_not_vision": True,
                },
            )
            row = {
                "round": round_id,
                "collection": collection,
                "actual_adam_steps": optimizer_step(optimizer),
                "actual_updates_this_round": count,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_mean": sum(losses) / len(losses),
                "update_seconds": time.monotonic() - update_started,
                "four_tensors_changed": changed,
                "plugin": str(plugin),
            }
            report["history"].append(row)
            report.update(actor_updates=optimizer_step(optimizer), latest_plugin=str(plugin))
            torch.save(
                {
                    "round": round_id,
                    "plugin": str(plugin),
                    "optimizer": optimizer.state_dict(),
                    "actual_adam_steps": optimizer_step(optimizer),
                    "public_sha256": actor.public_sha256,
                    "teacher_onnx_sha256": report["teacher_onnx_sha256"],
                    "history": report["history"],
                },
                output / "training_state.pt",
            )
            if args.phase == "train" and (round_id == -1 or (round_id + 1) % 5 == 0):
                report["status"] = "evaluating_student"
                progress()
                row["validation_evaluation"] = evaluate(task, actor, seed=7002)
            progress()
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        if features:
            features.close()
        if actor is not None:
            report["public_unchanged"] = public_digest(actor) == actor.public_sha256
        if task is not None:
            report["teacher_unchanged"] = tensor_module_digest(task.teacher) == report.get(
                "teacher_tensor_sha256"
            )
            task.env.close()
        progress()


if __name__ == "__main__":
    main()
