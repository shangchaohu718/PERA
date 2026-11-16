"""Native parkour teacher acceptance and independently trained FFN/router plugins."""

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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def rollout(task, actor=None, *, seed, seconds=20, features=None, beta=0.0, recording=None):
    import numpy as np
    import torch

    from modumotion_g1.domain_moe import set_domain_context

    task.reset(seed, actor)
    domain = "parkour_" + task.task
    active = torch.ones(task.N, dtype=torch.bool, device=task.device)
    successful = torch.zeros_like(active)
    near_steps = torch.zeros(task.N, dtype=torch.long, device=task.device)
    student_steps = teacher_steps = 0
    rows, episodes, poses, depths, commands, weights = (
        [],
        [None] * task.N,
        [],
        [],
        [],
        [],
    )
    start_p = task.metrics()[0].clone()
    minimum_distance = task.metrics()[2].clone()
    max_height = start_p[:, 2].clone()
    recording_finished = False
    for step in range(round(seconds / task.env.step_dt)):
        valid = active.clone()
        teacher_action = task.teacher_actions() if features is not None or actor is None else None
        if actor is not None:
            td, context = task.student_inputs()
            set_domain_context(actor, {domain: context})
            student_action = actor(td, mode="inference")["actions"]
            set_domain_context(actor, None)
            native_student = task.holo_to_native(student_action)
            if features is not None:
                row = features.snapshot(
                    task.native_to_holo(teacher_action), student_action, context
                )
                rows.append({k: v[valid.cpu()] for k, v in row.items()})
            block = actor.actor_module.layers[0]
            weights.append(
                float(block.last_group_weights[valid, ..., -1].mean())
                if block.active_domains
                else 0.0
            )
            if features is not None:
                choose = torch.rand(task.N, 1, device=task.device) < beta
                action = torch.where(choose, teacher_action, native_student)
                student_steps += int((valid & ~choose[:, 0]).sum())
                teacher_steps += int((valid & choose[:, 0]).sum())
            else:
                action = native_student
                student_steps += int(valid.sum())
        else:
            action = teacher_action
            teacher_steps += int(valid.sum())
        if recording is not None and not recording_finished and step % 2 == 0:
            poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
            depths.append(
                task.obs[0, task.depth_slice].reshape(*task.depth_shape)[-1].cpu().numpy().copy()
            )
            commands.append(task.command.command[0].cpu().numpy().copy())
        done = task.step(action)
        p, tilt, distance = task.metrics()
        finite = torch.isfinite(p).all(-1) & torch.isfinite(tilt)
        minimum_distance[valid] = torch.minimum(minimum_distance[valid], distance[valid])
        max_height[valid] = torch.maximum(max_height[valid], p[valid, 2])
        tm = task.env.termination_manager
        physical_failure = (tilt > 1.0) | ~finite
        fired = {}
        for name in (
            "base_contact",
            "bad_orientation",
            "terrain_out_bound",
            "time_out",
        ):
            fired[name] = tm.get_term(name).bool().clone()
        physical_failure |= (
            fired["base_contact"] | fired["bad_orientation"] | fired["terrain_out_bound"]
        )
        near_steps = torch.where((distance <= 0.4) & ~physical_failure, near_steps + 1, 0)
        goal = near_steps >= round(1.0 / task.env.step_dt)
        end = valid & (done | physical_failure | goal)
        successful |= end & goal & ~physical_failure
        last = step + 1 == round(seconds / task.env.step_dt)
        for env_id in (end | (valid & last)).nonzero().flatten().tolist():
            episodes[env_id] = {
                "env_id": env_id,
                "seed": seed,
                "success": bool(successful[env_id]),
                "physical_failure": bool(physical_failure[env_id]),
                "physical_survival_to_horizon": bool(last and not physical_failure[env_id]),
                "elapsed_s": float(task.elapsed[env_id]),
                "start_pelvis_xyz": start_p[env_id].tolist(),
                "final_pelvis_xyz": p[env_id].tolist(),
                "goal_ground_xyz": task.route_goal[env_id].tolist(),
                "progress_x_m": float(p[env_id, 0] - start_p[env_id, 0]),
                "minimum_goal_distance_m": float(minimum_distance[env_id]),
                "final_goal_distance_m": float(distance[env_id]),
                "maximum_pelvis_height_m": float(max_height[env_id]),
                "fired": {n: bool(v[env_id]) for n, v in fired.items()},
                "resets_before_episode_end": 0,
            }
        if end[0]:
            recording_finished = True
        active &= ~end
        if not active.any() or last:
            break
        if features is not None:
            # DAgger continues collecting on fresh episodes, not through falls.
            ids = (done | physical_failure | goal).nonzero().flatten()
            if len(ids):
                task.reset(seed, actor, env_ids=ids)
                active[ids] = True
        else:
            # Finished environments are reset solely to service the vector API;
            # their new episodes are excluded from every metric and recording.
            ids = done.nonzero().flatten()
            if len(ids):
                task.reset(seed, actor, env_ids=ids)
    if recording is not None:
        import mujoco

        recording.mkdir(parents=True, exist_ok=False)
        mujoco.mj_saveModel(task.cpu_model, str(recording / "scene.mjb"))
        np.savez_compressed(
            recording / "physics.npz",
            qpos=np.asarray(poses),
            depth=np.asarray(depths),
            commands=np.asarray(commands),
            fps=25,
        )
    report = {
        "task": task.task,
        "controller": "official_parkour_teacher"
        if actor is None
        else "frozen_HoloMotion_public_plus_parkour_FFN_router",
        "episodes": episodes,
        "successes": sum(bool(r and r["success"]) for r in episodes),
        "num_episodes": task.N,
        "mean_new_group_weight": float(np.mean(weights)) if weights else None,
        "student_controlled_env_steps": student_steps,
        "teacher_controlled_env_steps": teacher_steps,
        "actual_physics": True,
        "evaluates_first_episode_only": features is None,
        "horizon_s": seconds,
        "success_definition": "Reach actual ground goal within 0.4m and remain physically valid for 1s; no reset in evaluated episode",
    }
    if features is None:
        report["acceptance_passed"] = report["successes"] / task.N >= 0.9
    new = {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None
    return report, new


def main(
    *,
    task_class=None,
    rollout_function=None,
    source_additions=(),
    task_names=None,
    teacher_decision_function=None,
):
    from modumotion_g1.parkour_distillation import TASKS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=TASKS if task_names is None else task_names, required=True
    )
    parser.add_argument(
        "--phase", choices=("teacher", "preflight", "train", "evaluate"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path)
    parser.add_argument("--teacher-authorization", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--expert", choices=("on", "off"), default="on")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--bc-updates", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "train" and not args.teacher_report:
        parser.error("Training requires a native teacher report")
    if args.teacher_authorization and args.phase != "train":
        parser.error("Teacher authorization is only for training")
    if args.plugin and args.phase != "evaluate":
        parser.error("--plugin is for evaluation only")
    if args.expert == "off" and args.phase != "evaluate":
        parser.error("Disable experts only during evaluation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import torch

    from modumotion_g1.domain_moe import (
        add_domain,
        domain_parameters,
        install_domains,
        load_domain,
        public_digest,
        save_domain,
        set_domain_enabled,
        set_trainable_domain,
    )
    from modumotion_g1.feature_dagger import FrozenFeatureDagger
    from modumotion_g1.instinct_dagger import append_replay
    from modumotion_g1.official import load_official
    from modumotion_g1.parkour_distillation import PUBLIC_SHA, RELEASE, ParkourTask
    from modumotion_g1.depth_features import file_digest, optimizer_step

    task_class = ParkourTask if task_class is None else task_class
    sample_rollout = rollout if rollout_function is None else rollout_function

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45)
    torch.manual_seed(args.seed)
    report = {
        "status": "initializing",
        "arguments": vars(args),
        "actor_updates": 0,
        "history": [],
        "method": "BC_then_online_DAgger",
        "public_frozen": True,
        "vision_encoder_frozen": True,
        "uses_ppo": False,
        "uses_world_model": False,
        "fixed_merge_bias": 0.0,
        "teacher_checkpoint_sha256": file_digest(RELEASE / "model_57000.pt"),
    }
    started = time.monotonic()
    task = actor = features = None
    sources = [
        "modumotion_g1/parkour_distillation.py",
        "scripts/parkour_dagger.py",
        "modumotion_g1/teacher_authorization.py",
        "modumotion_g1/domain_moe.py",
        "modumotion_g1/feature_dagger.py",
        "modumotion_g1/perceptive_observations.py",
        *source_additions,
    ]
    report["source_sha256"] = {}
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        report["source_sha256"][name] = file_digest(ROOT / name)

    def progress():
        report["elapsed_wall_s"] = time.monotonic() - started
        report["pytorch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        atomic_json(output / "report.json", report)

    try:
        progress()
        if args.phase == "train":
            gate = json.loads(args.teacher_report.read_text())
            from modumotion_g1.teacher_authorization import teacher_training_decision

            authorization = (
                json.loads(args.teacher_authorization.read_text())
                if args.teacher_authorization
                else None
            )
            decide = teacher_decision_function or teacher_training_decision
            report["teacher_training"] = decide(
                gate,
                task=args.task,
                checkpoint_sha256=report["teacher_checkpoint_sha256"],
                authorization=authorization,
            )
            report["teacher_report_sha256"] = file_digest(args.teacher_report)
            for name in (
                "modumotion_g1/parkour_distillation.py",
                "scripts/parkour_dagger.py",
                *source_additions,
            ):
                if gate["source_sha256"][name] != report["source_sha256"][name]:
                    raise ValueError("Teacher acceptance used different task/interface code")
        task = task_class(args.task, args.num_envs, args.device)
        report.update(
            context_contract=task.contract(),
            teacher_tensor_sha256=task.teacher_sha,
            normalizer_sha256=task.normalizer_sha,
            encoder_sha256=task.encoder_sha,
        )
        if args.phase == "teacher":
            report["status"] = "evaluating_teacher"
            progress()
            with torch.no_grad():
                report["evaluation"], _ = sample_rollout(
                    task,
                    seed=args.seed,
                    seconds=args.seconds,
                    recording=output / "recording" if args.record else None,
                )
            report["status"] = "completed"
            return
        actor, critic, _ = load_official(args.device)
        del critic
        install_domains(actor)  # Freeze all public statistics BEFORE any forward.
        if actor.public_sha256 != PUBLIC_SHA:
            raise ValueError("Public Actor differs from the exact official G1 v1.4 release")
        domain = "parkour_" + args.task
        if args.plugin:
            if load_domain(actor, args.plugin, **task.contract()) != domain:
                raise ValueError("Wrong scene plugin")
        else:
            add_domain(actor, domain, initial_bias=0.0, **task.contract())
        report["public_sha256"] = actor.public_sha256
        if args.phase == "evaluate":
            set_trainable_domain(actor, None)
            set_domain_enabled(actor, domain, args.expert == "on")
            with torch.no_grad():
                report["evaluation"], _ = sample_rollout(
                    task,
                    actor,
                    seed=args.seed,
                    seconds=args.seconds,
                    recording=output / "recording" if args.record else None,
                )
            report["status"] = "completed"
            return
        set_trainable_domain(actor, domain)
        params = domain_parameters(actor)
        initial = {
            n: p.detach().cpu().clone() for n, p in actor.named_parameters() if p.requires_grad
        }
        if len(params) != 4 or len(initial) != 4:
            raise ValueError("Expected exactly four trainable FFN/router tensors")
        report.update(
            trainable_tensors=list(initial),
            trainable_parameters=sum(p.numel() for p in params),
        )
        features = FrozenFeatureDagger(actor, domain)
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        bank = None
        for round_id in [-1] if args.phase == "preflight" else range(-1, args.rounds):
            for name, digest in report["source_sha256"].items():
                if file_digest(ROOT / name) != digest:
                    raise RuntimeError("Bound source changed while training")
            beta = 1.0 if round_id < 0 else max(0.0, 0.8 - 0.1 * round_id)
            report.update(status="collecting", round=round_id, teacher_probability=beta)
            progress()
            with torch.no_grad():
                collection, new = sample_rollout(
                    task,
                    actor,
                    seed=13000 + round_id,
                    seconds=0.5 if args.phase == "preflight" else args.seconds,
                    features=features,
                    beta=beta,
                )
            bank = append_replay(bank, new, 60000)
            training_bank = {k: v.to(args.device) for k, v in bank.items()}
            count = (
                64
                if args.phase == "preflight"
                else args.bc_updates
                if round_id < 0
                else args.updates
            )
            losses = []
            for step in range(count):
                idx = torch.randint(len(bank["target"]), (args.batch_size,), device=args.device)
                batch = {k: v[idx] for k, v in training_bank.items()}
                loss = (features.predict(batch) - batch["target"]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite supervised loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 100 == 0:
                    report.update(
                        status="training",
                        actor_updates=optimizer_step(optimizer),
                        loss=losses[-1],
                    )
                    progress()
            del training_bank
            changed = {
                n: not torch.equal(p.detach().cpu(), initial[n])
                for n, p in actor.named_parameters()
                if n in initial
            }
            if not all(changed.values()) or any(not torch.isfinite(p).all() for p in params):
                raise RuntimeError("A new tensor did not update or became nonfinite")
            if public_digest(actor) != PUBLIC_SHA or any(
                p.grad is not None for p in actor.parameters() if not p.requires_grad
            ):
                raise RuntimeError("Frozen public Actor changed or received gradients")
            task.assert_frozen()
            plugin = output / f"round_{round_id + 1:03d}.domain.pt"
            save_domain(
                actor,
                domain,
                plugin,
                {
                    "method": report["method"],
                    "teacher_training": report.get("teacher_training"),
                    "actual_adam_steps": optimizer_step(optimizer),
                    "teacher_checkpoint_sha256": report["teacher_checkpoint_sha256"],
                },
            )
            report["history"].append(
                {
                    "round": round_id,
                    "actual_adam_steps": optimizer_step(optimizer),
                    "loss_first": losses[0],
                    "loss_last": losses[-1],
                    "loss_mean": sum(losses) / len(losses),
                    "four_tensors_changed": changed,
                    "collection": collection,
                }
            )
            report.update(actor_updates=optimizer_step(optimizer), latest_plugin=str(plugin))
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "actual_adam_steps": optimizer_step(optimizer),
                    "round": round_id,
                    "public_sha256": PUBLIC_SHA,
                    "plugin": str(plugin),
                },
                output / "training_state.pt",
            )
            progress()
        report["status"] = "completed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if features:
            features.close()
        if actor is not None:
            report["public_unchanged"] = public_digest(actor) == PUBLIC_SHA
        if task:
            task.assert_frozen()
            report["teacher_and_encoder_unchanged"] = True
            task.env.close()
        progress()


if __name__ == "__main__":
    main()
