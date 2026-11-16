#!/usr/bin/env python3
"""Distill local HUSKY into one explicitly switchable frozen-public G1 plugin."""

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

from pera.paths import ROOT


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def evaluate(
    task,
    actor=None,
    *,
    seconds=12,
    seed=8101,
    turn=False,
    output=None,
    expert_enabled=True,
):
    import mujoco
    import numpy as np
    import torch

    from modumotion_g1.domain_moe import set_active_domains, set_domain_context
    from modumotion_g1.husky_skateboarding import DOMAIN

    task.reset(seed, actor, perturb=True)
    active = np.ones(task.N, bool)
    initial = task.metrics()["board_xy"].copy()
    low = np.full(task.N, np.inf)
    distances = np.zeros(task.N)
    contact_counts = np.zeros((task.N, 2))
    contact_samples = np.zeros(task.N)
    failures = [None] * task.N
    weights = []
    qpos, commands = [], []
    for step in range(round(seconds * 50)):
        task.command[:] = [1, 0.35 if turn and 6 <= step * 0.02 < 18 else 0]
        with torch.no_grad():
            if actor is None:
                target = task.teacher_targets()
            else:
                set_active_domains(actor, [DOMAIN] if expert_enabled else [])
                td, context = task.student_inputs()
                set_domain_context(actor, {DOMAIN: context})
                mu = actor(td, mode="inference")["actions"]
                set_domain_context(actor, None)
                weights.append(
                    float(actor.actor_module.layers[0].last_group_weights[..., -1].mean())
                    if expert_enabled
                    else 0.0
                )
                target = task.holo_to_target(mu)
            if not torch.isfinite(target).all():
                raise FloatingPointError("Nonfinite executed target")
            if output is not None and step % 2 == 0 and active[0]:
                qpos.append(task.data[0].qpos.copy())
                commands.append(task.command[0].copy())
            bad = task.step_target(target, active)
        m = task.metrics()
        low[active] = np.minimum(low[active], m["pelvis_height"][active])
        distances[active] = np.maximum(distances[active], m["board_distance"][active])
        if step % 5 == 0:
            contact_counts[active] += task.contact_flags()[active]
            contact_samples[active] += 1
        ended = active & bad
        for i in np.flatnonzero(ended):
            failures[i] = {
                "time_s": float(task.steps[i] * 0.02),
                "pelvis_height_m": float(m["pelvis_height"][i]),
                "tilt_rad": float(m["tilt"][i]),
                "pelvis_board_distance_m": float(m["board_distance"][i]),
            }
        active &= ~bad
        if not active.any():
            break
    displacement = np.linalg.norm(task.metrics()["board_xy"] - initial, axis=-1)
    fractions = contact_counts / np.maximum(contact_samples[:, None], 1)
    warnings = [[int(w.number) for w in d.warning] for d in task.data]
    successful = (
        active
        & (displacement >= 0.25 * seconds)
        & (fractions[:, 0] >= 0.1)
        & (fractions[:, 1] >= 0.1)
        & ~np.array([any(w) for w in warnings])
    )
    report = {
        "controller": "released_HUSKY_teacher"
        if actor is None
        else "frozen_HoloMotion_public_plus_husky_skateboard"
        if expert_enabled
        else "frozen_HoloMotion_public_expert_off",
        "expert_enabled": actor is not None and expert_enabled,
        "robot_physical_joints": 29 if task.wrists else 23,
        "seed": seed,
        "turn_command": turn,
        "requested_seconds": seconds,
        "episodes": task.N,
        "successful_episodes": int(successful.sum()),
        "acceptance_passed": bool(successful.all()),
        "physically_survived": active.tolist(),
        "episode_seconds": (task.steps * 0.02).tolist(),
        "failed_at": failures,
        "board_displacement_m": displacement.tolist(),
        "minimum_pelvis_height_m": low.tolist(),
        "maximum_pelvis_board_distance_m": distances.tolist(),
        "foot_board_contact_fraction": fractions[:, 0].tolist(),
        "wheel_ground_contact_fraction": fractions[:, 1].tolist(),
        "warning_counts": warnings,
        "mean_new_group_weight": sum(weights) / len(weights) if weights else None,
        "resets_after_start": 0,
        "teacher_actions_executed": actor is None,
        "acceptance": "full physical survival, board displacement >=0.25m/s*duration, foot-board and wheel-ground contact in >=10% samples, no MuJoCo warnings",
        "scope": "HUSKY flat dynamic-board scene; fixed initial on-board stance with small pose perturbations; not mounting, course handover or exact speed tracking",
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "physics.npz", qpos=qpos, commands=commands, fps=25)
        mujoco.mj_saveModel(task.model, str(output / "scene.mjb"))
        atomic_json(output / "report.json", report)
    return report


def collect(task, actor, features, *, seed, beta, seconds=12):
    import numpy as np
    import torch

    from modumotion_g1.domain_moe import set_active_domains, set_domain_context
    from modumotion_g1.husky_skateboarding import DOMAIN

    task.reset(seed, actor, perturb=True)
    generator = np.random.default_rng(seed)
    rows, teacher_steps, student_steps, failed = [], 0, 0, 0
    set_active_domains(actor, [DOMAIN])
    for step in range(round(seconds * 50)):
        if step % 300 == 0:
            task.command[:, 0] = generator.uniform(0.8, 1.2, task.N)
            task.command[:, 1] = generator.uniform(-0.25, 0.25, task.N)
            task.command[0] = [1, 0]
        with torch.no_grad():
            target = task.teacher_targets()
            td, context = task.student_inputs()
            set_domain_context(actor, {DOMAIN: context})
            mu = actor(td, mode="inference")["actions"]
            set_domain_context(actor, None)
            rows.append(features.snapshot(task.target_to_holo(target), mu, context=context))
            choose_teacher = generator.random(task.N) < beta
            choose = torch.tensor(choose_teacher, device=task.device)[:, None]
            executed = torch.where(choose, target, task.holo_to_target(mu))
            if not torch.isfinite(executed).all():
                raise FloatingPointError("Nonfinite collection target")
            bad = task.step_target(executed)
            teacher_steps += int(choose_teacher.sum())
            student_steps += int((~choose_teacher).sum())
            if bad.any():
                ids = np.flatnonzero(bad)
                failed += len(ids)
                task.reset(seed + 10000 + step, actor, ids=ids, perturb=True)
    bank = {k: torch.cat([row[k] for row in rows]) for k in rows[0]}
    return bank, {
        "seed": seed,
        "samples": len(bank["target"]),
        "teacher_probability": beta,
        "teacher_executed_transitions": teacher_steps,
        "student_executed_transitions": student_steps,
        "failed_training_episodes_reset": failed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("teacher", "preflight", "train", "evaluate"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        help="Load the unified shared-Actor expert bundle for evaluation",
    )
    parser.add_argument("--expert", choices=("on", "off"), default="on")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--bc-updates", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if args.phase == "train" and args.teacher_report is None:
        parser.error("A passed teacher report is required")
    if args.phase == "evaluate" and args.plugin is None and args.registry is None:
        parser.error("Evaluation requires --plugin or --registry")
    if args.phase != "evaluate" and (args.registry is not None or args.expert == "off"):
        parser.error("--registry and --expert off are evaluation options")
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
        set_active_domains,
        set_trainable_domain,
    )
    from modumotion_g1.feature_dagger import FrozenFeatureDagger
    from modumotion_g1.husky_skateboarding import (
        DOMAIN,
        HUSKY,
        HuskySkateTask,
        context_contract,
        digest,
    )
    from modumotion_g1.official import load_official
    from modumotion_g1.depth_features import optimizer_step, tensor_module_digest

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.28)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    report = {
        "status": "initializing",
        "arguments": vars(args),
        "actor_updates": 0,
        "history": [],
        "public_frozen": True,
        "frozen_husky_encoder": True,
        "uses_vision": False,
        "method": "BC_then_online_DAgger",
        "uses_ppo": False,
        "uses_world_model": False,
        "fixed_merge_bias": 0.0,
        "soft_residual_cap": False,
        "new_domain": DOMAIN,
        "teacher_onnx_sha256": digest(HUSKY / "ckpts/test.onnx"),
        "upstream_scene_sha256": digest(HUSKY / "test_scene/mjlab_scene.xml"),
        "source_sha256": {},
        "context_contract": context_contract(),
        "actual_adam_budget": 64
        if args.phase == "preflight"
        else args.bc_updates + args.rounds * args.updates,
    }
    for name in (
        "modumotion_g1/husky_skateboarding.py",
        "scripts/husky_skateboard_dagger.py",
        "modumotion_g1/domain_moe.py",
        "modumotion_g1/feature_dagger.py",
        "modumotion_g1/perceptive_observations.py",
    ):
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        report["source_sha256"][name] = digest(ROOT / name)
    if args.registry:
        name = "modumotion_g1/expert_registry.py"
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        report["source_sha256"][name] = digest(ROOT / name)
        report["registry_sha256"] = digest(args.registry)
        shutil.copy2(args.registry, output / "source_snapshot/expert_registry.json")
    actor = features = task = None

    def save():
        report["elapsed_wall_s"] = time.monotonic() - started
        if args.device.startswith("cuda"):
            report["pytorch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        atomic_json(output / "report.json", report)

    try:
        save()
        task = HuskySkateTask(args.num_envs, args.device)
        report["onnx_torch_equivalence"] = task.teacher.verify_onnx()
        teacher_hash = tensor_module_digest(task.teacher)
        report["frozen_teacher_tensor_sha256"] = teacher_hash
        if args.phase == "teacher":
            report["evaluations"] = [
                evaluate(
                    task,
                    seconds=args.seconds,
                    seed=args.seed + i,
                    turn=bool(i),
                    output=output / f"eval_{i}",
                )
                for i in range(2)
            ]
            report["acceptance_passed"] = all(
                row["acceptance_passed"] for row in report["evaluations"]
            )
            report["status"] = "completed"
            return
        if args.phase == "train":
            teacher_gate = json.loads(args.teacher_report.read_text())
            if (
                not teacher_gate.get("acceptance_passed")
                or teacher_gate["teacher_onnx_sha256"] != report["teacher_onnx_sha256"]
            ):
                raise ValueError("Teacher acceptance failed or checkpoint differs")
            if (
                teacher_gate["source_sha256"]["modumotion_g1/husky_skateboarding.py"]
                != report["source_sha256"]["modumotion_g1/husky_skateboarding.py"]
            ):
                raise ValueError("Teacher scene/observation implementation changed")
            report["teacher_report_sha256"] = digest(args.teacher_report)
        actor, critic, _ = load_official(args.device)
        del critic
        released_digest = public_digest(actor)
        # The release updates observation statistics by default even in
        # inference/eval mode. Install the frozen wrapper BEFORE any forward.
        install_domains(actor)
        report["released_public_sha256_before_any_forward"] = released_digest
        task.reset(args.seed, actor)
        baseline_td, _ = task.student_inputs()
        # last_action is a live environment buffer. The OFF comparison needs
        # immutable inputs from before collection, not that buffer's later value.
        baseline_td = baseline_td.clone()
        with torch.no_grad():
            baseline = actor(baseline_td, mode="inference")["actions"].clone()
        if public_digest(actor) != released_digest:
            raise RuntimeError("Released public state changed during baseline inference")
        if args.registry:
            from modumotion_g1.expert_registry import load_expert_registry

            registry = load_expert_registry(actor, args.registry)
            if args.plugin and digest(args.plugin) != registry.experts[DOMAIN]["plugin_sha256"]:
                raise ValueError("Requested plugin differs from the registry")
            set_trainable_domain(actor, DOMAIN)
            report["loaded_domains"] = list(actor.actor_module.layers[0].domain_plugins)
            report["loaded_domain_count"] = len(registry.names)
        elif args.plugin:
            if load_domain(actor, args.plugin, **context_contract()) != DOMAIN:
                raise ValueError("Wrong plugin")
            set_trainable_domain(actor, DOMAIN)
        else:
            add_domain(actor, DOMAIN, initial_bias=0.0, **context_contract())
        report["public_sha256"] = actor.public_sha256
        initial = {
            n: p.detach().cpu().clone() for n, p in actor.named_parameters() if p.requires_grad
        }
        params = domain_parameters(actor)
        if len(params) != 4:
            raise ValueError("Expected exactly four trainable FFN/router tensors")
        report["trainable_tensors"] = list(initial)
        report["trainable_parameters"] = sum(p.numel() for p in params)
        if args.phase == "evaluate":
            set_trainable_domain(actor, None)
            report["evaluations"] = [
                evaluate(
                    task,
                    actor,
                    seconds=args.seconds,
                    seed=args.seed + i,
                    turn=bool(i),
                    output=output / f"eval_{i}",
                    expert_enabled=args.expert == "on",
                )
                for i in range(2)
            ]
            report["acceptance_passed"] = all(
                row["acceptance_passed"] for row in report["evaluations"]
            )
        else:
            features = FrozenFeatureDagger(actor, DOMAIN)
            optimizer = torch.optim.Adam(params, lr=1e-4)
            bank = None
            rounds = [-1] if args.phase == "preflight" else range(-1, args.rounds)
            for round_id in rounds:
                for name, expected in report["source_sha256"].items():
                    if digest(ROOT / name) != expected:
                        raise RuntimeError(f"Bound source changed: {name}")
                beta = 1.0 if round_id < 0 else max(0.0, 0.8 - 0.1 * round_id)
                report.update(status="collecting", round=round_id, teacher_probability=beta)
                save()
                new, collection = collect(
                    task,
                    actor,
                    features,
                    seed=1000 + round_id,
                    beta=beta,
                    seconds=0.5 if args.phase == "preflight" else 12,
                )
                torch.save(new, output / f"collection_{round_id + 1:03d}.pt")
                bank = (
                    new
                    if bank is None
                    else {k: torch.cat((bank[k], new[k]))[-60000:] for k in bank}
                )
                gpu_bank = {k: v.to(args.device) for k, v in bank.items()}
                count = (
                    64
                    if args.phase == "preflight"
                    else args.bc_updates
                    if round_id < 0
                    else args.updates
                )
                update_start, losses = time.monotonic(), []
                report["status"] = "training"
                for step in range(count):
                    ids = torch.randint(len(bank["target"]), (args.batch_size,), device=args.device)
                    batch = {k: v[ids] for k, v in gpu_bank.items()}
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
                            actor_updates=optimizer_step(optimizer),
                            last_loss=losses[-1],
                            updates_done_in_round=step + 1,
                        )
                        save()
                del gpu_bank
                changed = {
                    n: not torch.equal(p.detach().cpu(), initial[n])
                    for n, p in actor.named_parameters()
                    if n in initial
                }
                if not all(changed.values()) or any(not torch.isfinite(p).all() for p in params):
                    raise RuntimeError("Four-tensor update/finite check failed")
                if (
                    public_digest(actor) != actor.public_sha256
                    or tensor_module_digest(task.teacher) != teacher_hash
                ):
                    raise RuntimeError("Frozen public or teacher changed")
                if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen public received gradients")
                plugin = output / f"round_{round_id + 1:03d}.domain.pt"
                save_domain(
                    actor,
                    DOMAIN,
                    plugin,
                    {
                        "method": "HUSKY_BC_then_online_DAgger",
                        "actual_adam_steps": optimizer_step(optimizer),
                        "teacher_onnx_sha256": report["teacher_onnx_sha256"],
                        "six_wrist_teacher_targets": "zero",
                        "uses_vision": False,
                    },
                )
                row = {
                    "round": round_id,
                    "collection": collection,
                    "actual_adam_steps": optimizer_step(optimizer),
                    "updates_this_round": count,
                    "loss_first": losses[0],
                    "loss_last": losses[-1],
                    "mean_loss": sum(losses) / len(losses),
                    "update_seconds": time.monotonic() - update_start,
                    "four_tensors_changed": changed,
                    "plugin": str(plugin),
                }
                report["history"].append(row)
                report.update(actor_updates=optimizer_step(optimizer), latest_plugin=str(plugin))
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "round": round_id,
                        "plugin": str(plugin),
                        "actual_adam_steps": optimizer_step(optimizer),
                        "public_sha256": actor.public_sha256,
                    },
                    output / "training_state.pt",
                )
                if args.phase == "train" and (round_id == -1 or (round_id + 1) % 5 == 0):
                    report["status"] = "evaluating_student"
                    save()
                    row["validation"] = evaluate(task, actor, seed=9001)
                save()
            features.close()
            features = None
        # Explicit OFF must reproduce the released public output, including after training.
        set_active_domains(actor, [])
        actor.reset_kv_cache(task.N, task.device)
        actor.clear_env_cache(None)
        with torch.no_grad():
            disabled = actor(baseline_td, mode="inference")["actions"]
        torch.testing.assert_close(disabled, baseline, atol=0, rtol=0)
        report.update(
            status="completed",
            expert_off_public_bitwise_equal=True,
            four_tensors_changed={
                n: not torch.equal(p.detach().cpu(), initial[n])
                for n, p in actor.named_parameters()
                if n in initial
            },
        )
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if features:
            features.close()
        if actor is not None:
            report["public_unchanged"] = public_digest(actor) == actor.public_sha256
        if task is not None:
            report["teacher_encoder_unchanged"] = (
                tensor_module_digest(task.teacher) == task.teacher_digest
            )
        save()


if __name__ == "__main__":
    main()
