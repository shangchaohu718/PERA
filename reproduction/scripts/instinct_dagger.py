#!/usr/bin/env python3
"""Distill the official InstinctMJ task into a frozen HoloMotion scene FFN."""

from __future__ import annotations
import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

from pera.paths import ROOT

TEACHER = (
    ROOT
    / "checkpoints/instinct_supermimic_v0.1"
    / "g1_perceptive_shadowing_supermimic_model_8000.pt"
)
SOURCE = (
    ROOT / "data/instinct_supermimic/20251116_50cm_kneeClimbStep1" / "50cm_kneeClimbStep_noWall"
)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def prepare_dataset(task_id="knee_climb_no_wall"):
    """Copy one unmodified official task; never adapt its motion or mesh."""
    from modumotion_g1.instinct_tasks import (
        DEFAULT_TASK,
        prepare_task_dataset,
    )

    if task_id != DEFAULT_TASK:
        return prepare_task_dataset(task_id)
    import hashlib

    import yaml

    destination = ROOT / "data/instinct_climb_dagger_v1"
    destination.mkdir(exist_ok=True)
    original = SOURCE / "metadata_kneelClimbStep1.yaml"
    meta = yaml.safe_load(original.read_text())
    files = [meta["motion_files"][0]["motion_file"]]
    files += [meta["terrains"][0]["terrain_file"]]
    hashes = {}
    for name in files:
        source, target = SOURCE / name, destination / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("Official task copy changed")
        else:
            shutil.copy2(source, target)
        hashes[name] = digest
    text = original.read_text()
    target = destination / "metadata.yaml"
    if target.exists() and target.read_text() != text:
        raise ValueError("Official metadata copy changed")
    target.write_text(text)
    atomic_json(
        destination / "manifest.json",
        {
            "source": str(SOURCE),
            "release": "InstinctMJ_supermimic_v0.1",
            "task": "official_50cm_knee_climb_no_wall",
            "files_sha256": hashes,
            "motion_and_mesh_modified": False,
            "uses_pmt_tcrs": False,
            "student_reference": "official_teacher_task_reference",
        },
    )
    return destination


def main():
    from modumotion_g1.instinct_tasks import (
        DEFAULT_TASK,
        TASKS,
        get_task,
        teacher_quality_acceptable,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[name for name, _ in TASKS], default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("teacher", "preflight", "train", "evaluate"),
        default="preflight",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--bc-updates", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--plugin", type=Path)
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument(
        "--expert",
        choices=("on", "off"),
        help="Expert on/off; off bypasses new router/FFN (default: on)",
    )
    switch.add_argument("--public-only", action="store_true")
    parser.add_argument(
        "--teacher-report",
        type=Path,
        default=ROOT / "reports/instinct_dagger_v1/teacher/report.json",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-partial-teacher",
        action="store_true",
        help="Allow a partly successful teacher; retain actual failures",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    task_spec = get_task(args.task)
    if min(args.num_envs, args.rounds, args.updates, args.bc_updates) < 1:
        parser.error("Counts must be positive")
    if args.plugin and args.phase != "evaluate":
        parser.error("--plugin is only for evaluation")
    if args.resume and args.phase != "train":
        parser.error("--resume is only for training")
    if args.allow_partial_teacher and args.phase != "train":
        parser.error("--allow-partial-teacher is only for training")
    if args.public_only and args.phase != "evaluate":
        parser.error("--public-only is only for evaluation")
    if args.expert is not None and args.phase != "evaluate":
        parser.error("--expert is only for evaluation")
    if args.expert == "on" and args.plugin is None:
        parser.error("--expert on requires a trained --plugin")
    if min(args.batch_size, args.replay_capacity, args.eval_interval) < 1:
        parser.error("Batch, replay capacity and evaluation interval must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import torch

    from modumotion_g1.domain_moe import (
        add_domain,
        domain_parameters,
        load_domain,
        public_digest,
        save_domain,
        set_active_domains,
        set_domain_enabled,
        set_trainable_domain,
    )
    from modumotion_g1.feature_dagger import FrozenFeatureDagger
    from modumotion_g1.instinct_dagger import NativeTeacherTask, append_replay
    from modumotion_g1.official import load_official
    from modumotion_g1.depth_features import (
        file_digest,
        optimizer_step,
        tensor_module_digest,
    )

    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(0.4, args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    task, features, failure = None, None, None
    report = {
        "status": "initializing",
        "method": "BC_then_online_DAgger",
        "arguments": vars(args),
        "actor_updates": 0,
        "uses_world_model": False,
        "uses_ppo": False,
        "uses_tcrs": False,
        "fixed_merge_bias": 0.0,
        "uses_soft_residual_cap": False,
        "allow_partial_teacher": args.allow_partial_teacher,
        "public_frozen": True,
        "scene": task_spec["scene"],
        "task_id": args.task,
        "domain": task_spec["domain"],
        "student_reference": "official_teacher_task_reference",
        "teacher_checkpoint": str(TEACHER),
        "teacher_checkpoint_sha256": file_digest(TEACHER),
        "history": [],
    }
    source_files = [
        *sorted((ROOT / "modumotion_g1").glob("*.py")),
        ROOT / "scripts/instinct_dagger.py",
        ROOT / "scripts/run_instinct_dagger_pipeline.py",
        ROOT / "scripts/run_instinct_task_suite.py",
        ROOT / "scripts/run_flex_supermimic.sh",
        ROOT / "configs/g1_actor_io_contract.json",
    ]
    source_hashes = {}
    for source in source_files:
        relative = source.relative_to(ROOT)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = file_digest(source)
    report["source_sha256"] = source_hashes

    def progress():
        report["elapsed_s"] = time.monotonic() - started
        report["pytorch_peak_allocated_mib"] = torch.cuda.max_memory_allocated(args.device) / 2**20
        atomic_json(output / "report.json", report)

    def save_video(frames, name):
        if frames:
            import imageio.v2 as imageio

            path = output / name
            imageio.mimwrite(path, frames, fps=50, codec="libx264")
            return str(path)
        return None

    try:
        folder = prepare_dataset(args.task)
        report["dataset_manifest_sha256"] = file_digest(folder / "manifest.json")
        if args.phase == "train":
            teacher_report = json.loads(args.teacher_report.read_text())
            if (
                teacher_report.get("status") != "completed"
                or not teacher_quality_acceptable(
                    teacher_report["teacher_eval"], args.allow_partial_teacher
                )
                or teacher_report["teacher_checkpoint_sha256"]
                != report["teacher_checkpoint_sha256"]
                or teacher_report["dataset_manifest_sha256"] != report["dataset_manifest_sha256"]
            ):
                raise ValueError("Official teacher reproduction gate failed")
            report["teacher_reproduction_report_sha256"] = file_digest(args.teacher_report)
            report["teacher_reproduction_eval"] = teacher_report["teacher_eval"]
        progress()
        task = NativeTeacherTask(folder, TEACHER, args.num_envs, args.device, args.video)
        encoder_hash = tensor_module_digest(task.depth)
        teacher_hash = tensor_module_digest(task.teacher_actor)
        report.update(encoder_sha256=encoder_hash, teacher_sha256=teacher_hash)
        # Validate the extracted encoder against its native Actor's branch.
        obs = task.reset(args.seed)
        with torch.no_grad():
            expected = task.teacher_actor.encoders._parallel_blocks["depth_image"](
                task.teacher_norm(obs)[:, task.depth_slice].reshape(-1, 1, 18, 32)
            )
            actual = task.depth(obs[:, task.depth_slice])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report["native_encoder_feature_bitwise_equal"] = True
        if args.phase == "teacher":
            result, _, frames = task.rollout(controller="teacher", seed=args.seed, video=args.video)
            report.update(
                status="completed",
                teacher_eval=result,
                video=save_video(frames, "teacher.mp4"),
            )
            return
        actor, critic, _ = load_official(args.device)
        del critic
        context = {
            "context_dim": 32,
            "context_schema": {
                "kind": "instinct_released_depth_features",
                "depth_shape": [1, 18, 32],
                "normalization": "released_policy_normalizer",
                "frozen": True,
            },
            "encoder_sha256": encoder_hash,
        }
        domain = task_spec["domain"]
        restore = None
        if args.resume:
            restore = torch.load(args.resume, map_location="cpu", weights_only=True)
            for key in [
                "teacher_checkpoint_sha256",
                "encoder_sha256",
                "dataset_manifest_sha256",
            ]:
                if restore[key] != report[key]:
                    raise ValueError(f"Resume {key} differs")
            plugin = args.resume.parent / restore["plugin"]
        else:
            plugin = args.plugin
        if plugin:
            if load_domain(actor, plugin, **context) != domain:
                raise ValueError("Unexpected plugin domain")
            set_trainable_domain(actor, domain)
        else:
            add_domain(actor, domain, initial_bias=0.0, **context)
        set_active_domains(actor, [domain], {domain: 0.0})
        params = domain_parameters(actor)
        initial = {
            n: p.detach().cpu().clone() for n, p in actor.named_parameters() if p.requires_grad
        }
        report.update(
            public_sha256=actor.public_sha256,
            trainable_tensors=list(initial),
            trainable_parameters=sum(p.numel() for p in params),
        )
        probe = torch.randn(args.num_envs, 29, device=args.device)
        torch.testing.assert_close(task.to_holo(task.to_native(probe)), probe, atol=2e-6, rtol=2e-6)
        report["action_roundtrip_passed"] = True
        if args.phase == "evaluate":
            enabled = not args.public_only and args.expert != "off"
            set_domain_enabled(actor, domain, enabled)
            report["expert_enabled"] = enabled
            result, _, frames = task.rollout(actor, seed=args.seed, video=args.video)
            report.update(
                status="completed",
                student_eval=result,
                video=save_video(frames, "student.mp4"),
            )
            return
        features = FrozenFeatureDagger(actor, domain)
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        bank, shards, first = None, [], -1
        if restore:
            if restore["public_sha256"] != actor.public_sha256:
                raise ValueError("Resume public base differs")
            optimizer.load_state_dict(restore["optimizer"])
            first = restore["round"] + 1
            report["history"] = restore["history"]
            shards = restore["collection_shards"]
            for path in shards:
                new = torch.load(path, map_location="cpu", weights_only=True)
                bank = append_replay(bank, new, args.replay_capacity)
        schedule = [-1] if args.phase == "preflight" else range(first, args.rounds)
        for round_id in schedule:
            if any(file_digest(ROOT / name) != digest for name, digest in source_hashes.items()):
                raise RuntimeError("Experiment source changed after launch")
            # -1 is pure-teacher BC initialization; 0.. are online DAgger.
            beta = 1.0 if round_id < 0 else max(0.0, 0.8 - round_id * 0.2)
            collected, new, _ = task.rollout(
                actor,
                features,
                controller="teacher" if round_id < 0 else "mixed",
                teacher_probability=beta,
                seed=101 + round_id,
                max_steps=24 if args.phase == "preflight" else None,
            )
            shard = output / f"collection_{round_id + 1:03d}.pt"
            torch.save(new, shard)
            shards.append(str(shard))
            bank = append_replay(bank, new, args.replay_capacity)
            count = (
                64
                if args.phase == "preflight"
                else (args.bc_updates if round_id < 0 else args.updates)
            )
            losses, start_round = [], time.monotonic()
            for update_id in range(count):
                ids = torch.randint(len(bank["target"]), (args.batch_size,))
                keys = (
                    "x",
                    "router_x",
                    "native",
                    "anchor",
                    "shared",
                    "residual",
                    "context",
                    "target",
                )
                batch = {k: bank[k][ids].to(args.device) for k in keys}
                prediction = features.predict(batch)
                loss = (prediction - batch["target"]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite supervised loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = {
                    n: float(p.grad.norm()) if p.grad is not None else None
                    for n, p in actor.named_parameters()
                    if p.requires_grad
                }
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (update_id + 1) % 100 == 0:
                    report.update(
                        status="training",
                        round=round_id,
                        actor_updates=optimizer_step(optimizer),
                        last_loss=losses[-1],
                        updates_done_in_round=update_id + 1,
                    )
                    progress()
            if public_digest(actor) != actor.public_sha256:
                raise RuntimeError("Public Actor changed")
            if tensor_module_digest(task.depth) != encoder_hash:
                raise RuntimeError("Frozen encoder changed")
            if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):
                raise RuntimeError("Frozen public parameters received gradients")
            row = {
                "round": round_id,
                "phase": "BC" if round_id < 0 else "DAgger",
                "collection": collected,
                "replay_size": len(bank["target"]),
                "actual_updates_this_round": count,
                "actual_adam_steps": optimizer_step(optimizer),
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_mean": sum(losses) / len(losses),
                "update_seconds": time.monotonic() - start_round,
                "gradient_norms_last_update": grads,
            }
            plugin_path = output / f"round_{round_id + 1:03d}.domain.pt"
            save_domain(
                actor,
                domain,
                plugin_path,
                {
                    "method": "InstinctMJ_official_teacher_DAgger",
                    "task_id": args.task,
                    "teacher_checkpoint_sha256": report["teacher_checkpoint_sha256"],
                    "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                    "reference_mode": "official_teacher_task_reference",
                    "fixed_merge_bias": 0.0,
                    "actual_adam_steps": optimizer_step(optimizer),
                },
            )
            if args.phase == "train" and (
                round_id < 0
                or (round_id + 1) % args.eval_interval == 0
                or round_id == args.rounds - 1
            ):
                row["student_eval"], _, _ = task.rollout(actor, seed=7001)
            report["history"].append(row)
            report.update(
                status="training",
                actor_updates=optimizer_step(optimizer),
                latest_plugin=str(plugin_path),
                round=round_id,
                parameter_changes={
                    n: not torch.equal(initial[n], p.detach().cpu())
                    for n, p in actor.named_parameters()
                    if n in initial
                },
            )
            state = {
                **{
                    k: report[k]
                    for k in [
                        "public_sha256",
                        "encoder_sha256",
                        "teacher_checkpoint_sha256",
                        "dataset_manifest_sha256",
                    ]
                },
                "format": "instinct_teacher_dagger_v1",
                "round": round_id,
                "plugin": plugin_path.name,
                "optimizer": optimizer.state_dict(),
                "history": report["history"],
                "collection_shards": shards,
            }
            temporary = output / "training_state.tmp"
            torch.save(state, temporary)
            temporary.replace(output / "training_state.pt")
            progress()
            print(
                json.dumps(
                    {
                        "round": round_id,
                        "actor_updates": report["actor_updates"],
                        "loss": losses[-1],
                        "evaluation": row.get("student_eval"),
                        "elapsed_s": report["elapsed_s"],
                    }
                ),
                flush=True,
            )
        report.update(
            status="completed",
            public_unchanged=public_digest(actor) == actor.public_sha256,
            encoder_unchanged=tensor_module_digest(task.depth) == encoder_hash,
            teacher_unchanged=tensor_module_digest(task.teacher_actor) == teacher_hash,
            student_executed_transitions=sum(
                h["collection"]["student_executed_transitions"] for h in report["history"]
            ),
            teacher_queries=sum(h["collection"]["teacher_queries"] for h in report["history"]),
        )
        if not all(report["parameter_changes"].values()):
            raise RuntimeError("One or more of the four new tensors did not change")
    except BaseException as error:
        failure = error
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        progress()
        if features is not None:
            features.close()
        if task is not None:
            task.close()
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
