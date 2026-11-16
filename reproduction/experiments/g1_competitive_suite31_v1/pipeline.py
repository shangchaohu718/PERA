"""Serial native31 distillation. Default is CPU audit; --run explicitly trains."""

import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from .deduplication import duplicate

from pera.paths import ROOT

CODE = Path(__file__).resolve().parent
INTERPRETERS = {
    k: Path(os.environ.get("PERA_" + k.upper() + "_PYTHON", sys.executable))
    for k in [
        "mimicx",
        "pmt",
        "kick",
        "instinct",
        "parkour",
        "velocity",
        "husky",
        "ultra",
        "tennis",
    ]
}
DEFAULT_PY = Path(sys.executable)
RETENTION_PY = Path(os.environ.get("PERA_NATIVE_PYTHON", sys.executable))


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def load_config(path):
    c = read(path)
    ids = [t["id"] for t in c["tasks"]]
    if len(ids) != 31 or len(set(ids)) != 31:
        raise ValueError("Expected exactly31 independent task IDs")
    if (
        c["format"] != "g1_competitive_suite31_v1"
        or c["routing"] != "true_top2_8_public_plus_1_new"
    ):
        raise ValueError("Wrong base/routing method")
    if (
        not c["freeze_public"]
        or c["use_world_model"]
        or c["use_ppo"]
        or not c["train_independent_plugins"]
        or c["activate_outputs"]
        or not c.get("deduplicate_existing_same_base_tasks", True)
    ):
        raise ValueError("Unsupported training/activation setting")
    if (
        c["seed_updates"] + c["rounds"] * c["updates_per_round"] != c["maximum_updates_per_task"]
        or len(c["teacher_probability"]) != c["rounds"]
    ):
        raise ValueError("Update schedule mismatch")
    if (
        not 0 < c["old_reference_fraction"] < 1
        or not 0 < c["persistent_teacher_fraction"] < 1
        or c["old_reference_fraction"] + c["persistent_teacher_fraction"] >= 1
    ):
        raise ValueError("Invalid replay proportions")
    if c["base_sha256"] != sha(ROOT / c["base_checkpoint"]):
        raise ValueError("Current G1 base hash differs")
    return c


def environment(family, run_dir):
    e = os.environ.copy()
    paths = [
        ROOT / "src",
        ROOT / "integrations/HoloMotion",
        ROOT / "src",
        ROOT / "reproduction",
        ROOT,
    ]
    if family in ["instinct", "parkour", "velocity", "husky"]:
        paths.extend(
            [
                ROOT / "third_party/InstinctMJ/src",
                ROOT / "third_party/mjlab/src",
                ROOT / "third_party/instinct_rl",
            ]
        )
    if family == "ultra":
        paths.extend(
            [
                ROOT / "reproduction/experiments/ultra_box_v1",
                ROOT / "third_party/ULTRA/ultra",
            ]
        )
    if family == "tennis":
        paths.extend(
            [
                ROOT / "reproduction/experiments/athlete_tennis_v1",
                ROOT / "third_party/ATHLETE/athlete/src",
            ]
        )
    if family == "pmt":
        paths = [
            ROOT / "reproduction/experiments/pmt_dagger_v1/python_overlay",
            ROOT / "third_party/PMT",
            *paths,
        ]
        e.update(
            PMT_REPO_ROOT=str(ROOT / "third_party/PMT"),
            PMT_ASSET_DIR=str(ROOT / "reproduction/experiments/pmt_dagger_v1/asset_overlay"),
            PMT_TERRAIN_MOTION_ROOT=str(ROOT / "data/pmt_bigmap_official_v1/assets/motions"),
            PMT_TERRAIN_ASSET_DIR=str(ROOT / "data/pmt_bigmap_official_v1/assets/terrain"),
        )
    e.update(
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        TORCH_COMPILE_DISABLE="1",
        MUJOCO_GL="egl",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        WANDB_MODE="disabled",
        OMNI_KIT_ACCEPT_EULA="YES",
        ACCEPT_EULA="Y",
    )
    # Root partition /tmp may be full; task-local temporary/cache stays under /home.
    e.update(TMPDIR=str(run_dir / "tmp"), XDG_CACHE_HOME=str(run_dir / "cache"))
    return e


def command(t, config, out, preflight=False, resume=False):
    cmd = [
        str(INTERPRETERS.get(t["family"], DEFAULT_PY)),
        "-m",
        "experiments.g1_competitive_suite31_v1.worker",
        "--config",
        str(config),
        "--task",
        t["id"],
        "--output",
        str(out),
    ]
    if preflight:
        cmd.append("--preflight-only")
    if resume:
        cmd.append("--resume")
    return cmd


def audit(c, config):
    missing = []
    jobs = []
    for t in c["tasks"]:
        existing = duplicate(t, c["base_sha256"])
        py = INTERPRETERS.get(t["family"], DEFAULT_PY)
        if not py.is_file():
            missing.append(str(py))
        evidence = t.get("legacy_evidence_plugin")
        if (
            t.get("enabled", True)
            and t["family"] == "kick"
            and (not evidence or not (ROOT / evidence).is_file())
        ):
            missing.append(evidence or t["id"] + ": legacy evidence missing")
        jobs.append(
            dict(
                index=t["index"],
                id=t["id"],
                label=t["label"],
                enabled=t.get("enabled", True) and existing is None,
                skip_reason=t.get("skip_reason") or (existing["reason"] if existing else None),
                existing_training=existing,
                native_runtime=str(py),
                num_envs=t["num_envs"],
                context_dim=t["context_dim"],
                legacy_evidence_plugin=evidence,
                legacy_evidence_sha256=sha(ROOT / evidence)
                if evidence and (ROOT / evidence).is_file()
                else None,
                command=command(t, config, Path("<RUN>") / t["id"]),
            )
        )
    return dict(
        status="cpu_audit_passed" if not missing else "missing_resources",
        training_started=False,
        CUDA_initialized=False,
        base_sha256=c["base_sha256"],
        tasks=jobs,
        missing=missing,
        maximum_Adam_updates_per_task=c["maximum_updates_per_task"],
        scheduled_tasks=sum(t["enabled"] for t in jobs),
        skipped_tasks=[t["id"] for t in jobs if not t["enabled"]],
        maximum_queue_updates=c["maximum_updates_per_task"] * sum(t["enabled"] for t in jobs),
        independent_plugins=True,
        combined39_expert_policy_validated=False,
        native_GPU_preflight_completed=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/g1_competitive_suite31_v1.json",
    )
    p.add_argument("--output", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--tasks", nargs="+", help="Optional subset of exact IDs")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument(
        "--allow-retrain",
        action="store_true",
        help="Explicitly retrain named --tasks despite existing same-base training",
    )
    a = p.parse_args()
    if a.allow_retrain and not a.tasks:
        p.error("--allow-retrain requires explicit --tasks")
    config = a.config.resolve()
    c = load_config(config)
    check = audit(c, config)
    if not a.run and not a.preflight_only:
        print(json.dumps(check, indent=2, ensure_ascii=False))
        if a.output:
            write(a.output, check)
        if check["missing"]:
            sys.exit(2)
        return
    if check["missing"]:
        raise ValueError(check["missing"])
    if a.resume and a.preflight_only:
        p.error("Resume is for actual training")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    out = (
        a.resume or a.output or ROOT / "reports/g1_competitive_suite31_v1" / ("run_" + stamp)
    ).resolve()
    out.mkdir(parents=True, exist_ok=bool(a.resume))
    lock = (out / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # One queue globally; this lock never stops or manipulates unrelated jobs.
    global_lock = (CODE / "queue.lock").open("a")
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for n in ["tmp", "cache"]:
        (out / n).mkdir(exist_ok=True)
    if a.resume:
        state = read(out / "pipeline.json")
        config = out / "config.json"
        if sha(a.config.resolve()) != sha(config):
            raise ValueError("Resume configuration differs")
        c = load_config(config)
    else:
        write(out / "config.json", c)
        config = out / "config.json"
        write(out / "audit.json", check)
        state = dict(
            status="starting",
            created_cst=stamp,
            tasks={},
            preflight_only=a.preflight_only,
            base_sha256=c["base_sha256"],
            automatic_activation=False,
        )
        sources = {str(x.relative_to(ROOT)): sha(x) for x in CODE.glob("*.py")}
        write(out / "source_sha256.json", sources)
    selected = set(a.tasks or [t["id"] for t in c["tasks"] if t.get("enabled", True)])
    unknown = selected - set(t["id"] for t in c["tasks"])
    if unknown:
        p.error("Unknown tasks: " + ",".join(sorted(unknown)))
    state.update(pid=os.getpid(), current_task=None)
    state["disabled_tasks"] = [
        {"id": t["id"], "reason": t.get("skip_reason")}
        for t in c["tasks"]
        if not t.get("enabled", True)
    ]
    stopping = [False]
    child = [None]

    def stop(*_):
        stopping[0] = True
        if child[0] is not None and child[0].poll() is None:
            os.killpg(child[0].pid, signal.SIGTERM)

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, stop)

    def save():
        state["actual_Adam_updates"] = sum(
            x.get("actual_Adam_updates", 0) for x in state["tasks"].values()
        )
        state["finished_tasks"] = sum(
            x.get("status")
            in [
                "completed",
                "teacher_failed",
                "failed",
                "preflight_passed",
                "skipped_duplicate",
            ]
            for x in state["tasks"].values()
        )
        write(out / "pipeline.json", state)

    def run(cmd, log, e, entry):
        with log.open("a") as f:
            child[0] = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=e,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            entry.update(child_pid=child[0].pid, command=cmd)
            save()
            t0 = time.monotonic()
            stop_at = None
            while child[0].poll() is None:
                if stopping[0] and stop_at is None:
                    stop_at = time.monotonic()
                if stop_at is not None and time.monotonic() - stop_at > 60:
                    os.killpg(child[0].pid, signal.SIGKILL)
                if time.monotonic() - t0 > c["task_timeout_seconds"]:
                    os.killpg(child[0].pid, signal.SIGTERM)
                    try:
                        child[0].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child[0].pid, signal.SIGKILL)
                    entry["timeout"] = True
                current = Path(entry.get("output", "")) / "report.json"
                if current.is_file():
                    progress = read(current)
                    entry.update(
                        actual_Adam_updates=progress.get("actual_Adam_updates", 0),
                        worker_phase=progress.get("status"),
                        worker_elapsed_s=progress.get("elapsed_s"),
                    )
                save()
                time.sleep(2)
            return child[0].wait()

    try:
        for task in c["tasks"]:
            if task["id"] not in selected:
                continue
            if stopping[0] or (out / "PAUSE").exists():
                break
            entry = state["tasks"].setdefault(
                task["id"], dict(status="pending", actual_Adam_updates=0)
            )
            existing = duplicate(task, c["base_sha256"], exclude_run=out)
            if not a.allow_retrain and (not task.get("enabled", True) or existing):
                entry.update(
                    status="skipped_duplicate",
                    existing_training=existing,
                    reason=task.get("skip_reason")
                    or (existing["reason"] if existing else "Task disabled in configuration"),
                )
                save()
                continue
            if (
                entry["status"] in ["completed", "preflight_passed"]
                or entry["status"] in ["failed", "teacher_failed"]
                and not a.retry_failed
            ):
                continue
            state.update(status="waiting_for_gpu", current_task=task["id"])
            save()
            t0 = time.monotonic()
            while True:
                if stopping[0] or (out / "PAUSE").exists():
                    break
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                if (
                    r.returncode == 0
                    and float(r.stdout.splitlines()[0]) >= c["minimum_free_gpu_mib"]
                ):
                    break
                if time.monotonic() - t0 > c["resource_wait_seconds"]:
                    raise TimeoutError("GPU resource wait expired; queue can resume")
                time.sleep(10)
            if stopping[0] or (out / "PAUSE").exists():
                break
            existing = duplicate(task, c["base_sha256"], exclude_run=out)
            if existing and not a.allow_retrain:
                entry.update(
                    status="skipped_duplicate",
                    existing_training=existing,
                    reason=existing["reason"],
                )
                save()
                continue
            taskdir = out / task["id"]
            resume = taskdir.joinpath("training_state.pt").exists()
            if taskdir.exists() and not resume:
                # Preserve failed/preflight attempts; never overwrite artifacts.
                taskdir.rename(out / (task["id"] + "_previous_" + str(time.time_ns())))
            cmd = command(task, config, taskdir, a.preflight_only, resume)
            entry.update(status="running", output=str(taskdir))
            state["status"] = "running"
            save()
            code = run(
                cmd,
                out / (task["id"] + ".log"),
                environment(task["family"], out),
                entry,
            )
            result = (
                read(taskdir / "report.json")
                if (taskdir / "report.json").exists()
                else dict(status="failed", error="Worker exited without report")
            )
            entry.update(
                status=result["status"],
                actual_Adam_updates=result.get("actual_Adam_updates", 0),
                exit_code=code,
                task_accepted=result.get("task_accepted", False),
                checkpoint=result.get("best_checkpoint"),
                error=result.get("error"),
            )
            if stopping[0]:
                entry["status"] = "paused"
            if code and entry["status"] not in ["paused", "teacher_failed"]:
                entry["status"] = "failed"
            if not stopping[0] and entry["status"] == "completed":
                state["status"] = "evaluating_original128"
                save()
                retpath = out / (task["id"] + "_old128.json")
                e = environment("retention", out)
                retcmd = [
                    str(RETENTION_PY),
                    "-m",
                    "experiments.g1_competitive_suite31_v1.evaluate_retention",
                    "--domain",
                    entry["checkpoint"],
                    "--output",
                    str(retpath),
                ]
                retcode = run(retcmd, out / (task["id"] + "_old128.log"), e, entry)
                if retcode == 0:
                    ret = read(retpath)
                    baseline_record = read(ROOT / "configs/base_retention_baseline.json")
                    if baseline_record["base_sha256"] != c["base_sha256"]:
                        raise ValueError(
                            "Measure a retention baseline for this base checkpoint first"
                        )
                    baseline = baseline_record["evaluation"]
                    new = sorted(set(ret["failed"]) - set(baseline["failed"]))
                    keep = not new and ret["norm_return"] >= baseline["norm_return"] - 0.02
                    entry.update(
                        retention_accepted=keep,
                        new_old128_failures=new,
                        retention_report=str(retpath),
                        accepted_for_integration=bool(entry["task_accepted"] and keep),
                    )
                else:
                    entry.update(
                        retention_accepted=False,
                        retention_error="See old128 log",
                        accepted_for_integration=False,
                    )
            save()
            print(
                json.dumps(dict(task=task["label"], **entry), ensure_ascii=False),
                flush=True,
            )
        state["status"] = "paused" if stopping[0] or (out / "PAUSE").exists() else "queue_finished"
        state["accepted_tasks"] = [
            k for k, v in state["tasks"].items() if v.get("accepted_for_integration")
        ]
        state["not_accepted_tasks"] = [
            k for k, v in state["tasks"].items() if not v.get("accepted_for_integration")
        ]
        save()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        if child[0] is not None and child[0].poll() is None:
            os.killpg(child[0].pid, signal.SIGTERM)
            try:
                child[0].wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child[0].pid, signal.SIGKILL)
    print(str(out / "pipeline.json"), flush=True)


if __name__ == "__main__":
    main()
