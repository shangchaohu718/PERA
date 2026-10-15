"""Inventory immutable K1 policy checkpoints without importing torch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iteration(path):
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return None if match is None else int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-all", action="store_true")
    args = parser.parse_args()
    records = []
    for run in sorted(path for path in args.runs.rglob("*") if path.is_dir()):
        checkpoints = sorted((p for p in run.glob("model_*.pt") if iteration(p) is not None),
                             key=iteration)
        if not checkpoints:
            continue
        runner_path = run / "mjlab_runner.json"
        runner = json.loads(runner_path.read_text()) if runner_path.is_file() else {}
        motions = [Path(item).name for item in (runner.get("motion_files") or [])]
        # Five phases provide behavioral diversity candidates. Actual behavioral
        # labels must be assigned from rollout measurements, never inferred here.
        indices = sorted({0, len(checkpoints)//4, len(checkpoints)//2,
                          3*len(checkpoints)//4, len(checkpoints)-1})
        selected = {checkpoints[i] for i in indices}
        for checkpoint in checkpoints:
            stat = checkpoint.stat()
            records.append({
                "checkpoint_id": str(checkpoint.relative_to(args.runs.parent)),
                "path": str(checkpoint.resolve()), "iteration": iteration(checkpoint),
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "mode": oct(stat.st_mode & 0o777), "sha256": (
                    sha256(checkpoint) if args.hash_all or checkpoint in selected else None),
                "candidate": checkpoint in selected,
                "behavior_class": "unmeasured",
                "motion_files": motions,
                "moonwalk_reference_verified": any(
                    "moon" in name.lower() for name in motions),
                "actor_backend": runner.get("actor_backend"),
            })
    summary = {
        "schema": 1, "runs_root": str(args.runs.resolve()),
        "checkpoint_count": len(records),
        "candidate_count": sum(item["candidate"] for item in records),
        "moonwalk_verified_count": sum(item["moonwalk_reference_verified"] for item in records),
        "warning": ("No checkpoint is linked to a motion filename containing 'moon'; "
                    "do not relabel ordinary walk checkpoints as moonwalk policies."),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "checkpoint_count", "candidate_count", "moonwalk_verified_count", "warning")}, indent=2))


if __name__ == "__main__":
    main()
