#!/usr/bin/env python3
"""Materialize or execute the frozen Isaac Teacher V1 collection campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import yaml


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--job-index-min", type=int, default=1,
                        help="first 1-based job to execute; manifest still contains all jobs")
    parser.add_argument("--job-index-max", type=int,
                        help="last 1-based job to execute; manifest still contains all jobs")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("schema") != 1:
        raise ValueError("unsupported campaign schema")
    source_root = Path(cfg["source_root"])
    bundle_root = Path(cfg["bundle_root"])
    launcher = resolve(source_root, cfg["launcher"])
    collector = Path(cfg["collector"])
    phase_labels = resolve(source_root, cfg["phase_labels"])
    output_dir = Path(cfg["output_dir"])
    jobs = []
    for checkpoint in cfg["checkpoints"]:
        checkpoint_path = resolve(source_root, checkpoint["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        policy_bundle = resolve(bundle_root, checkpoint["policy_bundle"])
        if not policy_bundle.is_file():
            raise FileNotFoundError(policy_bundle)
        for seed in cfg["seeds"]:
            for friction in cfg["friction_conditions"]:
                trajectory_id = f"{checkpoint['id']}_s{seed}_{friction['name']}"
                output = output_dir / checkpoint["id"] / f"{trajectory_id}.npz"
                command = [
                    str(launcher), str(collector),
                    "--checkpoint", str(checkpoint_path),
                    "--policy-bundle", str(policy_bundle),
                    "--checkpoint-id", checkpoint["id"],
                    "--env-config-class", checkpoint["env_config_class"],
                    "--output", str(output), "--trajectory-id", trajectory_id,
                    "--seed", str(seed), "--phase-labels", str(phase_labels),
                    "--static-friction", str(friction["static"]),
                    "--dynamic-friction", str(friction["dynamic"]),
                    "--manual-seed-without-replicator",
                ]
                if args.headless:
                    command.append("--headless")
                jobs.append({
                    "checkpoint_id": checkpoint["id"], "split": checkpoint["split"],
                    "seed": seed, "friction_condition": friction["name"],
                    "output": str(output), "command": command,
                })
    manifest = {"schema": 1, "config": str(args.config.resolve()), "jobs": jobs}
    args.job_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.job_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.execute:
        print(json.dumps({"jobs": len(jobs), "manifest": str(args.job_manifest)}, indent=2))
        return
    if args.job_index_min < 1 or (
            args.job_index_max is not None and args.job_index_max < args.job_index_min):
        parser.error("invalid job index range")
    for index, job in enumerate(jobs, 1):
        if index < args.job_index_min or (
                args.job_index_max is not None and index > args.job_index_max):
            continue
        output = Path(job["output"])
        if output.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(jobs)}] {job['checkpoint_id']} seed={job['seed']} "
              f"friction={job['friction_condition']}", flush=True)
        subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
