#!/usr/bin/env python3
"""Materialize or execute the frozen-policy counterfactual collection campaign."""
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
    parser.add_argument("--job-index-min", type=int, default=1)
    parser.add_argument("--job-index-max", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("schema") != 1:
        raise ValueError("unsupported campaign schema")
    source_root, bundle_root = Path(cfg["source_root"]), Path(cfg["bundle_root"])
    launcher = resolve(source_root, cfg["launcher"])
    collector = Path(cfg["collector"])
    labels = resolve(source_root, cfg["phase_labels"])
    output_root = Path(cfg["output_dir"])
    frictions = cfg["friction_conditions"]
    jobs = []
    for checkpoint in cfg["checkpoints"]:
        source = resolve(source_root, checkpoint["path"])
        bundle = resolve(bundle_root, checkpoint["policy_bundle"])
        for seed_index, seed in enumerate(cfg["seeds"]):
            for probe_index, probe in enumerate(cfg["probe_frames"]):
                for joint in cfg["joints"]:
                    for delta_index, delta in enumerate(cfg["deltas_rad"]):
                        # A deterministic Latin rotation exposes every seed and
                        # phase stratum to every friction without a full extra
                        # Cartesian factor.
                        friction = frictions[(seed_index + probe_index + delta_index)
                                             % len(frictions)]
                        delta_tag = str(delta).replace("0.", "")
                        joint_tag = joint.removesuffix("_joint")
                        identity = (f"{checkpoint['id']}_s{seed}_f{probe['frame']:03d}_"
                                    f"{joint_tag}_d{delta_tag}_{friction['name']}")
                        output = output_root / checkpoint["split"] / checkpoint["id"] / \
                            f"{identity}.npz"
                        command = [
                            str(launcher), str(collector), "--headless", "--device", "cuda:0",
                            "--checkpoint", str(source), "--policy-bundle", str(bundle),
                            "--checkpoint-id", checkpoint["id"],
                            "--env-config-class", checkpoint["env_config_class"],
                            "--phase-labels", str(labels), "--output", str(output),
                            "--seed", str(seed), "--probe-frame", str(probe["frame"]),
                            "--joint-name", joint, "--delta-rad", str(delta),
                            "--context-steps", str(cfg["context_steps"]),
                            "--static-friction", str(friction["static"]),
                            "--dynamic-friction", str(friction["dynamic"]),
                            "--manual-seed-without-replicator",
                        ]
                        jobs.append({
                            "checkpoint_id": checkpoint["id"], "split": checkpoint["split"],
                            "seed": seed, "probe_frame": probe["frame"],
                            "stratum": probe["stratum"], "joint": joint,
                            "delta_rad": delta, "friction_condition": friction["name"],
                            "output": str(output), "command": command,
                        })
    manifest = {"schema": 1, "config": str(args.config.resolve()),
                "job_count": len(jobs), "jobs": jobs}
    args.job_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.job_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"jobs": len(jobs), "manifest": str(args.job_manifest.resolve())},
                         indent=2))
        return
    if args.job_index_min < 1 or (args.job_index_max is not None and
                                  args.job_index_max < args.job_index_min):
        parser.error("invalid job index range")
    for index, job in enumerate(jobs, 1):
        if index < args.job_index_min or (args.job_index_max is not None and
                                          index > args.job_index_max):
            continue
        output = Path(job["output"])
        if output.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(jobs)}] {job['checkpoint_id']} frame={job['probe_frame']} "
              f"joint={job['joint']} delta={job['delta_rad']}", flush=True)
        subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
