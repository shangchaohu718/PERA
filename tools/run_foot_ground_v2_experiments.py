#!/usr/bin/env python3
"""Run the one-change-at-a-time V2 task-profile and intent ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--intent-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--experiment", action="append",
                        help="run only named experiment(s); repeat to select several")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--diagnostic-old-test", action="store_true",
                        help="old V1 test is diagnostic only and cannot select V2")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    optimization = config["optimization"]
    epochs = args.epochs or optimization["epochs"]
    if epochs < 1:
        parser.error("epochs must be positive")
    train_tool = Path(__file__).with_name("train_foot_ground_teacher.py")
    eval_tool = Path(__file__).with_name("evaluate_foot_ground_teacher.py")
    runs = []
    for experiment in config["experiments"]:
        if args.experiment and experiment["name"] not in set(args.experiment):
            continue
        manifest = args.intent_manifest if experiment["reference_intent"] \
            else args.baseline_manifest
        for seed in optimization["seeds"]:
            directory = args.output_root / experiment["name"] / f"seed_{seed}"
            runs.append({
                **experiment, "seed": int(seed), "manifest": str(manifest.resolve()),
                "output_dir": str(directory.resolve()),
                "teacher": str((directory / "teacher.pt").resolve()),
                "validation": str((directory / "validation.json").resolve()),
                "old_test_diagnostic": str((directory / "old_test_diagnostic.json").resolve()),
            })
    if not runs:
        parser.error("experiment selection matched no configured runs")
    plan = {
        "schema": 2, "config": str(args.config.resolve()), "epochs": epochs,
        "runs": runs, "selection_data": "validation only",
        "old_v1_test_policy": "development diagnostic only; forbidden for selection",
        "final_test_policy": "new checkpoint-exclusive source required",
        "actor_updates": 0, "ppo_updates": 0,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps({"runs": len(runs), "plan": str(plan_path.resolve())}, indent=2))
        return
    for item in runs:
        teacher = Path(item["teacher"])
        if not teacher.exists():
            train_command = [
                str(args.launcher), str(train_tool), "--manifest", item["manifest"],
                "--output-dir", item["output_dir"], "--epochs", str(epochs),
                "--seed", str(item["seed"]), "--device", args.device,
                "--learning-rate", str(optimization["learning_rate"]),
                "--contact-weight", str(optimization["contact_weight"]),
                "--task-profile", item["task_profile"], "--replay-updates",
                str(optimization["replay_updates"]), "--gradient-audit-every",
                str(optimization["gradient_audit_every"]),
                "--latent-dim", str(item.get("latent_dim", 0)),
            ]
            if "latent_group_dims" in item:
                train_command.extend(
                    ["--latent-group-dims", *map(str, item["latent_group_dims"])])
            run(train_command)
        run([
            str(args.launcher), str(eval_tool), "--checkpoint", item["teacher"],
            "--manifest", item["manifest"], "--split", "validation",
            "--output", item["validation"], "--device", args.device,
        ])
        if args.diagnostic_old_test:
            threshold = json.loads(Path(item["validation"]).read_text())["threshold"]
            run([
                str(args.launcher), str(eval_tool), "--checkpoint", item["teacher"],
                "--manifest", item["manifest"], "--split", "test",
                "--threshold", str(threshold), "--output", item["old_test_diagnostic"],
                "--device", args.device,
            ])
    print(json.dumps({"completed_runs": len(runs),
                      "old_test_diagnostic": args.diagnostic_old_test}, indent=2))


if __name__ == "__main__":
    main()
