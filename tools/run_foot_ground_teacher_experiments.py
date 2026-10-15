#!/usr/bin/env python3
"""Run the three-seed contact-weight ablation and held-out evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_motion.foot_ground_teacher.evaluation import aggregate_metric_dicts  # noqa: E402


def weight_name(value: float) -> str:
    return str(value).replace(".", "p")


def validation_quality_pass(reports: list[dict], gate: dict) -> bool:
    if len(reports) < gate["required_test_seeds"]:
        return False
    metrics = aggregate_metric_dicts([report["aggregate_metrics"] for report in reports])
    for foot in ("left", "right"):
        values = metrics["feet"][foot]
        if not (
            values["contact"]["f1"] >= gate["contact_f1_min"]
            and values["contact"]["brier"] <= gate["contact_brier_max"]
            and values["contact_mode"]["macro_f1"] >= gate["contact_mode_macro_f1_min"]
            and values["tangent_velocity"]["rmse"]
                <= gate["tangent_velocity_rmse_m_s_max"]
            and values["slip_speed"]["rmse"] <= gate["slip_speed_rmse_m_s_max"]
            and values["normal_force"]["rmse"] <= gate["normal_force_rmse_n_max"]
            and values["tangential_force"]["rmse"]
                <= gate["tangential_force_rmse_n_max"]
        ):
            return False
    dynamics = metrics["dynamics"]
    return (
        dynamics["delta_root_angular_velocity"]["rmse"]
            <= gate["root_angular_velocity_delta_rmse_rad_s_max"]
        and dynamics["delta_joint_velocity_vs_persistence"]
            ["relative_improvement"]["rmse"]
            >= gate["delta_joint_velocity_vs_persistence_rmse_improvement_min"]
    )


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True,
                        help="Python or environment launcher placed before each tool path")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="after validation-only selection, open the held-out test split")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = config["optimization"]["seeds"]
    weights = config["loss"]["contact_weight_ablation"]
    train_tool = Path(__file__).with_name("train_foot_ground_teacher.py")
    eval_tool = Path(__file__).with_name("evaluate_foot_ground_teacher.py")
    runs = []
    for weight in weights:
        for seed in seeds:
            output_dir = args.output_root / f"contact_{weight_name(weight)}" / f"seed_{seed}"
            runs.append({
                "contact_weight": float(weight), "seed": int(seed),
                "output_dir": str(output_dir),
                "teacher": str(output_dir / "teacher.pt"),
                "validation": str(output_dir / "validation.json"),
                "test": str(output_dir / "test.json"),
            })
    plan = {
        "schema": 1, "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()), "epochs": args.epochs,
        "device": args.device, "runs": runs,
        "selection_rule": "highest pooled validation per-foot contact F1 across three seeds",
        "quality_policy": ("validation/test gates are reported independently; a failed gate "
                           "blocks integration but does not suppress diagnostic held-out eval"),
        "test_policy": "evaluate held-out checkpoint only for selected contact weight",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps({"runs": len(runs), "plan": str(plan_path)}, indent=2))
        return

    for item in runs:
        output_dir = Path(item["output_dir"])
        teacher = Path(item["teacher"])
        if not teacher.exists():
            run([
                str(args.launcher), str(train_tool),
                "--manifest", str(args.manifest), "--output-dir", str(output_dir),
                "--epochs", str(args.epochs), "--seed", str(item["seed"]),
                "--device", args.device, "--learning-rate",
                str(config["optimization"]["learning_rate"]),
                "--contact-weight", str(item["contact_weight"]),
                "--replay-updates", str(config["optimization"]["replay_updates"]),
                "--gradient-audit-every",
                str(config["optimization"]["gradient_audit_every"]),
            ])
        run([
            str(args.launcher), str(eval_tool), "--checkpoint", str(teacher),
            "--manifest", str(args.manifest), "--split", "validation",
            "--output", item["validation"], "--device", args.device,
        ])

    weight_scores = {}
    weight_quality_pass = {}
    for weight in weights:
        selected = [item for item in runs if item["contact_weight"] == float(weight)]
        reports = [json.loads(Path(item["validation"]).read_text(encoding="utf-8"))
                   for item in selected]
        pooled = aggregate_metric_dicts([report["aggregate_metrics"] for report in reports])
        weight_scores[str(weight)] = sum(
            pooled["feet"][foot]["contact"]["f1"] for foot in ("left", "right")
        ) / 2
        weight_quality_pass[str(weight)] = validation_quality_pass(
            reports, config["quality_gate"])
    selected_weight = float(max(weight_scores, key=weight_scores.get))
    selected_runs = [item for item in runs if item["contact_weight"] == selected_weight]
    selection = {
        "schema": 1, "validation_mean_contact_f1_by_weight": weight_scores,
        "validation_quality_pass_by_weight": weight_quality_pass,
        "selected_contact_weight": selected_weight,
        "selected_validation_gate_pass": weight_quality_pass[str(selected_weight)],
        "selected_without_test_access": True,
        "heldout_evaluated": False,
        "test_runs": [item["test"] for item in selected_runs],
    }
    selection_path = args.output_root / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    if not args.evaluate_test:
        print(json.dumps(selection, indent=2))
        return
    for item in selected_runs:
        validation = json.loads(Path(item["validation"]).read_text(encoding="utf-8"))
        run([
            str(args.launcher), str(eval_tool), "--checkpoint", item["teacher"],
            "--manifest", str(args.manifest), "--split", "test",
            "--threshold", str(validation["threshold"]),
            "--output", item["test"], "--device", args.device,
        ])
    selection["heldout_evaluated"] = True
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
