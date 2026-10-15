#!/usr/bin/env python3
"""Compare V2 validation-only ablations with the frozen V1 validation result."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_motion.foot_ground_teacher.evaluation import aggregate_metric_dicts  # noqa: E402


def gradients(directory: Path) -> dict:
    values: dict[str, list[float]] = {}
    for name in glob.glob(str(directory / "seed_*" / "metrics.jsonl")):
        for line in Path(name).read_text().splitlines():
            row = json.loads(line)
            for key, value in row.items():
                if key.startswith("gradient_norm/") or key.startswith("gradient_cosine/"):
                    values.setdefault(key, []).append(value)
    return {key: {"mean": statistics.mean(items),
                  "negative_fraction": (sum(x < 0 for x in items) / len(items)
                                        if key.startswith("gradient_cosine/") else None),
                  "count": len(items)} for key, items in sorted(values.items())}


def compact(metrics: dict, active: list[str]) -> dict:
    result = {"active_supervision": active, "feet": {}}
    for foot in ("left", "right"):
        item = metrics["feet"][foot]
        result["feet"][foot] = {
            "contact_bce": item["contact"]["bce"],
            "contact_brier": item["contact"]["brier"],
            "contact_f1": item["contact"]["f1"],
            "contact_mode_macro_f1": item["contact_mode"]["macro_f1"],
            "tangent_velocity_rmse_m_s": item["tangent_velocity"]["rmse"],
            "normal_force_rmse_n": item["normal_force"]["rmse"],
        }
    if "delta_joint_velocity" in active:
        dynamics = metrics["dynamics"]
        result["dynamics"] = {
            "delta_joint_velocity_rmse": dynamics["delta_joint_velocity"]["rmse"],
            "delta_joint_velocity_vs_persistence_rmse_improvement":
                dynamics["delta_joint_velocity_vs_persistence"]
                ["relative_improvement"]["rmse"],
            "delta_root_angular_velocity_rmse":
                dynamics["delta_root_angular_velocity"]["rmse"],
        }
    else:
        result["dynamics"] = {"status": "not supervised; emitted head is not evaluated"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v1 = json.loads(args.v1_report.read_text())
    output = {
        "schema": 1, "selection_split": "validation",
        "old_v1_test_accessed_for_v2_selection": False,
        "final_test_status": "not_run_new_checkpoint_required",
        "profiles": {
            "v1": compact(v1["validation_aggregate"], [
                "contact_bce", "contact_mode_ce", "relative_velocity_tangent",
                "normal_force", "delta_joint_velocity",
                "delta_root_angular_velocity", "joint_velocity_residual",
            ])
        },
    }
    for profile in args.profiles:
        directory = args.experiment_root / profile
        reports = [json.loads(Path(path).read_text()) for path in
                   sorted(glob.glob(str(directory / "seed_*" / "validation.json")))]
        if len(reports) != 3:
            raise ValueError(f"{profile} needs exactly three validation reports")
        active = reports[0]["active_supervision"]
        if any(report["active_supervision"] != active for report in reports):
            raise ValueError(f"{profile} supervision contracts differ across seeds")
        aggregate = aggregate_metric_dicts(
            [report["aggregate_metrics"] for report in reports])
        output["profiles"][profile] = {
            **compact(aggregate, active),
            "seeds": [report["seed"] for report in reports],
            "thresholds": [report["threshold"] for report in reports],
            "gradient_audit": gradients(directory),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"profiles": list(output["profiles"]),
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
