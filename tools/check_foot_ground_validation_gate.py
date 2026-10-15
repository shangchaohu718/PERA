#!/usr/bin/env python3
"""Check the selected three-seed Teacher candidate before opening held-out test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_motion.foot_ground_teacher.evaluation import aggregate_metric_dicts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = yaml.safe_load(args.config.read_text(encoding="utf-8"))["quality_gate"]
    plan = json.loads((args.experiment_root / "experiment_plan.json").read_text())
    selection = json.loads((args.experiment_root / "selection.json").read_text())
    if selection.get("heldout_evaluated"):
        raise ValueError("validation gate must be frozen before held-out evaluation")
    selected = [item for item in plan["runs"]
                if item["contact_weight"] == selection["selected_contact_weight"]]
    reports = [json.loads(Path(item["validation"]).read_text()) for item in selected]
    metrics = aggregate_metric_dicts([report["aggregate_metrics"] for report in reports])
    checks = []

    def check(name, value, relation, limit):
        passed = value >= limit if relation == ">=" else value <= limit
        checks.append({"name": name, "value": value, "operator": relation,
                       "limit": limit, "passed": passed})

    check("validation seeds", len(reports), ">=", gate["required_test_seeds"])
    for foot in ("left", "right"):
        values = metrics["feet"][foot]
        check(f"{foot} contact F1", values["contact"]["f1"], ">=",
              gate["contact_f1_min"])
        check(f"{foot} contact Brier", values["contact"]["brier"], "<=",
              gate["contact_brier_max"])
        check(f"{foot} mode macro F1", values["contact_mode"]["macro_f1"], ">=",
              gate["contact_mode_macro_f1_min"])
        check(f"{foot} tangent RMSE", values["tangent_velocity"]["rmse"], "<=",
              gate["tangent_velocity_rmse_m_s_max"])
        check(f"{foot} slip-speed RMSE", values["slip_speed"]["rmse"], "<=",
              gate["slip_speed_rmse_m_s_max"])
        check(f"{foot} normal-force RMSE", values["normal_force"]["rmse"], "<=",
              gate["normal_force_rmse_n_max"])
        check(f"{foot} tangential-force RMSE", values["tangential_force"]["rmse"], "<=",
              gate["tangential_force_rmse_n_max"])
    dynamics = metrics["dynamics"]
    check("root angular delta RMSE", dynamics["delta_root_angular_velocity"]["rmse"],
          "<=", gate["root_angular_velocity_delta_rmse_rad_s_max"])
    check("joint delta improvement vs persistence",
          dynamics["delta_joint_velocity_vs_persistence"]["relative_improvement"]["rmse"],
          ">=", gate["delta_joint_velocity_vs_persistence_rmse_improvement_min"])
    minimum_coverage = min(
        min(report["aggregate_coverage"][name] for name in
            ("stable", "sliding", "impact", "active_sliding", "unexpected_slip"))
        for report in reports
    )
    check("minimum validation stratum support", minimum_coverage, ">=",
          gate["minimum_stratum_samples"])
    result = {
        "schema": 1, "selected_contact_weight": selection["selected_contact_weight"],
        "heldout_accessed": False, "seeds": [report["seed"] for report in reports],
        "aggregate_metrics": metrics, "checks": checks,
        "validation_gate_passed": all(item["passed"] for item in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"validation_gate_passed": result["validation_gate_passed"],
                      "passed": sum(item["passed"] for item in checks),
                      "total": len(checks), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
