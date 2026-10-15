#!/usr/bin/env python3
"""Aggregate three-seed Teacher V1 results and apply the frozen quality gate."""

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


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check(checks: list[dict], name: str, value, operator: str, limit) -> None:
    passed = value is not None and (value >= limit if operator == ">=" else value <= limit)
    checks.append({"name": name, "value": value, "operator": operator,
                   "limit": limit, "passed": passed})


def _training_diagnostics(item: dict) -> dict:
    path = Path(item["output_dir"]) / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    audit_rows = [row for row in rows if "gradient_norm/contact_group" in row]
    audit_keys = sorted({key for row in audit_rows for key in row
                         if key.startswith("gradient_norm/")
                         or key.startswith("gradient_cosine/")})
    loss_keys = sorted(key for key in rows[-1] if key.startswith("loss/"))
    return {
        "contact_weight": item["contact_weight"], "seed": item["seed"],
        "optimizer_updates": len(rows), "gradient_audits": len(audit_rows),
        "initial_total_loss": rows[0]["total"], "final_total_loss": rows[-1]["total"],
        "final_atomic_losses": {key.removeprefix("loss/"): rows[-1][key]
                                for key in loss_keys},
        "gradient_audit_means": {
            key: sum(row[key] for row in audit_rows if key in row)
            / sum(key in row for row in audit_rows)
            for key in audit_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["quality_gate"]
    selection = _load(args.experiment_root / "selection.json")
    plan = _load(args.experiment_root / "experiment_plan.json")
    selected_weight = selection["selected_contact_weight"]
    selected = [item for item in plan["runs"]
                if item["contact_weight"] == selected_weight]
    validation_reports = [_load(item["validation"]) for item in selected]
    test_reports = [_load(item["test"]) for item in selected]
    validation = aggregate_metric_dicts(
        [report["aggregate_metrics"] for report in validation_reports])
    test = aggregate_metric_dicts(
        [report["aggregate_metrics"] for report in test_reports])
    checks: list[dict] = []
    validation_gate_passed = bool(selection.get("selected_validation_gate_pass", False))
    checks.append({"name": "selected validation quality gate",
                   "value": validation_gate_passed, "operator": "==",
                   "limit": True, "passed": validation_gate_passed})
    _check(checks, "independent test seeds", len(test_reports), ">=",
           gate["required_test_seeds"])
    heldout_ok = all(report["heldout_checkpoint_test"] for report in test_reports)
    checks.append({"name": "held-out checkpoint protocol", "value": heldout_ok,
                   "operator": "==", "limit": True, "passed": heldout_ok})
    for foot in ("left", "right"):
        metrics = test["feet"][foot]
        _check(checks, f"{foot} contact F1", metrics["contact"]["f1"], ">=",
               gate["contact_f1_min"])
        _check(checks, f"{foot} contact Brier", metrics["contact"]["brier"], "<=",
               gate["contact_brier_max"])
        _check(checks, f"{foot} contact-mode macro F1",
               metrics["contact_mode"]["macro_f1"], ">=",
               gate["contact_mode_macro_f1_min"])
        _check(checks, f"{foot} tangent-velocity RMSE",
               metrics["tangent_velocity"]["rmse"], "<=",
               gate["tangent_velocity_rmse_m_s_max"])
        _check(checks, f"{foot} slip-speed RMSE", metrics["slip_speed"]["rmse"], "<=",
               gate["slip_speed_rmse_m_s_max"])
        _check(checks, f"{foot} normal-force RMSE",
               metrics["normal_force"]["rmse"], "<=",
               gate["normal_force_rmse_n_max"])
        _check(checks, f"{foot} tangential-force RMSE",
               metrics["tangential_force"]["rmse"], "<=",
               gate["tangential_force_rmse_n_max"])
    dynamics = test["dynamics"]
    _check(checks, "root angular-velocity delta RMSE",
           dynamics["delta_root_angular_velocity"]["rmse"], "<=",
           gate["root_angular_velocity_delta_rmse_rad_s_max"])
    _check(checks, "joint-velocity delta improvement vs persistence",
           dynamics["delta_joint_velocity_vs_persistence"]["relative_improvement"]["rmse"],
           ">=", gate["delta_joint_velocity_vs_persistence_rmse_improvement_min"])
    required_coverage = ("stable", "sliding", "impact", "active_sliding",
                         "unexpected_slip")
    minimum_coverage = {
        name: min(report["aggregate_coverage"][name] for report in test_reports)
        for name in required_coverage
    }
    for name, value in minimum_coverage.items():
        _check(checks, f"test coverage {name}", value, ">=",
               gate["minimum_stratum_samples"])
    leakage = max(
        record["metrics"]["mask_audit"]["contact_regression_noncontact_leakage"]
        for report in test_reports for record in report["records"]
    )
    _check(checks, "non-contact regression leakage", leakage, "<=", 0)
    result = {
        "schema": 1,
        "selected_contact_weight": selected_weight,
        "selected_validation_gate_passed": validation_gate_passed,
        "selected_without_test_access": selection["selected_without_test_access"],
        "seeds": [report["seed"] for report in test_reports],
        "minimum_test_coverage_per_seed": minimum_coverage,
        "training_runs": [_training_diagnostics(item) for item in plan["runs"]],
        "validation_aggregate": validation,
        "heldout_test_aggregate": test,
        "checks": checks,
        "quality_gate_passed": all(item["passed"] for item in checks),
        "deployment_claim": False,
        "student_identifiability_note": (
            "This privileged Teacher result does not establish that force, contact point, "
            "friction, or disturbance are identifiable from q/dq/IMU/tau/action history."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"quality_gate_passed": result["quality_gate_passed"],
                      "checks_passed": sum(item["passed"] for item in checks),
                      "checks_total": len(checks), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
