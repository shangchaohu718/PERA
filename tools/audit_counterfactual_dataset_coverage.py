#!/usr/bin/env python3
"""Audit whether action-counterfactual data is large and diverse enough to train on."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_motion.foot_ground_teacher.counterfactual import evaluate_action_counterfactual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-checkpoint", action="append", default=[])
    parser.add_argument("--min-valid-probes", type=int, default=200)
    parser.add_argument("--min-checkpoints", type=int, default=3)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-deltas", type=int, default=3)
    parser.add_argument("--min-frictions", type=int, default=3)
    parser.add_argument("--min-material-per-role", type=int, default=20)
    parser.add_argument("--min-contact-mode", type=int, default=10)
    parser.add_argument("--state-tolerance", type=float, default=1e-6)
    parser.add_argument("--target-tolerance-rad", type=float, default=1e-4)
    parser.add_argument("--null-tangent-noise-max-m-s", type=float, default=1e-3)
    parser.add_argument("--null-force-noise-max-n", type=float, default=.1)
    args = parser.parse_args()

    records = []
    for path in sorted(set(p.resolve() for p in args.datasets)):
        with np.load(path, allow_pickle=False) as z:
            required = {
                "target_contact", "target_contact_mode", "target_tangent_velocity",
                "target_normal_force",
                "processed_target_delta_rad", "delayed_actuator_target_delta_rad",
                "null_target_contact", "null_target_tangent_velocity",
                "null_target_normal_force", "delta_rad", "reference_intent",
                "metadata_json",
            }
            missing = sorted(required - set(z.files))
            if missing:
                records.append({"path": str(path), "valid": False,
                                "failure": f"missing fields: {missing}"})
                continue
            value = {name: torch.from_numpy(z[name]) for name in required
                     if name != "metadata_json"}
            metadata = json.loads(str(z["metadata_json"]))
        # Predictions do not affect data-quality metrics; using truth here is
        # explicit and never presented as Teacher performance.
        report = evaluate_action_counterfactual(
            predicted_contact_probability=value["target_contact"].float(),
            predicted_tangent_velocity=value["target_tangent_velocity"],
            predicted_normal_force=value["target_normal_force"],
            target_contact=value["target_contact"].bool(),
            target_tangent_velocity=value["target_tangent_velocity"],
            target_normal_force=value["target_normal_force"],
            delta_rad=value["delta_rad"],
            processed_target_delta_rad=value["processed_target_delta_rad"],
            null_target_contact=value["null_target_contact"].bool(),
            null_target_tangent_velocity=value["null_target_tangent_velocity"],
            null_target_normal_force=value["null_target_normal_force"],
        )
        q = report["data_quality"]
        delayed = value["delayed_actuator_target_delta_rad"]
        delta = value["delta_rad"]
        checks = {
            "state": metadata.get("pre_branch_state_max_abs_error", float("inf"))
                     <= args.state_tolerance,
            "processed_plus": q["processed_plus_abs_error_rad"] <= args.target_tolerance_rad,
            "processed_minus": q["processed_minus_abs_error_rad"] <= args.target_tolerance_rad,
            "delayed_plus": float((delayed[:, 1] - delta).abs().max())
                            <= args.target_tolerance_rad,
            "delayed_minus": float((delayed[:, 2] + delta).abs().max())
                             <= args.target_tolerance_rad,
            "null_contact": q["null_contact_match_rate"] == 1.0,
            "null_tangent": q["null_tangent_velocity_noise_rmse"]
                            <= args.null_tangent_noise_max_m_s,
            "null_force": q["null_normal_force_noise_rmse_n"]
                          <= args.null_force_noise_max_n,
        }
        side = 0 if metadata["joint_name"].startswith("left_") else 1
        intent = value["reference_intent"][0, -1]
        role = "active_slide" if bool(intent[4 + side]) else \
               "planned_anchor" if bool(intent[2 + side]) else "neutral"
        base_modes = value["target_contact_mode"][0, 0].tolist() \
            if "target_contact_mode" in value else []
        records.append({
            "path": str(path), "valid": all(checks.values()), "checks": checks,
            "checkpoint_id": metadata["checkpoint_id"], "seed": metadata["seed"],
            "frame": metadata["probe_frame"], "joint_side": ("left", "right")[side],
            "selected_joint_role": role, "delta_rad": metadata["requested_delta_rad"],
            "friction": metadata["dynamic_friction"], "base_contact_modes": base_modes,
            "tangent_material": (q["tangent_informative_response_rate"] or 0.0) > 0,
            "force_material": (q["force_informative_response_rate"] or 0.0) > 0,
            "quality": q,
        })

    valid = [r for r in records if r.get("valid")]
    count = lambda pred: sum(bool(pred(r)) for r in valid)
    checkpoints = sorted({r["checkpoint_id"] for r in valid})
    seeds = sorted({r["seed"] for r in valid})
    deltas = sorted({r["delta_rad"] for r in valid})
    frictions = sorted({r["friction"] for r in valid})
    heldout = sorted(set(args.heldout_checkpoint) & set(checkpoints))
    coverage = {
        "attempted": len(records), "valid": len(valid),
        "checkpoint_ids": checkpoints, "seeds": seeds,
        "delta_rad_levels": deltas, "dynamic_friction_levels": frictions,
        "valid_by_joint_side": {side: count(lambda r, s=side: r["joint_side"] == s)
                                for side in ("left", "right")},
        "valid_by_selected_joint_role": {
            role: count(lambda r, x=role: r["selected_joint_role"] == x)
            for role in ("active_slide", "planned_anchor", "neutral")},
        "tangent_material_by_role": {
            role: count(lambda r, x=role: r["selected_joint_role"] == x
                        and r["tangent_material"])
            for role in ("active_slide", "planned_anchor")},
        "force_material_by_role": {
            role: count(lambda r, x=role: r["selected_joint_role"] == x
                        and r["force_material"])
            for role in ("active_slide", "planned_anchor")},
        "base_contact_mode_counts": {
            str(mode): sum(r["base_contact_modes"].count(mode) for r in valid)
            for mode in range(4)},
        "heldout_checkpoint_ids": heldout,
    }
    checks = [
        {"name": "valid probes", "value": len(valid), "minimum": args.min_valid_probes},
        {"name": "checkpoint sources", "value": len(checkpoints), "minimum": args.min_checkpoints},
        {"name": "simulator seeds", "value": len(seeds), "minimum": args.min_seeds},
        {"name": "delta levels", "value": len(deltas), "minimum": args.min_deltas},
        {"name": "friction levels", "value": len(frictions), "minimum": args.min_frictions},
        {"name": "heldout checkpoint sources", "value": len(heldout), "minimum": 1},
    ]
    for side in ("left", "right"):
        checks.append({"name": f"{side} joint probes",
                       "value": coverage["valid_by_joint_side"][side],
                       "minimum": args.min_material_per_role})
    for role in ("active_slide", "planned_anchor"):
        for target in ("tangent", "force"):
            checks.append({"name": f"{target} material {role}",
                           "value": coverage[f"{target}_material_by_role"][role],
                           "minimum": args.min_material_per_role})
    for mode, name in enumerate(("separation", "stable", "sliding", "impact")):
        checks.append({"name": f"base contact mode {name}",
                       "value": coverage["base_contact_mode_counts"][str(mode)],
                       "minimum": args.min_contact_mode})
    for check in checks:
        check["passed"] = check["value"] >= check["minimum"]
    payload = {
        "schema": 1, "purpose": "counterfactual training readiness only",
        "coverage": coverage, "checks": checks,
        "training_ready": all(c["passed"] for c in checks),
        "records": records, "actor_updates": 0, "ppo_updates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"training_ready": payload["training_ready"],
                      "valid": len(valid), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
