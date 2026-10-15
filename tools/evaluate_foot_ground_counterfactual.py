#!/usr/bin/env python3
"""Evaluate a frozen Teacher on same-state base/plus/minus Isaac branches."""
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

from unified_motion.foot_ground_teacher.counterfactual import evaluate_action_counterfactual  # noqa: E402
from unified_motion.foot_ground_teacher.model import FootGroundTeacher, FootGroundTeacherConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-tolerance", type=float, default=1e-6)
    parser.add_argument("--processed-target-tolerance-rad", type=float, default=1e-4)
    parser.add_argument("--delayed-target-tolerance-rad", type=float, default=1e-4)
    parser.add_argument("--null-tangent-noise-max-m-s", type=float, default=1e-3)
    parser.add_argument("--null-force-noise-max-n", type=float, default=.1)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = FootGroundTeacher(FootGroundTeacherConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(args.device).eval()
    with np.load(args.dataset, allow_pickle=False) as archive:
        values = {name: torch.from_numpy(archive[name]).to(args.device) for name in archive.files
                  if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"]))
    actions = values["executed_action"]
    if actions.ndim != 4 or actions.shape[1] != 3:
        raise ValueError("executed_action must be [N,3,T,22]")
    samples, branches, steps, _ = actions.shape
    repeat = lambda value: value[:, None].expand(samples, branches, *value.shape[1:]).flatten(0, 1)
    reference = values.get("reference_intent")
    if model.config.reference_intent_dim == 0:
        reference = None
    elif reference is None:
        raise ValueError("intent-enabled Teacher requires reference_intent in the dataset")
    with torch.inference_mode():
        output = model(
            repeat(values["student_observation"]),
            repeat(values["privileged_current"]), actions.flatten(0, 1),
            repeat(values["foot_current"]), repeat(values["episode_start"]),
            None if reference is None else repeat(reference),
        )
    final = lambda name: output[name][:, -1].reshape(samples, branches, *output[name].shape[2:])
    report = evaluate_action_counterfactual(
        predicted_contact_probability=final("contact_logits").sigmoid(),
        predicted_tangent_velocity=final("relative_velocity_tangent"),
        predicted_normal_force=final("normal_force"),
        target_contact=values["target_contact"].bool(),
        target_tangent_velocity=values["target_tangent_velocity"],
        target_normal_force=values["target_normal_force"],
        delta_rad=values["delta_rad"], contact_threshold=args.threshold,
        processed_target_delta_rad=values.get("processed_target_delta_rad"),
        null_target_contact=(None if "null_target_contact" not in values else
                             values["null_target_contact"].bool()),
        null_target_tangent_velocity=values.get("null_target_tangent_velocity"),
        null_target_normal_force=values.get("null_target_normal_force"),
    )
    quality = report["data_quality"]
    delayed_delta = values.get("delayed_actuator_target_delta_rad")
    delayed_plus_error = delayed_minus_error = None
    if delayed_delta is not None:
        delayed_plus_error = float((delayed_delta[:, 1] - values["delta_rad"]).abs().max())
        delayed_minus_error = float((delayed_delta[:, 2] + values["delta_rad"]).abs().max())
    checks = [
        {"name": "pre-branch state equality", "value": metadata.get(
            "pre_branch_state_max_abs_error"), "operator": "<=",
         "limit": args.state_tolerance},
        {"name": "processed plus target", "value": quality[
            "processed_plus_abs_error_rad"], "operator": "<=",
         "limit": args.processed_target_tolerance_rad},
        {"name": "processed minus target", "value": quality[
            "processed_minus_abs_error_rad"], "operator": "<=",
         "limit": args.processed_target_tolerance_rad},
        {"name": "delayed actuator plus target", "value": delayed_plus_error,
         "operator": "<=", "limit": args.delayed_target_tolerance_rad},
        {"name": "delayed actuator minus target", "value": delayed_minus_error,
         "operator": "<=", "limit": args.delayed_target_tolerance_rad},
        {"name": "null replicate available", "value": quality[
            "null_replicate_available"], "operator": "==", "limit": True},
        {"name": "null contact deterministic", "value": quality.get(
            "null_contact_match_rate"), "operator": "==", "limit": 1.0},
        {"name": "null tangent noise", "value": quality.get(
            "null_tangent_velocity_noise_rmse"), "operator": "<=",
         "limit": args.null_tangent_noise_max_m_s},
        {"name": "null force noise", "value": quality.get(
            "null_normal_force_noise_rmse_n"), "operator": "<=",
         "limit": args.null_force_noise_max_n},
    ]
    for check in checks:
        value, limit = check["value"], check["limit"]
        check["passed"] = value is not None and (
            value == limit if check["operator"] == "==" else value <= limit)
    sensitivity_checks = [
        {"name": "tangent response exceeds numerical noise", "value": quality.get(
            "tangent_response_above_noise_rate"), "operator": ">", "limit": 0.0},
        {"name": "force response exceeds numerical noise", "value": quality.get(
            "force_response_above_noise_rate"), "operator": ">", "limit": 0.0},
    ]
    for check in sensitivity_checks:
        value, limit = check["value"], check["limit"]
        check["passed"] = value is not None and value > limit
    result = {"schema": 1, "teacher_seed": payload["seed"],
              "teacher_task_profile": payload.get("task_profile", "v1"),
              "dataset": str(args.dataset.resolve()), "dataset_metadata": metadata,
              "metrics": report, "collection_quality_checks": checks,
              "collection_quality_gate_passed": all(x["passed"] for x in checks),
              "sensitivity_informativeness_checks": sensitivity_checks,
              "tangent_sensitivity_probe_informative": sensitivity_checks[0]["passed"],
              "force_sensitivity_probe_informative": sensitivity_checks[1]["passed"],
              "tangent_material_probe_informative": (
                  (quality.get("tangent_informative_response_rate") or 0.0) > 0.0),
              "force_material_probe_informative": (
                  (quality.get("force_informative_response_rate") or 0.0) > 0.0),
              "actor_updates": 0, "ppo_updates": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"samples": samples, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
