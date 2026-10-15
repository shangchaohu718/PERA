#!/usr/bin/env python3
"""Audit pooled contact calibration and true-mode strata for three Teacher seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_motion.foot_ground_teacher.calibration import contact_calibration  # noqa: E402
from unified_motion.foot_ground_teacher.model import FootGroundTeacher, FootGroundTeacherConfig  # noqa: E402
from unified_motion.foot_ground_teacher.training import load_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.validation_reports):
        parser.error("checkpoints and validation-reports must have equal length")
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["records"] if row["split"] == args.split]
    collected = {foot: {name: [] for name in
                 ("probability", "truth", "true_mode", "predicted_mode", "threshold")}
                 for foot in range(2)}
    seeds, profiles = [], []
    for checkpoint, validation_path in zip(args.checkpoints, args.validation_reports):
        payload = torch.load(checkpoint, map_location=args.device, weights_only=True)
        model = FootGroundTeacher(FootGroundTeacherConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(args.device).eval()
        threshold = float(json.loads(validation_path.read_text())["threshold"])
        seeds.append(payload["seed"])
        profiles.append(payload.get("task_profile", "v1"))
        for row in rows:
            batch = load_batch(row["path"], args.device)
            with torch.inference_mode():
                output = model(batch.student_observation, batch.privileged_current,
                    batch.executed_action, batch.foot_current, batch.episode_start,
                    batch.reference_intent)
            valid = batch.valid_transition
            for foot in range(2):
                values = collected[foot]
                values["probability"].append(output["contact_logits"][..., foot][valid].sigmoid())
                values["truth"].append(batch.contact[..., foot][valid])
                values["true_mode"].append(batch.contact_mode[..., foot][valid])
                values["predicted_mode"].append(
                    output["contact_mode_logits"][..., foot, :][valid].argmax(-1))
                values["threshold"].append(torch.full_like(
                    output["contact_logits"][..., foot][valid], threshold))
    report = {"schema": 1, "split": args.split, "seeds": seeds,
              "task_profiles": sorted(set(profiles)), "feet": {}}
    for foot, name in enumerate(("left", "right")):
        values = {key: torch.cat(items) for key, items in collected[foot].items()}
        report["feet"][name] = contact_calibration(**values, bins=args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"split": args.split, "seeds": seeds,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
