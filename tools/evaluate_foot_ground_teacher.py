"""Validation-threshold selection or frozen held-out Teacher V1 evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from unified_motion.foot_ground_teacher.evaluation import (  # noqa: E402
    aggregate_metric_dicts, choose_contact_threshold, contact_mode_coverage,
    evaluate_predictions,
)
from unified_motion.foot_ground_teacher.model import (  # noqa: E402
    FootGroundTeacher, FootGroundTeacherConfig,
)
from unified_motion.foot_ground_teacher.training import load_batch  # noqa: E402
from unified_motion.foot_ground_teacher.objective import TASK_PROFILE_ACTIVE_LOSSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float,
                        help="required for test; forbidden for validation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.split == "test") != (args.threshold is not None):
        parser.error("test requires --threshold; validation selects its threshold")
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = FootGroundTeacher(FootGroundTeacherConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(args.device).eval()
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["records"] if row["split"] == args.split]
    if not rows:
        raise ValueError(f"manifest has no {args.split} records")
    if args.split == "test":
        heldout = set(manifest.get("heldout_checkpoints", ()))
        if not any(row["metadata"]["checkpoint_id"] in heldout for row in rows):
            raise ValueError("test has no held-out policy checkpoint")
    reports = []
    threshold = args.threshold
    evaluated = []
    for row in rows:
        batch = load_batch(row["path"], args.device)
        with torch.no_grad():
            output = model(batch.student_observation, batch.privileged_current,
                           batch.executed_action, batch.foot_current, batch.episode_start,
                           batch.reference_intent)
        evaluated.append((row, batch, output))
    if threshold is None:
        logits = torch.cat([item[2]["contact_logits"].flatten() for item in evaluated])
        truth = torch.cat([item[1].contact.flatten() for item in evaluated])
        mask = torch.cat([item[1].valid_transition.unsqueeze(-1)
                          .expand_as(item[1].contact).flatten() for item in evaluated])
        threshold = choose_contact_threshold(logits, truth, mask)
    for row, batch, output in evaluated:
        reports.append({"source": row["path"], "checkpoint_id": row["metadata"]["checkpoint_id"],
                        "coverage": contact_mode_coverage(batch),
                        "metrics": evaluate_predictions(output, batch,
                                                        contact_threshold=threshold)})
    coverage = {name: sum(record["coverage"][name] for record in reports)
                for name in reports[0]["coverage"]}
    task_profile = payload.get("task_profile", "v1")
    result = {"schema": 1, "split": args.split, "threshold": threshold,
              "seed": payload["seed"], "teacher_optimizer_updates": payload["optimizer_updates"],
              "task_profile": task_profile,
              "active_supervision": payload.get(
                  "active_supervision", list(TASK_PROFILE_ACTIVE_LOSSES[task_profile])),
              "heldout_checkpoint_test": args.split == "test", "records": reports,
              "aggregate_coverage": coverage,
              "aggregate_metrics": aggregate_metric_dicts(
                  [record["metrics"] for record in reports]),
              "deployment_claim": False,
              "identifiability_note": ("Contact force and point are privileged targets; this result "
                                       "does not establish observability from q/dq/IMU/tau/history alone.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"split": args.split, "threshold": threshold,
                      "records": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
