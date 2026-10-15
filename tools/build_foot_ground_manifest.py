"""Build leakage-safe dataset splits and audit contact-mode coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from unified_motion.foot_ground_teacher.dataset import build_manifest  # noqa: E402
from unified_motion.foot_ground_teacher.schema import CONTACT_MODES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--heldout-checkpoint", action="append", required=True)
    parser.add_argument("--validation-checkpoint", action="append", required=True)
    parser.add_argument("--exclude-dir", action="append", default=[])
    parser.add_argument("--minimum-mode-samples", type=int, default=1)
    args = parser.parse_args()
    excluded = set(args.exclude_dir)
    paths = sorted(path for path in args.data_dir.rglob("*.npz")
                   if not excluded.intersection(path.relative_to(args.data_dir).parts[:-1]))
    if not paths:
        raise FileNotFoundError(f"no NPZ trajectories in {args.data_dir}")
    manifest = build_manifest(
        paths, heldout_checkpoints=args.heldout_checkpoint,
        validation_checkpoints=args.validation_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    coverage = {}
    for split in ("train", "validation", "test"):
        counts = {name: 0 for name in CONTACT_MODES}
        counts.update(left_anchor_right_slide=0, right_anchor_left_slide=0,
                      double_support=0, contact_handoff=0,
                      active_sliding=0, active_sliding_contact=0,
                      unexpected_slip=0,
                      valid_transitions=0, invalid_transitions=0)
        for row in manifest["records"]:
            if row["split"] != split:
                continue
            with np.load(row["path"], allow_pickle=False) as data:
                valid = data["valid_transition"].astype(bool)
                mode = data["contact_mode"]
                contact = data["contact"].astype(bool)
                active = data["active_sliding"].astype(bool)
                unexpected = data["unexpected_slip"].astype(bool)
                counts["valid_transitions"] += int(valid.sum())
                counts["invalid_transitions"] += int((~valid).sum())
                foot_valid = valid[:, None]
                for index, name in enumerate(CONTACT_MODES):
                    counts[name] += int(((mode == index) & foot_valid).sum())
                counts["active_sliding"] += int((active & foot_valid).sum())
                counts["active_sliding_contact"] += int(
                    (active & contact & foot_valid).sum())
                counts["unexpected_slip"] += int((unexpected & foot_valid).sum())
                counts["double_support"] += int((contact.all(1) & valid).sum())
                counts["left_anchor_right_slide"] += int((
                    (mode[:, 0] == 1) & (mode[:, 1] == 2) & valid).sum())
                counts["right_anchor_left_slide"] += int((
                    (mode[:, 1] == 1) & (mode[:, 0] == 2) & valid).sum())
                switch = contact[1:] != contact[:-1]
                counts["contact_handoff"] += int((switch.any(1) & valid[1:] & valid[:-1]).sum())
        coverage[split] = counts
    missing = [name for name in CONTACT_MODES
               if coverage["train"][name] < args.minimum_mode_samples]
    required_patterns = ("left_anchor_right_slide", "right_anchor_left_slide",
                         "double_support", "contact_handoff",
                         "active_sliding_contact", "unexpected_slip")
    missing += [name for name in required_patterns
                if coverage["train"][name] < args.minimum_mode_samples]
    report = {"schema": 1, "coverage": coverage, "minimum_mode_samples":
              args.minimum_mode_samples, "training_coverage_pass": not missing,
              "missing_or_underrepresented": missing}
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
