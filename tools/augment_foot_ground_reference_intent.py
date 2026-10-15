#!/usr/bin/env python3
"""Copy a V1 manifest into a V2 dataset with causal reference-intent features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_motion.foot_ground_teacher.dataset import file_sha256, write_trajectory  # noqa: E402
from unified_motion.foot_ground_teacher.reference import (  # noqa: E402
    REFERENCE_INTENT_NAMES, reference_intent_from_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.phase_labels, allow_pickle=False) as labels:
        role_code = labels["role_code"]
        cycle_start = int(labels["cycle_start_50hz"])
        cycle_frames = int(labels["cycle_frames"])
    records = []
    for row in manifest["records"]:
        source = Path(row["path"])
        with np.load(source, allow_pickle=False) as archive:
            if "reference_frame" not in archive:
                raise ValueError(f"{source} lacks current reference_frame")
            arrays = {name: archive[name] for name in archive.files
                      if name != "metadata_json"}
        arrays["reference_intent"] = reference_intent_from_frames(
            arrays["reference_frame"], role_code,
            cycle_start=cycle_start, cycle_frames=cycle_frames)
        metadata = dict(row["metadata"])
        metadata["reference_intent_contract"] = {
            "schema": 1, "names": list(REFERENCE_INTENT_NAMES),
            "source": "current_reference_frame_t",
            "future_realized_contact_used": False,
            "phase_labels": str(args.phase_labels.resolve()),
            "cycle_start_50hz": cycle_start, "cycle_frames": cycle_frames,
        }
        target = args.output_dir / metadata["checkpoint_id"] / source.name
        write_trajectory(target, arrays, metadata)
        records.append({**row, "path": str(target.resolve()),
                        "sha256": file_sha256(target), "metadata": metadata})
    output = {**manifest, "records": records, "reference_intent_schema": {
        "schema": 1, "names": list(REFERENCE_INTENT_NAMES),
        "future_realized_contact_used": False,
    }}
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records),
                      "output_manifest": str(args.output_manifest.resolve())}, indent=2))


if __name__ == "__main__":
    main()
