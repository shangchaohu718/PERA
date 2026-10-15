#!/usr/bin/env python3
"""Validate physical/schema invariants of collected Teacher V1 trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from unified_motion.foot_ground_teacher.dataset import (  # noqa: E402
    file_sha256,
    read_metadata,
)
from unified_motion.foot_ground_teacher.schema import CONTACT_MODES  # noqa: E402
from unified_motion.foot_ground_teacher.training import load_batch  # noqa: E402


def audit(path: Path, expected_sources: dict[str, str] | None = None) -> dict:
    # Runs all tensor-level shape, dtype, finite, mask, normal, and load-ratio checks.
    load_batch(path)
    metadata = read_metadata(path)
    errors = []
    if expected_sources is not None:
        expected = expected_sources.get(metadata["checkpoint_id"])
        if expected is None:
            errors.append("checkpoint_id is absent from frozen bundle audit")
        elif metadata["checkpoint_sha256"] != expected:
            errors.append("trajectory checkpoint SHA-256 differs from frozen source")
    with np.load(path, allow_pickle=False) as data:
        contact = data["contact"]
        contact_valid = data["contact_valid"]
        mode = data["contact_mode"]
        normal_force = data["normal_force"]
        tangent_velocity = data["relative_velocity_tangent"]
        slip_speed = data["slip_speed"]
        impulse = data["contact_impulse"]
        valid = data["valid_transition"]
        if not np.array_equal(contact_valid, contact):
            errors.append("contact_valid differs from active contact")
        if np.any(contact & (normal_force <= 1.0)):
            errors.append("active contact has force at/below the 1 N threshold")
        if np.any(contact & (mode == 0)):
            errors.append("active contact is classified as separation")
        if np.any(((mode == 1) | (mode == 2)) & ~contact):
            errors.append("stable/sliding mode lacks final active contact")
        if np.any(impulse < 0):
            errors.append("negative contact impulse")
        if contact_valid.any() and not np.allclose(
                np.linalg.norm(tangent_velocity, axis=-1)[contact_valid],
                slip_speed[contact_valid], atol=1e-5, rtol=1e-5):
            errors.append("slip_speed is inconsistent with tangent velocity")
        raw_required = {
            "raw_contact_found", "raw_contact_position_w", "raw_contact_normal_w",
            "raw_contact_force_w", "raw_substep_contact_found",
            "raw_substep_contact_position_w", "raw_substep_contact_normal_w",
            "raw_substep_normal_force_n", "raw_substep_friction_found",
            "raw_substep_friction_position_w", "raw_substep_friction_force_w",
        }
        if missing := sorted(raw_required - set(data.files)):
            errors.append(f"missing raw solver arrays: {missing}")
            physics_substeps = None
        else:
            physics_substeps = int(data["raw_substep_contact_found"].shape[2])
            if physics_substeps != 4:
                errors.append(f"expected 4 physics substeps, got {physics_substeps}")
            raw_found = data["raw_substep_contact_found"].astype(bool)
            raw_force = data["raw_substep_normal_force_n"]
            active_raw = raw_found & (raw_force > 1.0)
            expected_final_force = (raw_force[:, :, -1] * active_raw[:, :, -1]).sum(-1)
            if not np.allclose(normal_force, expected_final_force, atol=1e-3, rtol=1e-4):
                errors.append("normal force differs from final-substep solver sum")
            expected_impulse = (raw_force * active_raw).sum(axis=(-1, -2)) * 0.005
            if not np.allclose(impulse, expected_impulse, atol=1e-4, rtol=1e-4):
                errors.append("contact impulse differs from four-substep force integral")
            final_found = data["raw_contact_found"].astype(bool)
            final_normals = data["raw_contact_normal_w"]
            if final_found.any() and np.any(final_normals[..., 2][final_found] < 0):
                errors.append("raw contact normal is oriented below the ground plane")
        mode_counts = {
            CONTACT_MODES[index]: [int((mode[:, foot] == index).sum()) for foot in range(2)]
            for index in range(len(CONTACT_MODES))
        }
        active_counts = [int(contact[:, foot].sum()) for foot in range(2)]
        record = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "checkpoint_id": metadata["checkpoint_id"],
            "seed": metadata["seed"],
            "trajectory_id": metadata["trajectory_id"],
            "termination_reason": metadata["termination_reason"],
            "frames": int(len(valid)),
            "valid_transitions": int(valid.sum()),
            "physics_substeps_per_control_step": physics_substeps,
            "active_contact_frames_left_right": active_counts,
            "mode_counts_left_right": mode_counts,
            "normal_force_max_n": float(normal_force.max()),
            "tangential_force_max_n": float(
                np.linalg.norm(data["tangential_force"], axis=-1).max()),
            "contact_impulse_max_ns": float(impulse.max()),
            "slip_speed_max_m_s": float(slip_speed.max()),
            "verified": not errors,
            "errors": errors,
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-report", type=Path,
                        help="verified frozen_policy_bundles.json provenance report")
    args = parser.parse_args()
    expected_sources = None
    if args.bundle_report is not None:
        bundle_report = json.loads(args.bundle_report.read_text(encoding="utf-8"))
        if not bundle_report.get("all_verified"):
            raise ValueError("bundle report is not fully verified")
        expected_sources = {
            record["checkpoint_id"]: record["source_sha256"]
            for record in bundle_report["records"]
        }
    records = [audit(path, expected_sources) for path in args.paths]
    report = {
        "schema": 1,
        "trajectory_count": len(records),
        "all_verified": all(record["verified"] for record in records),
        "bundle_provenance_checked": expected_sources is not None,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "trajectory_count": len(records),
        "all_verified": report["all_verified"],
        "output": str(args.output),
    }, indent=2))
    if not report["all_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
