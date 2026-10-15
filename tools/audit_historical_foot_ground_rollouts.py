#!/usr/bin/env python3
"""Audit old Moonwalk NPZ files for Foot-Ground Teacher V1 label sufficiency."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import zipfile


# Exact names are deliberate: similarly named fixed foot sites and aggregate
# forces are useful diagnostics, but are not substitutes for raw contact data.
REQUIRED_DIRECT_FIELDS = {
    "student_joint_position": "22-D current joint positions",
    "student_joint_velocity": "22-D current joint velocities",
    "student_imu_angular_velocity_body": "current body-frame IMU angular velocity",
    "student_imu_projected_gravity_body": "current body-frame projected gravity",
    "student_estimated_joint_torque": "22-D estimated/applied joint torque",
    "student_previous_executed_action": "previous executed action",
    "executed_action": "current executed action",
    "contact_point_world": "raw physical contact point(s)",
    "contact_normal_world": "raw physical contact normal(s)",
    "contact_force_world": "raw per-contact force vector(s)",
    "foot_body_position_world": "foot rigid-body origin",
    "foot_body_linear_velocity_world": "foot rigid-body linear velocity",
    "foot_body_angular_velocity_world": "foot rigid-body angular velocity",
    "ground_velocity_world": "ground velocity at the contact point",
    "next_joint_velocity": "next-step 22-D joint velocity",
    "next_root_angular_velocity_body": "next-step body-frame root angular velocity",
    "valid_transition": "terminal/reset-safe transition mask",
    "termination_reason": "episode termination reason",
}

KNOWN_DIAGNOSTIC_PROXIES = {
    "action": "policy action; execution/action-manager semantics are not recorded",
    "foot_contact_force_xyz_n": "per-foot aggregate force, not raw per-contact records",
    "foot_contact_force_norm_n": "force magnitude; does not separate normal/tangent",
    "foot_ground_compatible_contact": "derived contact Boolean",
    "foot_point_actual": "fixed tracked foot sites, not physical contact points",
    "foot_point_velocity_actual": "fixed tracked site velocity, not selected contact-point velocity",
    "foot_site_min_height_m": "fixed-site geometry diagnostic",
    "joint_pos_actual": "joint position only; no paired joint velocity/torque",
    "root_quat_actual": "root orientation diagnostic",
    "role_code": "reference-phase foot-role annotation",
}

RELATED_KEY_TOKENS = (
    "action", "contact", "force", "friction", "impulse", "joint", "normal",
    "policy_observation", "root", "seed", "termination", "torque", "velocity",
)

PARTIALLY_RECOVERABLE_CONCEPTS = {
    "joint_position": ("joint_pos_actual",),
    "joint_velocity": ("joint_vel_actual",),
    "root_angular_velocity_world": ("root_ang_vel_actual_w",),
    "root_orientation_for_body_frame_conversion": ("root_quat_actual",),
    "applied_torque": ("applied_torque",),
    "executed_action": ("action_executed", "action_manager_processed"),
    "policy_observation": ("policy_observation",),
    "native_contact_indicator": (
        "foot_contact_region_native_valid", "foot_ground_compatible_contact"),
    "aggregate_normal_force": ("foot_contact_region_normal_force_n",),
    "aggregate_normal_impulse": ("foot_contact_region_normal_impulse_ns",),
}

IRRECOVERABLE_FROM_RECORDED_ARRAYS = {
    "raw_contact_point": "fixed tracked foot sites are not solver contact points",
    "raw_contact_normal": "no solver contact normal is recorded",
    "raw_tangential_contact_force": "only aggregate XYZ or normal force is recorded",
    "contact_point_velocity": (
        "foot_point_velocity_actual is a fixed site velocity and the required "
        "rigid-body angular velocity/contact lever arm pair is absent"),
    "moving_ground_velocity": "ground velocity and ground-body identity are absent",
    "substep_contact_records": "only control-frame aggregates are recorded",
    "termination_safe_transition_mask": "termination/reset provenance is absent",
    "randomization_metadata": "seed, friction, mass and disturbances are not complete per trajectory",
}


def npz_keys(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.evaluation_root.rglob("*.npz"))
    key_files: dict[str, list[str]] = defaultdict(list)
    unreadable = []
    file_keys: dict[str, set[str]] = {}
    for path in files:
        try:
            keys = npz_keys(path)
        except (OSError, zipfile.BadZipFile) as error:
            unreadable.append({"path": str(path), "error": str(error)})
            continue
        file_keys[str(path)] = keys
        for key in keys:
            key_files[key].append(str(path))

    direct_coverage = {
        field: {
            "description": description,
            "file_count": len(key_files.get(field, [])),
            "examples": key_files.get(field, [])[:3],
        }
        for field, description in REQUIRED_DIRECT_FIELDS.items()
    }
    proxy_coverage = {
        field: {
            "limitation": limitation,
            "file_count": len(key_files.get(field, [])),
            "examples": key_files.get(field, [])[:3],
        }
        for field, limitation in KNOWN_DIAGNOSTIC_PROXIES.items()
    }
    complete_files = [
        path for path, keys in file_keys.items()
        if REQUIRED_DIRECT_FIELDS.keys() <= keys
    ]
    related_key_inventory = {
        key: {
            "file_count": len(paths),
            "examples": paths[:3],
        }
        for key, paths in sorted(key_files.items())
        if any(token in key.lower() for token in RELATED_KEY_TOKENS)
    }
    partial_coverage = []
    for path, keys in file_keys.items():
        present = {
            concept: next((alias for alias in aliases if alias in keys), None)
            for concept, aliases in PARTIALLY_RECOVERABLE_CONCEPTS.items()
        }
        score = sum(alias is not None for alias in present.values())
        partial_coverage.append({
            "path": path,
            "recoverable_concept_count": score,
            "recoverable_concepts": {
                concept: alias for concept, alias in present.items() if alias is not None},
        })
    partial_coverage.sort(key=lambda row: (-row["recoverable_concept_count"], row["path"]))
    report = {
        "schema": 1,
        "evaluation_root": str(args.evaluation_root.resolve()),
        "npz_file_count": len(files),
        "readable_file_count": len(file_keys),
        "unreadable": unreadable,
        "teacher_v1_direct_field_coverage": direct_coverage,
        "known_diagnostic_proxy_coverage": proxy_coverage,
        "related_key_inventory": related_key_inventory,
        "best_partial_diagnostic_files": partial_coverage[:20],
        "partial_diagnostic_concept_count": len(PARTIALLY_RECOVERABLE_CONCEPTS),
        "irrecoverable_from_recorded_arrays": IRRECOVERABLE_FROM_RECORDED_ARRAYS,
        "complete_teacher_v1_files": complete_files,
        "historical_data_accepted_for_teacher_v1_training": bool(complete_files),
        "verdict": (
            "At least one historical file has every direct V1 field."
            if complete_files else
            "No historical file contains the complete direct V1 input/label contract. "
            "Diagnostic proxies must not be promoted to physical ground truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "npz_file_count": len(files),
        "complete_teacher_v1_files": len(complete_files),
        "accepted": bool(complete_files),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
