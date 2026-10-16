#!/usr/bin/env python3
"""Merge four independently replayed Isaac branches after strict checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BRANCHES = ("base", "plus", "minus", "null")
HISTORY = ("student_observation", "privileged_current", "foot_current",
           "episode_start", "reference_intent")
MATCH_METADATA = ("checkpoint_id", "seed", "probe_frame", "joint_name",
                  "requested_delta_rad", "dynamic_friction", "context_steps")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for branch in BRANCHES:
        parser.add_argument(f"--{branch}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-tolerance", type=float, default=0.0)
    parser.add_argument("--target-tolerance-rad", type=float, default=1e-4)
    args = parser.parse_args()

    paths = {name: getattr(args, name).resolve() for name in BRANCHES}
    archives = {name: np.load(path, allow_pickle=False)
                for name, path in paths.items()}
    try:
        metadata = {name: json.loads(str(z["metadata_json"]))
                    for name, z in archives.items()}
        for name in BRANCHES:
            if metadata[name].get("schema") != 2 or \
                    metadata[name].get("single_branch") != name:
                raise ValueError(f"{name} input is not its schema-2 single branch")
        baseline = metadata["base"]
        for field in MATCH_METADATA:
            values = [metadata[name].get(field) for name in BRANCHES]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"branch metadata mismatch for {field}: {values}")

        history_max_abs_error = 0.0
        for field in HISTORY:
            reference = archives["base"][field]
            for name in BRANCHES[1:]:
                value = archives[name][field]
                if reference.shape != value.shape:
                    raise ValueError(f"history shape mismatch for {field}")
                if reference.dtype == np.bool_:
                    error = 0.0 if np.array_equal(reference, value) else 1.0
                else:
                    error = float(np.max(np.abs(reference - value)))
                history_max_abs_error = max(history_max_abs_error, error)
        # Actions are identical before the deliberately changed final command.
        reference_action = archives["base"]["executed_action"][:, :-1]
        for name in BRANCHES[1:]:
            value = archives[name]["executed_action"][:, :-1]
            history_max_abs_error = max(
                history_max_abs_error, float(np.max(np.abs(reference_action - value))))
        if history_max_abs_error > args.history_tolerance:
            raise ValueError(
                f"pre-action histories differ by {history_max_abs_error:.9g}, "
                f"tolerance is {args.history_tolerance:.9g}")

        def selected(field: str) -> np.ndarray:
            values = np.asarray([float(archives[name][field][0])
                                 for name in ("base", "plus", "minus")],
                                dtype=np.float32)
            return (values - values[0])[None]

        processed_delta = selected("processed_target_selected_rad")
        requested = float(baseline["requested_delta_rad"])
        if abs(float(processed_delta[0, 1]) - requested) > args.target_tolerance_rad or \
                abs(float(processed_delta[0, 2]) + requested) > args.target_tolerance_rad:
            raise ValueError("processed single-branch action differences are incorrect")

        arrays = {field: archives["base"][field] for field in HISTORY}
        arrays.update(
            executed_action=np.stack(
                [archives[name]["executed_action"][0]
                 for name in ("base", "plus", "minus")], axis=0)[None],
            target_contact=np.stack(
                [archives[name]["target_contact"][0]
                 for name in ("base", "plus", "minus")], axis=0)[None],
            target_contact_mode=np.stack(
                [archives[name]["target_contact_mode"][0]
                 for name in ("base", "plus", "minus")], axis=0)[None],
            target_tangent_velocity=np.stack(
                [archives[name]["target_tangent_velocity"][0]
                 for name in ("base", "plus", "minus")], axis=0)[None],
            target_normal_force=np.stack(
                [archives[name]["target_normal_force"][0]
                 for name in ("base", "plus", "minus")], axis=0)[None],
            processed_target_delta_rad=processed_delta,
            delayed_actuator_target_delta_rad=selected(
                "delayed_actuator_target_selected_rad"),
            applied_torque_delta_nm=selected("applied_torque_selected_nm"),
            actuator_time_lag_physics_steps=np.stack(
                [archives[name]["actuator_time_lag_physics_steps"]
                 for name in ("base", "plus", "minus")], axis=1),
            null_target_contact=archives["null"]["target_contact"],
            null_target_tangent_velocity=archives["null"]["target_tangent_velocity"],
            null_target_normal_force=archives["null"]["target_normal_force"],
            delta_rad=np.asarray([requested], dtype=np.float32),
        )
        merged_metadata = {
            **baseline, "schema": 3,
            "branches": ["base", "plus", "minus"],
            "null_replicate_branch": "independent_base_duplicate",
            "single_branch": None, "same_state_branching": True,
            "branch_execution_mode": "independent_process_replay",
            "pre_branch_state_max_abs_error": history_max_abs_error,
            "processed_target_delta_rad": processed_delta[0].tolist(),
            "delayed_actuator_target_delta_rad":
                arrays["delayed_actuator_target_delta_rad"][0].tolist(),
            "applied_torque_delta_nm": arrays["applied_torque_delta_nm"][0].tolist(),
            "source_files": {name: str(path) for name, path in paths.items()},
            "source_sha256": {name: digest(path) for name, path in paths.items()},
            "optimizer_updates": 0, "actor_updates": 0, "ppo_updates": 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp.npz")
        np.savez_compressed(
            temporary, **arrays,
            metadata_json=np.asarray(json.dumps(merged_metadata, sort_keys=True)))
        temporary.replace(args.output)
        print(json.dumps({"output": str(args.output.resolve()),
                          "history_max_abs_error": history_max_abs_error,
                          "processed_target_delta_rad": processed_delta[0].tolist()},
                         indent=2))
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == "__main__":
    main()
