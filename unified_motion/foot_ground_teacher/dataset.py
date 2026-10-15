"""Immutable per-trajectory NPZ records and leakage-safe manifest helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .schema import CONTROL_DT_S, SCHEMA_VERSION, deterministic_split


REQUIRED_METADATA = (
    "checkpoint_id", "checkpoint_sha256", "seed", "reference_id",
    "trajectory_id", "start_phase", "friction", "mass_scale",
    "disturbance", "termination_reason", "control_dt_s",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_trajectory(path: str | Path, arrays: Mapping[str, np.ndarray],
                     metadata: Mapping, *, overwrite: bool = False) -> Path:
    """Write one complete trajectory atomically; never append partial rollouts."""
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    missing = [name for name in REQUIRED_METADATA if name not in metadata]
    if missing:
        raise ValueError(f"missing trajectory metadata: {missing}")
    if abs(float(metadata["control_dt_s"]) - CONTROL_DT_S) > 1e-9:
        raise ValueError("Teacher V1 requires 50Hz control")
    if not arrays or "valid_transition" not in arrays:
        raise ValueError("trajectory needs arrays and valid_transition")
    length = len(arrays["valid_transition"])
    if length < 1 or any(len(value) != length for value in arrays.values()):
        raise ValueError("all trajectory arrays must have equal positive length")
    if np.asarray(arrays["valid_transition"]).dtype != np.bool_:
        raise TypeError("valid_transition must be boolean")
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(
        {"schema": SCHEMA_VERSION, **metadata}, sort_keys=True))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)
    return target


def read_metadata(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError(f"{path} lacks metadata_json")
        metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema in {path}")
    return metadata


def build_manifest(paths, *, heldout_checkpoints=(), validation_checkpoints=(),
                   validation_fraction=.15, test_fraction=.15) -> dict:
    heldout_checkpoints = set(heldout_checkpoints)
    validation_checkpoints = set(validation_checkpoints)
    if heldout_checkpoints & validation_checkpoints:
        raise ValueError("validation and held-out checkpoint sets overlap")
    records = []
    for path in sorted(map(Path, paths)):
        metadata = read_metadata(path)
        checkpoint_id = metadata["checkpoint_id"]
        if checkpoint_id in heldout_checkpoints:
            split = "test"
        elif checkpoint_id in validation_checkpoints:
            split = "validation"
        elif validation_checkpoints:
            split = "train"
        else:
            split = deterministic_split(metadata,
                heldout_checkpoints=heldout_checkpoints,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction)
        records.append({
            "path": str(path.resolve()), "sha256": file_sha256(path),
            "split": split,
            "metadata": metadata,
        })
    test_sources = {r["metadata"]["checkpoint_id"] for r in records if r["split"] == "test"}
    train_sources = {r["metadata"]["checkpoint_id"] for r in records if r["split"] == "train"}
    if heldout_checkpoints and not set(heldout_checkpoints).issubset(test_sources):
        raise ValueError("not every held-out checkpoint appears in the test set")
    if set(heldout_checkpoints) & train_sources:
        raise RuntimeError("held-out checkpoint leaked into training")
    validation_sources = {
        r["metadata"]["checkpoint_id"] for r in records if r["split"] == "validation"}
    if validation_checkpoints and not validation_checkpoints.issubset(validation_sources):
        raise ValueError("not every validation checkpoint appears in the validation set")
    if validation_checkpoints & (train_sources | test_sources):
        raise RuntimeError("validation checkpoint leaked into train/test")
    return {"schema": SCHEMA_VERSION, "records": records,
            "heldout_checkpoints": sorted(heldout_checkpoints),
            "validation_checkpoints": sorted(validation_checkpoints),
            "checkpoint_source_exclusive": bool(validation_checkpoints)}
