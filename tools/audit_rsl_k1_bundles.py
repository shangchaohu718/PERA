#!/usr/bin/env python3
"""Audit safe frozen RSL bundles against their immutable source checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from unified_motion.rsl_k1_policy import (  # noqa: E402
    load_frozen_rsl_k1_bundle,
    sha256_file,
)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_root = Path(cfg["source_root"])
    bundle_root = Path(cfg["bundle_root"])
    records = []
    seen_ids: set[str] = set()
    seen_sources: set[Path] = set()
    for entry in cfg["checkpoints"]:
        checkpoint_id = entry["id"]
        if checkpoint_id in seen_ids:
            raise ValueError(f"duplicate checkpoint id: {checkpoint_id}")
        seen_ids.add(checkpoint_id)
        source = resolve(source_root, entry["path"]).resolve()
        bundle = resolve(bundle_root, entry["policy_bundle"]).resolve()
        if source in seen_sources:
            raise ValueError(f"duplicate source checkpoint: {source}")
        seen_sources.add(source)
        source_hash = sha256_file(source)
        policy = load_frozen_rsl_k1_bundle(bundle)
        errors = []
        if policy.metadata.checkpoint_id != checkpoint_id:
            errors.append("checkpoint_id mismatch")
        if policy.metadata.source_sha256 != source_hash:
            errors.append("source_sha256 mismatch")
        all_parameters_frozen = all(
            not parameter.requires_grad for parameter in policy.parameters())
        eval_mode = not policy.training
        if not all_parameters_frozen:
            errors.append("trainable parameter found")
        if not eval_mode:
            errors.append("policy is not in eval mode")
        records.append({
            "checkpoint_id": checkpoint_id,
            "split": entry["split"],
            "source": str(source),
            "source_sha256": source_hash,
            "source_iteration": policy.metadata.source_iteration,
            "bundle": str(bundle),
            "bundle_sha256": sha256_file(bundle),
            "safe_weights_only_load": True,
            "all_parameters_frozen": all_parameters_frozen,
            "eval_mode": eval_mode,
            "verified": not errors,
            "errors": errors,
        })
    report = {
        "schema": 1,
        "campaign": str(args.config.resolve()),
        "checkpoint_count": len(records),
        "all_verified": all(record["verified"] for record in records),
        "records": records,
    }
    if not report["all_verified"]:
        raise RuntimeError(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint_count": len(records),
        "all_verified": True,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
