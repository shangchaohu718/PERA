#!/usr/bin/env python3
"""Audit an explicit Teacher z for collapse and decoder-only consistency."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from unified_motion.foot_ground_teacher.model import (  # noqa: E402
    FootGroundTeacher, FootGroundTeacherConfig,
)
from unified_motion.foot_ground_teacher.training import load_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    config = FootGroundTeacherConfig(**payload["model_config"])
    if config.latent_dim <= 0:
        raise ValueError("checkpoint has no explicit latent z")
    model = FootGroundTeacher(config).to(args.device).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["records"] if row["split"] == args.split]
    if not rows:
        raise ValueError(f"manifest has no {args.split} rows")

    samples = []
    decoder_max_abs_error = 0.0
    decoded_names = None
    with torch.no_grad():
        for row in rows:
            batch = load_batch(row["path"], args.device)
            output = model(
                batch.student_observation, batch.privileged_current,
                batch.executed_action, batch.foot_current, batch.episode_start,
                batch.reference_intent)
            decoded = model.decode_physics(output["latent_z"])
            decoded_names = sorted(decoded)
            for name, value in decoded.items():
                decoder_max_abs_error = max(
                    decoder_max_abs_error,
                    float((value - output[name]).abs().max().cpu()))
            samples.append(output["latent_z"][batch.valid_transition].cpu().float())
    z = torch.cat(samples)
    mean = z.mean(0)
    std = z.std(0, unbiased=False)
    centered = z - mean
    covariance = centered.T @ centered / max(1, z.shape[0])
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    normalized = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    nonzero = normalized > 0
    effective_rank = float(torch.exp(
        -(normalized[nonzero] * normalized[nonzero].log()).sum()))
    participation_ratio = float(
        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12))
    report = {
        "schema": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "seed": payload["seed"],
        "latent_contract": "all physical heads decode from z plus fixed foot-side ID",
        "latent_dim": config.latent_dim,
        "sample_count": int(z.shape[0]),
        "decoder_output_names": decoded_names,
        "decoder_forward_max_abs_error": decoder_max_abs_error,
        "per_dimension_mean": mean.tolist(),
        "per_dimension_std": std.tolist(),
        "active_dimensions_std_gt_1e-3": int((std > 1e-3).sum()),
        "active_dimensions_std_gt_1e-2": int((std > 1e-2).sum()),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "tanh_saturation_abs_gt_0.99_rate": float((z.abs() > .99).float().mean()),
        "finite": bool(torch.isfinite(z).all()),
        "deployment_claim": False,
        "actor_updates": 0,
        "ppo_updates": 0,
    }
    if not math.isfinite(effective_rank) or not report["finite"]:
        raise RuntimeError("latent audit produced non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "latent_dim", "sample_count", "decoder_forward_max_abs_error",
        "active_dimensions_std_gt_1e-2", "effective_rank",
        "tanh_saturation_abs_gt_0.99_rate")}, indent=2))


if __name__ == "__main__":
    main()
