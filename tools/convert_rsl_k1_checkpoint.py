#!/usr/bin/env python3
"""Convert a native Isaac Lab/RSL-RL K1 checkpoint to a safe frozen bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from unified_motion.rsl_k1_policy import (  # noqa: E402
    load_frozen_rsl_k1_bundle,
    load_rsl_k1_checkpoint,
    save_frozen_rsl_k1_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-id", required=True,
        help="globally unique stable identity stored in the safe bundle")
    parser.add_argument("--trusted-source", action="store_true",
        help="allow pickle fallback for trusted checkpoints containing NumPy RNG state")
    parser.add_argument("--verify-rollout", type=Path,
        help="historical NPZ containing policy_observation and expected Actor output")
    parser.add_argument("--expected-action-key", default="policy_action_unfiltered")
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()

    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("output must not overwrite the native source checkpoint")

    source = load_rsl_k1_checkpoint(
        args.checkpoint, trusted_source=args.trusted_source,
        checkpoint_id=args.checkpoint_id,
    )
    save_frozen_rsl_k1_bundle(source, args.output)
    reloaded = load_frozen_rsl_k1_bundle(args.output)
    report = {
        "output": str(args.output),
        "checkpoint_id": source.metadata.checkpoint_id,
        "source_sha256": source.metadata.source_sha256,
        "source_iteration": source.metadata.source_iteration,
        "all_parameters_frozen": all(not p.requires_grad for p in reloaded.parameters()),
    }
    if args.verify_rollout:
        with np.load(args.verify_rollout, allow_pickle=False) as rollout:
            if "policy_observation" not in rollout or args.expected_action_key not in rollout:
                raise ValueError("verification NPZ lacks policy_observation or expected action")
            observation = torch.from_numpy(rollout["policy_observation"])
            expected = torch.from_numpy(rollout[args.expected_action_key])
        actual = reloaded.act(observation)
        error = (actual - expected).abs()
        report.update({
            "verification_rollout": str(args.verify_rollout),
            "expected_action_key": args.expected_action_key,
            "frames": int(observation.shape[0]),
            "max_abs_action_error": float(error.max()),
            "mean_abs_action_error": float(error.mean()),
            "atol": args.atol,
            "verified": bool(torch.allclose(actual, expected, atol=args.atol, rtol=0)),
        })
        if not report["verified"]:
            raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
