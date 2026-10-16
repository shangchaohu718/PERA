#!/usr/bin/env python3
"""Strict offline smoke test for the official HoloMotion G1 checkpoint.

This command never downloads anything.  It requires an existing full
HoloMotion checkout and checkpoint package, reconstructs the official network,
strict-loads both safetensors files, and runs one Actor/Critic forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unified_motion.holomotion_original_algorithm import (  # noqa: E402
    flat_observations_to_tensordict,
    load_original_holomotion_algorithm,
    original_holomotion_observation_dims,
)
from unified_motion.robot_contract import get_robot_contract  # noqa: E402


def _arguments() -> argparse.Namespace:
    default_root = PROJECT_ROOT.parent / "HoloMotion"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holomotion-root",
        type=Path,
        default=default_root,
        help=f"Full official HoloMotion checkout (default: {default_root})",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Checkpoint package directory (default: ROOT/checkpoints/holomotion_v1.4)",
    )
    parser.add_argument("--checkpoint-name", default="model_14000")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = args.holomotion_root.expanduser().resolve()
    run = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else root / "checkpoints" / "holomotion_v1.4"
    )
    config_path = run / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"official checkpoint config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot = config.get("robot", {})
    expected = get_robot_contract("g1_29dof")
    joint_names = tuple(robot.get("dof_names", ()))
    if joint_names != expected.joint_names:
        raise ValueError("checkpoint joint order is not the official G1 29-DoF contract")
    if (
        int(robot.get("actions_dim", -1)) != expected.dof
        or int(robot.get("dof_obs_size", -1)) != expected.dof
        or int(robot.get("num_bodies", -1)) != 30
    ):
        raise ValueError("checkpoint morphology metadata is not G1 29-DoF/30-body")

    algorithm = load_original_holomotion_algorithm(
        holomotion_root=root,
        run_dir=run,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )
    algorithm.actor.eval()
    algorithm.critic.eval()
    actor_dim, critic_dim = original_holomotion_observation_dims(29, 30)
    data = flat_observations_to_tensordict(
        torch.zeros(1, actor_dim, device=args.device),
        torch.zeros(1, critic_dim, device=args.device),
        dof=29,
        num_bodies=30,
    )
    with torch.inference_mode():
        actor = algorithm.actor(data, mode="inference", update_obs_norm=False)
        critic = algorithm.critic(data, update_obs_norm=False)
    actions = actor["actions"]
    values = critic["values"]
    if actions.shape != (1, 29) or values.shape != (1, 1):
        raise RuntimeError("official HoloMotion forward output shape mismatch")
    if not torch.isfinite(actions).all() or not torch.isfinite(values).all():
        raise RuntimeError("official HoloMotion forward produced NaN/Inf")
    print(json.dumps({
        "status": "passed",
        "checkpoint": str(run / f"{args.checkpoint_name}.pt"),
        "actor_observation_dim": actor_dim,
        "critic_observation_dim": critic_dim,
        "action_shape": list(actions.shape),
        "value_shape": list(values.shape),
        "device": args.device,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
