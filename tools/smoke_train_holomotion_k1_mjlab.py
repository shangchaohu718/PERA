"""Run one in-memory original HoloMotion PPO update against real mjlab physics.

This is a connectivity smoke test, not a long training command. It never
writes a checkpoint and defaults to one environment and eight control steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.holomotion_mjlab import (
    HoloMotionK1MJLabAdapter,
    HoloMotionOnnxContract,
    make_holomotion_k1_mjlab_cfg,
)
from unified_motion.holomotion_mjlab_ppo import HoloMotionMJLabPPO
from unified_motion.holomotion_mjlab_task import HoloMotionK1MJLabTask
from unified_motion.holomotion_original_algorithm import (
    load_original_holomotion_k1_algorithm,
)

DEFAULT_HOLOMOTION_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "HoloMotion" / "source"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holomotion-root",
        type=Path,
        default=DEFAULT_HOLOMOTION_ROOT,
        help="HoloMotion source root (default: vendored audited snapshot)",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="model_9999")
    parser.add_argument(
        "--load-optimizer",
        action="store_true",
        help="also restore optimizer moments from <run-dir>/<checkpoint-name>.pt",
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("--steps and --num-envs must be positive")
    if not args.device.startswith("cuda"):
        parser.error("mjlab requires a CUDA device")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no mjlab simulation was started")
    from mjlab.envs import ManagerBasedRlEnv

    device = torch.device(args.device)
    contract = HoloMotionOnnxContract.from_onnx(args.onnx)
    library = MotionLibrary.from_named_mjlab_file(
        args.motion, robot_id="k1_22dof", device=device
    )
    offsets = tuple(index * 0.02 for index in range(contract.future_frames + 2))
    sampler = MotionSampler(
        library,
        num_envs=args.num_envs,
        step_dt=0.02,
        future_offsets_s=offsets,
    )
    cfg = make_holomotion_k1_mjlab_cfg(
        args.asset_root, contract, num_envs=args.num_envs
    )
    env = ManagerBasedRlEnv(cfg, device=str(device))
    bridge = K1MJLabBridge(env, reference_steps=contract.future_frames + 2)
    adapter = HoloMotionK1MJLabAdapter(bridge, contract)
    task = HoloMotionK1MJLabTask(adapter, sampler)
    try:
        algorithm = load_original_holomotion_k1_algorithm(
            holomotion_root=args.holomotion_root,
            run_dir=args.run_dir,
            checkpoint_name=args.checkpoint_name,
            device=device,
        )
        trainer = HoloMotionMJLabPPO(algorithm, task)
        resume = None
        if args.load_optimizer:
            resume = trainer.load_optimizer_checkpoint(
                args.run_dir / f"{args.checkpoint_name}.pt"
            )
        rollout = trainer.collect(args.steps)
        actor_parameter = next(algorithm.actor.parameters())
        actor_before = actor_parameter.detach().clone()
        critic_parameter = next(algorithm.critic.parameters())
        critic_before = critic_parameter.detach().clone()
        metrics = trainer.update(rollout)
        result = {
            "device": str(device),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "reward_mean": float(rollout.rewards.mean()),
            "done_count": int(rollout.dones.sum()),
            "optimizer_resume_iteration": (
                resume.iteration if resume is not None else None
            ),
            "metrics": metrics.__dict__,
            "actor_parameter_max_change": float(
                (actor_parameter.detach() - actor_before).abs().max()
            ),
            "critic_parameter_max_change": float(
                (critic_parameter.detach() - critic_before).abs().max()
            ),
        }
        numeric = torch.tensor(
            [
                result["reward_mean"],
                *result["metrics"].values(),
                result["actor_parameter_max_change"],
                result["critic_parameter_max_change"],
            ],
            device=device,
        )
        if not torch.isfinite(numeric).all().item():
            raise RuntimeError(f"PPO smoke test produced non-finite values: {result}")
        if result["actor_parameter_max_change"] == 0.0:
            raise RuntimeError("actor optimizer did not change the tested parameter")
        if result["critic_parameter_max_change"] == 0.0:
            raise RuntimeError("critic optimizer did not change the tested parameter")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        task.close()


if __name__ == "__main__":
    main()
