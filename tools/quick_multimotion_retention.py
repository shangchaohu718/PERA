"""Quick deterministic one-environment-per-motion checkpoint retention audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from unified_motion.k1_policy_runtime import (
    frozen_k1_policy,
    load_k1_checkpoint,
    runtime_configuration,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion", type=Path, action="append")
    source.add_argument("--motion-dir", type=Path)
    parser.add_argument("--samples-per-motion", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_motion < 1:
        parser.error("--samples-per-motion must be positive")
    if args.motion_dir is not None:
        groups = {}
        for path in args.motion_dir.expanduser().resolve().glob("*_len_300.npz"):
            name, suffix = path.name.split("__start_", 1)
            start = int(suffix.split("_len_", 1)[0])
            groups.setdefault(name, []).append((start, path))
        selected = []
        for name in sorted(groups):
            paths = [path for _, path in sorted(groups[name])]
            if args.samples_per_motion == 1:
                indices = [len(paths) // 2]
            else:
                indices = [round(i * (len(paths) - 1) / (args.samples_per_motion - 1))
                           for i in range(args.samples_per_motion)]
            selected.extend(paths[index] for index in dict.fromkeys(indices))
        if not selected:
            parser.error("--motion-dir contains no complete 300-frame windows")
        args.motion = selected
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real mjlab evaluation requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from mjlab.envs import ManagerBasedRlEnv
    from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
    from unified_motion.envs.mjlab_k1 import K1MJLabBridge
    from unified_motion.holomotion_mjlab import (
        HoloMotionK1MJLabAdapter,
        HoloMotionOnnxContract,
        make_holomotion_k1_mjlab_cfg,
    )
    from unified_motion.holomotion_mjlab_task import HoloMotionK1MJLabTask

    loaded = load_k1_checkpoint(args.checkpoint, device=args.device)
    saved = runtime_configuration(loaded.payload).get("task_contract")
    if saved is None:
        raise ValueError("checkpoint has no saved task contract")
    contract = HoloMotionOnnxContract(**saved)
    library = MotionLibrary.from_named_mjlab_files(
        args.motion, robot_id="k1_22dof", device=args.device)
    count = len(args.motion)
    if count != len(library.clip_names):
        raise ValueError("motion library did not preserve one clip per input")
    lengths = library.clip_lengths.detach().cpu().tolist()
    if len(set(lengths)) != 1:
        raise ValueError(f"quick audit requires equal clip lengths, got {sorted(set(lengths))}")
    steps = int(lengths[0]) - 1
    sampler = MotionSampler(
        library,
        num_envs=count,
        step_dt=.02,
        future_offsets_s=tuple(i * .02 for i in range(contract.future_frames + 2)),
    )
    cfg = make_holomotion_k1_mjlab_cfg(str(args.asset_root), contract, num_envs=count)
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        task = HoloMotionK1MJLabTask(
            HoloMotionK1MJLabAdapter(
                K1MJLabBridge(env, reference_steps=contract.future_frames + 2), contract),
            sampler,
        )
        policy = frozen_k1_policy(loaded, ablation="full")
        policy.reset()
        clip_ids = torch.arange(count, device=args.device, dtype=torch.long)
        observation = task.reset(
            clip_ids=clip_ids,
            start_times_s=torch.zeros(count, device=args.device),
        )
        starts = torch.ones(count, device=args.device, dtype=torch.bool)
        alive = torch.ones_like(starts)
        failed = torch.zeros_like(starts)
        completed = torch.zeros_like(starts)
        survived = torch.zeros(count, device=args.device, dtype=torch.long)
        reward_sum = torch.zeros(count, device=args.device)
        metric_names = (
            "root_position_error_m",
            "root_orientation_error_rad",
            "root_linear_velocity_error_mps",
            "keybody_linear_velocity_rmse_mps",
        )
        metric_sum = {name: torch.zeros(count, device=args.device) for name in metric_names}
        metric_count = torch.zeros(count, device=args.device)
        for step in range(steps):
            action = policy.act(observation.actor, None, starts)
            transition = task.step(action)
            policy.record_action(getattr(task, "last_executed_action", action))
            active = alive.clone()
            reward_sum += torch.where(active, transition.reward, 0.)
            survived += active.long()
            for name in metric_names:
                value = transition.terms[name]
                metric_sum[name] += torch.where(active, value, 0.)
            metric_count += active
            physical = transition.physical_failure
            finished = transition.motion_finished
            failed |= active & physical
            completed |= active & finished & ~physical
            alive &= ~(physical | finished)
            starts = torch.zeros_like(starts)
            if not alive.any().item():
                break
            if step + 1 < steps:
                observation = task.observe()
        rows = []
        for index, path in enumerate(args.motion):
            denom = max(1., float(metric_count[index]))
            rows.append({
                "motion": str(path.resolve()),
                "clip_name": library.clip_names[index],
                "steps_survived": int(survived[index]),
                "physical_failure": bool(failed[index]),
                "reference_completion": bool(completed[index]),
                "reward_mean": float(reward_sum[index] / denom),
                "tracking": {
                    name: float(total[index] / denom)
                    for name, total in metric_sum.items()
                },
            })
        report = {
            "schema": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "seed": args.seed,
            "steps_per_clip": steps,
            "num_clips": count,
            "physical_failures": sum(row["physical_failure"] for row in rows),
            "reference_completions": sum(row["reference_completion"] for row in rows),
            "clips": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
