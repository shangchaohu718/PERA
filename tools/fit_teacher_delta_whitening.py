"""Fit fixed Teacher delta statistics from training motions only.

The Actor and all Physics/Teacher models are frozen.  The resulting JSON stores
per-coordinate mean/std in normalized-core units for every configured direct
prediction horizon.  It is intended for the isolated B1 whitening ablation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from tools.audit_teacher_multihorizon import GROUP_NAMES, _valid_horizon
from unified_motion.algorithms.sequence_training import parameter_stamp
from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.holomotion_mjlab import (
    HoloMotionK1MJLabAdapter,
    HoloMotionOnnxContract,
    make_holomotion_k1_mjlab_cfg,
)
from unified_motion.holomotion_mjlab_task import (
    HoloMotionK1MJLabTask,
    HoloMotionPhysicsRandomizationConfig,
)
from unified_motion.k1_policy_runtime import resume_k1_trainer, runtime_configuration
from unified_motion.llm.validation import file_sha256
from unified_motion.privileged_teacher import core_groups, normalized_core

DEFAULT_HOLOMOTION_ROOT = PROJECT_ROOT / "third_party" / "HoloMotion" / "source"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--training-motion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument("--std-floor", type=float, default=1e-3)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=730201)
    parser.add_argument("--holomotion-root", type=Path, default=DEFAULT_HOLOMOTION_ROOT)
    args = parser.parse_args()
    args.horizons = tuple(args.horizons)
    if (args.num_envs < 1 or args.steps <= max(args.horizons) or args.batches < 1
            or args.std_floor <= 0):
        parser.error("invalid environment/step/batch/std-floor configuration")
    if tuple(sorted(set(args.horizons))) != args.horizons or args.horizons[0] != 1:
        parser.error("horizons must be unique, increasing, and begin at 1")
    return args


class Moments:
    def __init__(self, width, device):
        self.sum = torch.zeros(width, dtype=torch.float64, device=device)
        self.square_sum = torch.zeros_like(self.sum)
        self.count = torch.zeros(width, dtype=torch.long, device=device)

    def add(self, value, mask):
        value = value.double()
        self.sum += torch.where(mask, value, 0.0).sum((0, 1))
        self.square_sum += torch.where(mask, value.square(), 0.0).sum((0, 1))
        self.count += mask.sum((0, 1))

    def finish(self, floor):
        if not bool((self.count > 0).all()):
            raise ValueError("some delta coordinates have no valid training samples")
        mean = self.sum / self.count
        variance = (self.square_sum / self.count - mean.square()).clamp_min(0.0)
        raw_std = variance.sqrt()
        std = raw_std.clamp_min(floor)
        return mean.float().cpu(), std.float().cpu(), raw_std.float().cpu(), self.count.cpu()


@torch.no_grad()
def add_rollout(moments, trainer, rollout, horizons):
    ids = rollout.privileged_teacher_env_ids
    current = rollout.privileged_teacher_core
    if ids is None or current is None or len(ids) != rollout.actions.shape[0]:
        raise ValueError("statistics fitting requires complete privileged truth")
    starts = rollout.physics_episode_start.index_select(0, ids)
    dones = rollout.dones.squeeze(-1).index_select(0, ids)
    truth = normalized_core(
        rollout.next_physics_observation, trainer.dynamics.robot).index_select(0, ids)
    core_valid = rollout.next_physics_observation.core_valid.index_select(0, ids)
    for horizon in horizons:
        length = current.shape[1] - horizon + 1
        valid_transition = _valid_horizon(starts, dones, horizon)
        future = truth[:, horizon - 1:length + horizon - 1]
        valid = (core_valid[:, horizon - 1:length + horizon - 1]
                 & valid_transition.unsqueeze(-1))
        moments[horizon].add(future - current[:, :length], valid)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; whitening fit was not started")
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    paths = sorted(args.training_motion_dir.expanduser().resolve().rglob("*.npz"))
    if not paths:
        raise FileNotFoundError("training-motion-dir contains no NPZ files")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runtime = runtime_configuration(payload)
    contract = HoloMotionOnnxContract(**runtime["task_contract"])
    library = MotionLibrary.from_named_mjlab_files(paths, robot_id="k1_22dof", device=device)
    offsets = tuple(index * 0.02 for index in range(contract.future_frames + 2))
    sampler = MotionSampler(library, num_envs=args.num_envs, step_dt=.02,
                            future_offsets_s=offsets)
    cfg = make_holomotion_k1_mjlab_cfg(args.asset_root, contract, num_envs=args.num_envs)
    cfg.seed = args.seed
    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        task = HoloMotionK1MJLabTask(
            HoloMotionK1MJLabAdapter(
                K1MJLabBridge(env, reference_steps=contract.future_frames + 2), contract),
            sampler,
            physics_randomization=HoloMotionPhysicsRandomizationConfig(
                **runtime["physics_randomization"]),
        )
        trainer = resume_k1_trainer(args.checkpoint, task,
                                    holomotion_root=args.holomotion_root)
        width = 9 + 2 * trainer.dynamics.robot.dof
        moments = {horizon: Moments(width, device) for horizon in args.horizons}
        frozen = (parameter_stamp(trainer.actor), parameter_stamp(trainer.dynamics),
                  parameter_stamp(trainer.privileged_teacher))
        minimum_remaining = args.steps * sampler.step_dt + float(sampler.offsets_s[-1])
        for _ in range(args.batches):
            trainer.reset(random_start=True, min_remaining_s=minimum_remaining)
            add_rollout(moments, trainer, trainer.collect(args.steps), args.horizons)
        if frozen != (parameter_stamp(trainer.actor), parameter_stamp(trainer.dynamics),
                      parameter_stamp(trainer.privileged_teacher)):
            raise RuntimeError("statistics fitting unexpectedly changed model parameters")
        statistics, summaries = {}, {}
        sections = core_groups(width)
        for horizon in args.horizons:
            mean, std, raw_std, count = moments[horizon].finish(args.std_floor)
            statistics[str(horizon)] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
            summaries[str(horizon)] = {
                name: {
                    "samples_min": int(count[section].min()),
                    "mean_abs_max": float(mean[section].abs().max()),
                    "raw_std_min": float(raw_std[section].min()),
                    "raw_std_max": float(raw_std[section].max()),
                    "clamped_dimensions": int((raw_std[section] < args.std_floor).sum()),
                }
                for name, section in zip(GROUP_NAMES, sections, strict=True)
            }
        document = {
            "schema_version": 1,
            "kind": "teacher_delta_whitening_statistics",
            "units": "normalized_core_delta",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "training_motion_dir": str(args.training_motion_dir.resolve()),
            "training_motion_files": [path.name for path in paths],
            "training_motion_count": len(paths),
            "fit_seed": args.seed,
            "num_envs": args.num_envs,
            "steps_per_batch": args.steps,
            "batches": args.batches,
            "std_floor": args.std_floor,
            "horizons": list(args.horizons),
            "weights_unchanged": True,
            "statistics": statistics,
            "summary": summaries,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({key: document[key] for key in (
            "training_motion_count", "fit_seed", "std_floor", "summary")},
            ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
