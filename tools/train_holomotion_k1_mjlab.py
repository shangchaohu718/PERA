"""Train the original HoloMotion K1 policy with the mjlab task adapter.

The number of iterations and output directory are mandatory so an accidental
invocation cannot silently start the upstream 10,000-iteration schedule.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np

from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.holomotion_mjlab import (
    HoloMotionK1MJLabAdapter,
    HoloMotionOnnxContract,
    make_holomotion_k1_mjlab_cfg,
)
from unified_motion.holomotion_mjlab_ppo import HoloMotionMJLabPPO
from unified_motion.holomotion_physics_training import HoloMotionPhysicsMJLabPPO
from unified_motion.holomotion_mjlab_runner import (
    HoloMotionMJLabRunner,
    HoloMotionRunnerConfig,
)
from unified_motion.holomotion_mjlab_task import (
    HoloMotionFootRewardConfig,
    HoloMotionK1MJLabTask,
    HoloMotionPhysicsRandomizationConfig,
)
from unified_motion.holomotion_original_algorithm import (
    load_original_holomotion_k1_algorithm,
)
from unified_motion.models.physics import DynamicsModel
from unified_motion.robot_contract import get_robot_contract

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
        "--resume-optimizer",
        action="store_true",
        help="restore optimizer moments and continue after the checkpoint iteration",
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    motion_source = parser.add_mutually_exclusive_group(required=True)
    motion_source.add_argument(
        "--motion",
        type=Path,
        action="append",
        help="named motion NPZ; repeat this option to build a multi-motion library",
    )
    motion_source.add_argument(
        "--motion-dir",
        type=Path,
        help="folder of project NPZ windows produced by the HDF5 converter",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-teacher", action="store_true")
    parser.add_argument(
        "--matched-physics-task",
        action="store_true",
        help=("enable the same foot rewards and physics randomization as the "
              "physics-teacher run, without adding the physics network"),
    )
    parser.add_argument("--physics-width", type=int, default=64)
    parser.add_argument("--physics-memory-width", type=int, default=64)
    parser.add_argument("--physics-latent-width", type=int, default=8)
    parser.add_argument("--physics-hidden-dim", type=int, default=128)
    parser.add_argument("--physics-burn-in-steps", type=int, default=64)
    parser.add_argument("--physics-train-steps", type=int, default=128)
    parser.add_argument("--physics-confidence-warmup-steps", type=int, default=128)
    parser.add_argument("--physics-ema-decay", type=float, default=0.99)
    parser.add_argument("--physics-dynamics-epochs", type=int, default=4)
    parser.add_argument("--left-foot-slip-weight", type=float, default=0.02)
    parser.add_argument("--right-foot-slip-weight", type=float, default=0.02)
    parser.add_argument("--left-foot-impact-weight", type=float, default=0.01)
    parser.add_argument("--right-foot-impact-weight", type=float, default=0.01)
    parser.add_argument("--foot-impact-threshold-n", type=float, default=100.0)
    args = parser.parse_args()
    positive = (
        "iterations",
        "steps_per_env",
        "num_envs",
        "save_interval",
        "log_interval",
        "physics_burn_in_steps",
        "physics_train_steps",
        "physics_confidence_warmup_steps",
        "physics_dynamics_epochs",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("iteration, step, environment, save, and log counts must be positive")
    if any(getattr(args, name) < 2 for name in (
        "physics_width", "physics_memory_width", "physics_latent_width",
        "physics_hidden_dim",
    )):
        parser.error("physics model widths must be at least 2")
    if not args.device.startswith("cuda"):
        parser.error("mjlab training requires a CUDA device")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 0.0 < args.physics_ema_decay < 1.0:
        parser.error("physics EMA decay must be in (0,1)")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no mjlab training was started")
    from mjlab.envs import ManagerBasedRlEnv

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    contract = HoloMotionOnnxContract.from_onnx(args.onnx)
    if args.motion_dir is not None:
        library = MotionLibrary.from_project_folder(
            args.motion_dir, robot_id="k1_22dof", device=device
        )
    else:
        library = MotionLibrary.from_named_mjlab_files(
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
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg, device=str(device))
    matched_physics_task = args.physics_teacher or args.matched_physics_task
    task = HoloMotionK1MJLabTask(
        HoloMotionK1MJLabAdapter(
            K1MJLabBridge(env, reference_steps=contract.future_frames + 2), contract
        ),
        sampler,
        foot_reward_config=(
            HoloMotionFootRewardConfig(
                left_slip_weight=args.left_foot_slip_weight,
                right_slip_weight=args.right_foot_slip_weight,
                left_impact_weight=args.left_foot_impact_weight,
                right_impact_weight=args.right_foot_impact_weight,
                impact_threshold_n=args.foot_impact_threshold_n,
            )
            if matched_physics_task else None
        ),
        physics_randomization=(
            HoloMotionPhysicsRandomizationConfig()
            if matched_physics_task else None
        ),
    )
    try:
        algorithm = load_original_holomotion_k1_algorithm(
            holomotion_root=args.holomotion_root,
            run_dir=args.run_dir,
            checkpoint_name=args.checkpoint_name,
            device=device,
        )
        if args.physics_teacher:
            dynamics = DynamicsModel(
                get_robot_contract("k1_22dof"),
                physics_width=args.physics_width,
                memory_width=args.physics_memory_width,
                latent_width=args.physics_latent_width,
                hidden_dim=args.physics_hidden_dim,
                system_id_dim=task.physics_teacher_dim,
                teacher_student=True,
                uncertainty=True,
            ).to(device)
            trainer = HoloMotionPhysicsMJLabPPO(
                algorithm,
                task,
                dynamics,
                dynamics_burn_in_steps=args.physics_burn_in_steps,
                dynamics_train_steps=args.physics_train_steps,
                confidence_warmup_steps=args.physics_confidence_warmup_steps,
                dynamics_ema_decay=args.physics_ema_decay,
                dynamics_epochs=args.physics_dynamics_epochs,
            )
        else:
            trainer = HoloMotionMJLabPPO(algorithm, task)
        start_iteration = 0
        if args.resume_optimizer:
            resumed = trainer.load_optimizer_checkpoint(
                args.run_dir / f"{args.checkpoint_name}.pt"
            )
            start_iteration = resumed.iteration + 1
        runner_config = HoloMotionRunnerConfig(
            num_iterations=args.iterations,
            steps_per_env=args.steps_per_env,
            start_iteration=start_iteration,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
        )
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        source_config = args.run_dir.expanduser().resolve() / "config.yaml"
        destination_config = output / "config.yaml"
        if not destination_config.exists():
            shutil.copy2(source_config, destination_config)
        metadata = {
            "source_run": str(args.run_dir.expanduser().resolve()),
            "source_checkpoint": args.checkpoint_name,
            "motion_files": (
                [str(path.expanduser().resolve()) for path in args.motion]
                if args.motion is not None
                else None
            ),
            "motion_dir": (
                str(args.motion_dir.expanduser().resolve())
                if args.motion_dir is not None
                else None
            ),
            "resume_optimizer": args.resume_optimizer,
            "seed": args.seed,
            "physics_teacher": args.physics_teacher,
            "matched_physics_task": matched_physics_task,
            "physics_model": ({
                "physics_width": args.physics_width,
                "memory_width": args.physics_memory_width,
                "latent_width": args.physics_latent_width,
                "hidden_dim": args.physics_hidden_dim,
                "teacher_dim": task.physics_teacher_dim,
                "uncertainty": True,
                "burn_in_steps": args.physics_burn_in_steps,
                "train_steps": args.physics_train_steps,
                "confidence_warmup_steps": args.physics_confidence_warmup_steps,
                "ema_decay": args.physics_ema_decay,
                "dynamics_epochs": args.physics_dynamics_epochs,
            } if matched_physics_task else None),
            "foot_reward": ({
                "left_slip_weight": args.left_foot_slip_weight,
                "right_slip_weight": args.right_foot_slip_weight,
                "left_impact_weight": args.left_foot_impact_weight,
                "right_impact_weight": args.right_foot_impact_weight,
                "impact_threshold_n": args.foot_impact_threshold_n,
            } if args.physics_teacher else None),
            "runner": asdict(runner_config),
        }
        metadata_path = output / "mjlab_runner.json"
        if metadata_path.exists():
            raise FileExistsError(f"runner metadata already exists: {metadata_path}")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records = HoloMotionMJLabRunner(trainer, output, runner_config).run()
        print(json.dumps(records[-1], ensure_ascii=False, indent=2))
    finally:
        task.close()


if __name__ == "__main__":
    main()
