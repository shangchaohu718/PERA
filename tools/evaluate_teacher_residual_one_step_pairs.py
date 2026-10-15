"""Same-state one-step counterfactual for a frozen K1 Teacher residual.

Each source environment is cloned into a paired destination environment inside
one MJWarp batch.  The source executes the base mean and the destination
executes the Teacher-corrected mean, so the only intended difference during the
next control interval is the residual action.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
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
    frozen_k1_policy, load_k1_checkpoint, runtime_configuration,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--task-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.pairs < 1 or args.seed < 0:
        parser.error("pairs must be positive and seed non-negative")
    return args


@torch.no_grad()
def pair_initial_conditions(task, observation, pairs: int):
    src = torch.arange(pairs, device=task.device)
    dst = src + pairs
    sim = task.bridge.env.sim
    entity = task.bridge.entity

    # Copy every randomized physical field used by this task.
    for name in ("geom_friction", "body_mass", "body_inertia"):
        value = getattr(sim.model, name)
        value[dst] = value[src]
    task._friction[dst] = task._friction[src]
    task._trunk_mass_scale[dst] = task._trunk_mass_scale[src]

    data = entity.data
    entity.write_root_link_pose_to_sim(data.root_link_pose_w[src].clone(), env_ids=dst)
    entity.write_root_link_velocity_to_sim(data.root_link_vel_w[src].clone(), env_ids=dst)
    entity.write_joint_state_to_sim(
        data.joint_pos[src].clone(), data.joint_vel[src].clone(), env_ids=dst)

    sampler = task.sampler
    for name in ("clip_ids", "start_times_s", "step_counts", "yaw_delta", "xy_shift"):
        value = getattr(sampler, name)
        value[dst] = value[src]
    adapter = task.adapter
    adapter.last_action_onnx[dst] = adapter.last_action_onnx[src]
    adapter.previous_reference_root_pose[dst] = adapter.previous_reference_root_pose[src]
    adapter._previous_reference_valid[dst] = adapter._previous_reference_valid[src]

    from mjlab.managers.event_manager import RecomputeLevel
    sim.recompute_constants(RecomputeLevel.set_const)
    sim.forward(); sim.sense()

    # reset() already produced an observation.  Rebuild it after cloning so the
    # paired policy inputs are derived from the copied state, not stale buffers.
    task._pending_reference = None
    task._pending_body_reference = None
    adapter._pending_reference_root_pose = None
    observation = task.observe()
    actor_difference = float((observation.actor[src] - observation.actor[dst]).abs().max())
    critic_difference = float((observation.critic[src] - observation.critic[dst]).abs().max())
    if actor_difference > 1.0e-5 or critic_difference > 1.0e-5:
        raise RuntimeError(
            f"paired observations differ before action: actor={actor_difference}, "
            f"critic={critic_difference}"
        )
    return observation, src, dst, actor_difference, critic_difference


def paired_summary(base: torch.Tensor, corrected: torch.Tensor, *, lower_is_better: bool):
    delta = corrected - base
    improved = delta < 0 if lower_is_better else delta > 0
    return {
        "base_mean": float(base.mean()),
        "corrected_mean": float(corrected.mean()),
        "corrected_minus_base_mean": float(delta.mean()),
        "corrected_better_fraction": float(improved.float().mean()),
    }


def slice_batch(value, ids):
    if isinstance(value, torch.Tensor):
        return value.index_select(0, ids)
    if is_dataclass(value):
        return replace(value, **{
            field.name: slice_batch(getattr(value, field.name), ids)
            for field in fields(value)
        })
    if value is None:
        return None
    raise TypeError(f"unsupported batched value {type(value)!r}")


class PairedSourceTask:
    """Expose only the source half to the frozen privileged policy."""

    def __init__(self, task, ids):
        self.task = task
        self.ids = ids

    def privileged_physics_observation(self):
        return slice_batch(self.task.privileged_physics_observation(), self.ids)

    def physics_teacher_target(self):
        return self.task.physics_teacher_target().index_select(0, self.ids)


def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real paired mjlab evaluation requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from mjlab.envs import ManagerBasedRlEnv
    from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
    from unified_motion.envs.mjlab_k1 import K1MJLabBridge
    from unified_motion.holomotion_mjlab import (
        HoloMotionK1MJLabAdapter, HoloMotionOnnxContract, make_holomotion_k1_mjlab_cfg,
    )
    from unified_motion.holomotion_mjlab_task import (
        HoloMotionK1MJLabTask, HoloMotionPhysicsRandomizationConfig,
    )
    from unified_motion.llm.validation import file_sha256

    loaded = load_k1_checkpoint(args.checkpoint, device=args.device)
    config = runtime_configuration(loaded.payload)
    if config.get("physics_control_mode") != "teacher":
        raise ValueError("checkpoint is not a Teacher-controlled residual")
    contract = HoloMotionOnnxContract.from_json(args.task_contract)
    saved = config.get("task_contract")
    if saved is not None and HoloMotionOnnxContract(**saved) != contract:
        raise ValueError("task contract differs from checkpoint")
    total = 2 * args.pairs
    library = MotionLibrary.from_named_mjlab_files(
        args.motion, robot_id="k1_22dof", device=args.device)
    sampler = MotionSampler(library, num_envs=total, step_dt=.02,
        future_offsets_s=tuple(i*.02 for i in range(contract.future_frames+2)))
    cfg = make_holomotion_k1_mjlab_cfg(
        str(args.asset_root), contract, num_envs=total)
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        task = HoloMotionK1MJLabTask(
            HoloMotionK1MJLabAdapter(K1MJLabBridge(
                env, reference_steps=contract.future_frames+2), contract), sampler,
            physics_randomization=(None if config.get("physics_randomization") is None else
                HoloMotionPhysicsRandomizationConfig(**config["physics_randomization"])),
        )
        observation = task.reset(
            random_start=True,
            min_remaining_s=float(task.sampler.offsets_s[-1]) + task.sampler.step_dt,
        )
        observation, src, dst, actor_diff, critic_diff = pair_initial_conditions(
            task, observation, args.pairs)
        policy = frozen_k1_policy(loaded, ablation="full")
        policy.reset()
        source_observation = slice_batch(observation, src)
        starts = torch.ones(args.pairs, device=task.device, dtype=torch.bool)
        corrected_action = policy.act_privileged(
            source_observation, PairedSourceTask(task, src), starts)
        base_action = policy.last_base_action
        residual = policy.last_residual
        # Both branches use the one policy evaluation from the source state.
        # This prevents unobserved destination contact-cache differences from
        # changing Teacher output before the intended action intervention.
        mixed_action = torch.cat((base_action, corrected_action), dim=0)
        transition = task.step(mixed_action)

        metrics = {
            "reward": paired_summary(
                transition.reward[src], transition.reward[dst], lower_is_better=False),
        }
        for name in (
            "root_position_error_m", "root_orientation_error_rad",
            "root_linear_velocity_error_mps", "root_angular_velocity_error_radps",
            "keybody_linear_velocity_rmse_mps", "keybody_angular_velocity_rmse_radps",
            "action_rate_l2", "joint_pos_limits", "undesired_contacts",
        ):
            value = transition.terms[name]
            metrics[name] = paired_summary(
                value[src], value[dst], lower_is_better=True)

        term = env.action_manager.get_term("joint_pos")
        before_clip = term.raw_action * term.scale + term.offset
        clipped = (before_clip - term._processed_actions).abs() > 1.0e-7
        force = task.bridge.entity.data.actuator_force
        policy_limits = force.new_tensor(contract.action_scale) * force.new_tensor(
            contract.joint_stiffness) * 4.0
        effort_limits = task.bridge.robot.to_simulator_order(
            policy_limits, task.bridge.joint_names).unsqueeze(0)
        torque_ratio = force.abs() / effort_limits

        report = {
            "schema": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "pairs": args.pairs,
            "seed": args.seed,
            "motion_sha256": [file_sha256(path) for path in args.motion],
            "task_contract": asdict(contract),
            "pre_action_pair_max_difference": {
                "actor_observation": actor_diff,
                "critic_observation": critic_diff,
                "base_proposal": 0.0,
                "residual": 0.0,
            },
            "residual": {
                "rms": float(residual.square().mean().sqrt()),
                "mean_abs": float(residual.abs().mean()),
                "max_abs": float(residual.abs().max()),
            },
            "metrics": metrics,
            "controller": {
                "base_target_clip_fraction": float(clipped[src].float().mean()),
                "corrected_target_clip_fraction": float(clipped[dst].float().mean()),
                "base_torque_saturation_fraction_99pct": float(
                    (torque_ratio[src] >= .99).float().mean()),
                "corrected_torque_saturation_fraction_99pct": float(
                    (torque_ratio[dst] >= .99).float().mean()),
                "base_torque_to_limit_max": float(torque_ratio[src].max()),
                "corrected_torque_to_limit_max": float(torque_ratio[dst].max()),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
