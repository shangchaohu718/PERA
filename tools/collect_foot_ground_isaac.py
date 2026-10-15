#!/usr/bin/env python3
"""Collect one frozen-policy Teacher V1 trajectory in the source Isaac task.

Launch this through the Moonwalk project's ``scripts/run_isaac51_k1.sh`` so its
Isaac, task-registration, asset, and RSL overlay paths are available.  This
command performs inference only; it constructs no optimizer and never runs PPO.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--policy-bundle", type=Path,
                    help="safe tensor-only bundle; native checkpoint remains provenance only")
parser.add_argument("--checkpoint-id", required=True,
                    help="globally unique policy identity; do not use model_3 alone")
parser.add_argument("--env-config-class", required=True,
                    help="module:Class that originally trained the checkpoint")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trajectory-id", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--phase-labels", type=Path, required=True,
                    help="NPZ with role_code [frame,left/right], where 2 means active slide")
parser.add_argument("--static-friction", type=float, default=.6)
parser.add_argument("--dynamic-friction", type=float, default=.5)
parser.add_argument("--mass-scale", type=float, default=1.0)
parser.add_argument("--force-frame-zero", action="store_true")
parser.add_argument("--trusted-source", action="store_true")
parser.add_argument("--manual-seed-without-replicator", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import random  # noqa: E402
import torch  # noqa: E402

import k1_moonwalk.project_task  # noqa: F401,E402
from unified_motion.foot_ground_teacher.collector import (  # noqa: E402
    FootGroundTrajectoryRecorder,
)
from unified_motion.foot_ground_teacher.features import (  # noqa: E402
    FOOT_CURRENT_DIM,
    pack_foot_current,
    pack_privileged_current,
    pack_student_observation,
    simple_joint_velocity_delta,
)
from unified_motion.foot_ground_teacher.isaac_collection import (  # noqa: E402
    ISAAC_RAW_SENSOR_NAMES,
    IsaacFootContactAccumulator,
    configure_isaac_teacher_contact_sensors,
)
from unified_motion.rsl_k1_policy import (  # noqa: E402
    load_frozen_rsl_k1_bundle,
    load_rsl_k1_checkpoint,
    verify_frozen_rsl_k1_provenance,
)


def config_class(path: str):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def policy_observation(observation):
    if isinstance(observation, dict):
        observation = observation["policy"]
    if observation.shape != (1, 125):
        raise RuntimeError(f"source Isaac policy observation is {observation.shape}, expected [1,125]")
    return observation


def joint_ids_for(action_term, robot):
    device = robot.data.joint_pos.device
    if isinstance(action_term._joint_ids, slice):
        ids = torch.arange(robot.num_joints, device=device)[action_term._joint_ids]
    else:
        ids = torch.as_tensor(action_term._joint_ids, device=device, dtype=torch.long)
    if ids.numel() != 22:
        raise RuntimeError("source action term must control exactly 22 joints")
    return ids


def quat_inverse_apply_wxyz(quaternion, vector):
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    w = quaternion[..., :1]
    xyz = -quaternion[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    return vector + 2 * (w * uv + torch.cross(xyz, uv, dim=-1))


def quat_apply_wxyz(quaternion, vector):
    conjugate = quaternion.clone()
    conjugate[..., 1:] *= -1
    return quat_inverse_apply_wxyz(conjugate, vector)


def state_after_step(unwrapped, terminal, foot_ids, action_joint_ids):
    robot = unwrapped.scene["robot"]
    if terminal is None:
        return {
            "joint_pos": robot.data.joint_pos[:, action_joint_ids].clone(),
            "joint_vel": robot.data.joint_vel[:, action_joint_ids].clone(),
            "root_quat": robot.data.root_quat_w.clone(),
            "root_ang_vel_w": robot.data.root_ang_vel_w.clone(),
            "foot_pos": robot.data.body_pos_w[:, foot_ids].clone(),
            "foot_quat": robot.data.body_quat_w[:, foot_ids].clone(),
            "foot_lin_vel": robot.data.body_link_lin_vel_w[:, foot_ids].clone(),
            "foot_ang_vel": robot.data.body_ang_vel_w[:, foot_ids].clone(),
        }
    return {
        "joint_pos": terminal["joint_pos"][:, action_joint_ids].clone(),
        "joint_vel": terminal["joint_vel"][:, action_joint_ids].clone(),
        "root_quat": terminal["root_state_w"][:, 3:7].clone(),
        "root_ang_vel_w": terminal["root_state_w"][:, 10:13].clone(),
        "foot_pos": terminal["body_pos_w"][:, foot_ids].clone(),
        "foot_quat": terminal["body_quat_w"][:, foot_ids].clone(),
        "foot_lin_vel": terminal["body_lin_vel_w"][:, foot_ids].clone(),
        "foot_ang_vel": terminal["body_ang_vel_w"][:, foot_ids].clone(),
    }


def termination_reason(terminal, *, truncated: bool, max_steps: bool) -> str:
    if terminal is not None:
        active = [name for name, value in terminal["termination_reasons"].items()
                  if bool(value[0].item())]
        return "+".join(active) if active else "terminated_unspecified"
    if truncated:
        return "time_limit"
    if max_steps:
        return "collector_max_steps"
    return "terminated_unspecified"


if args.max_steps < 1 or args.max_steps > 500:
    raise ValueError("--max-steps must be in [1,500]")
if not 0 < args.dynamic_friction <= args.static_friction:
    raise ValueError("require 0 < dynamic friction <= static friction")
if args.mass_scale != 1.0:
    raise ValueError("this source task has no audited mass-scaling event; use --mass-scale 1.0")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

cfg = config_class(args.env_config_class)()
cfg.scene.num_envs = 1
cfg.episode_length_s = args.max_steps / 50.0
cfg.commands.motion.play = bool(args.force_frame_zero)
cfg.commands.motion.start_frame_reset_ratio = 1.0 if args.force_frame_zero else 0.0
cfg.events.add_joint_default_pos = None
cfg.events.base_com = None
cfg.events.push_robot = None
cfg.events.physics_material.params.update({
    "static_friction_range": (args.static_friction, args.static_friction),
    "dynamic_friction_range": (args.dynamic_friction, args.dynamic_friction),
    "restitution_range": (0.0, 0.0),
    "num_buckets": 1,
})
cfg.observations.policy.enable_corruption = False
if args.manual_seed_without_replicator:
    cfg.seed = None
else:
    cfg.seed = args.seed
configure_isaac_teacher_contact_sensors(cfg.scene, max_contact_points=8)
print("[collector] task and contact-sensor configuration ready", flush=True)

if args.policy_bundle is not None:
    policy = load_frozen_rsl_k1_bundle(args.policy_bundle)
    verify_frozen_rsl_k1_provenance(
        policy, checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id)
elif args.trusted_source:
    policy = load_rsl_k1_checkpoint(
        args.checkpoint, trusted_source=True, checkpoint_id=args.checkpoint_id)
else:
    raise ValueError(
        "rollout requires --policy-bundle; direct native RSL loading is available only "
        "with the explicit --trusted-source escape hatch"
    )
print("[collector] frozen policy provenance verified", flush=True)

print("[collector] creating Isaac environment", flush=True)
try:
    environment = gym.make("Booster-K1-Moonwalk-v0", cfg=cfg)
except BaseException:  # Isaac can raise SystemExit during native-extension startup.
    print("[collector] environment creation failed", file=sys.stderr, flush=True)
    traceback.print_exc()
    raise
print("[collector] Isaac environment created", flush=True)
unwrapped = environment.unwrapped
policy = policy.to(unwrapped.device).eval().requires_grad_(False)
print("[collector] policy moved to environment device", flush=True)
robot = unwrapped.scene["robot"]
command = unwrapped.command_manager.get_term("motion")
action_term = unwrapped.action_manager.get_term("joint_pos")
action_joint_ids = joint_ids_for(action_term, robot)
print("[collector] robot, motion command, and action term resolved", flush=True)
foot_ids = [robot.body_names.index(name) for name in (
    "left_ankle_roll_link", "right_ankle_roll_link")]
sensors = tuple(unwrapped.scene[name] for name in ISAAC_RAW_SENSOR_NAMES)
print("[collector] foot bodies and raw contact sensors resolved", flush=True)
contact_accumulator = IsaacFootContactAccumulator(sensors, batch=1, slots=8)
contact_accumulator.subscribe()
print("[collector] physics-substep contact accumulator subscribed", flush=True)
observation, _ = environment.reset()
print("[collector] environment reset complete", flush=True)
observation = policy_observation(observation)
role_code = np.load(args.phase_labels, allow_pickle=False)["role_code"]
print("[collector] phase labels loaded", flush=True)
if role_code.ndim != 2 or role_code.shape[1] != 2:
    raise ValueError("phase role_code must have shape [frames,2]")
start_frame = int(command.time_steps[0].item())
reference_path = str(Path(cfg.commands.motion.motion_file).resolve())
recorder = FootGroundTrajectoryRecorder(
    checkpoint_id=args.checkpoint_id,
    checkpoint_sha256=policy.metadata.source_sha256,
    seed=args.seed, reference_id=reference_path,
    trajectory_id=args.trajectory_id,
    start_phase=start_frame / max(int(command.motion.time_step_total) - 1, 1),
    friction=args.dynamic_friction, mass_scale=args.mass_scale,
    disturbance={"type": "none", "configured_push_event": False},
)
print("[collector] trajectory recorder initialized", flush=True)
current_foot = torch.zeros((1, 2, FOOT_CURRENT_DIM), device=unwrapped.device)
episode_start = torch.ones(1, dtype=torch.bool, device=unwrapped.device)
zero_disturbance = torch.zeros((1, 3), device=unwrapped.device)
final_reason = "collector_max_steps"

try:
    for step_index in range(args.max_steps):
        current_joint_pos = robot.data.joint_pos[:, action_joint_ids].clone()
        current_joint_vel = robot.data.joint_vel[:, action_joint_ids].clone()
        current_root_ang = robot.data.root_ang_vel_b.clone()
        previous_action = unwrapped.action_manager.prev_action.clone()
        student = pack_student_observation(
            joint_position=current_joint_pos, joint_velocity=current_joint_vel,
            imu_angular_velocity_body=current_root_ang,
            imu_projected_gravity_body=robot.data.projected_gravity_b.clone(),
            estimated_joint_torque=robot.data.applied_torque[:, action_joint_ids].clone(),
            previous_executed_action=previous_action,
        )
        privileged = pack_privileged_current(
            root_linear_velocity_body=robot.data.root_lin_vel_b.clone(),
            root_height=robot.data.root_pos_w[:, 2:3].clone(),
            friction=torch.full((1, 1), args.dynamic_friction, device=unwrapped.device),
            mass_scale=torch.full((1, 1), args.mass_scale, device=unwrapped.device),
            applied_external_force_body=zero_disturbance,
            applied_external_torque_body=zero_disturbance,
        )
        reference_frame_t = int(command.time_steps[0].item())
        with torch.inference_mode():
            action = policy.act(observation)
        contact_accumulator.begin_control_step()
        next_observation, _, terminated, truncated, info = environment.step(action)
        done = bool((terminated | truncated)[0].item())
        terminal = info.get("terminal_state") if done else None
        next_state = state_after_step(unwrapped, terminal, foot_ids, action_joint_ids)
        root_x = next_state["root_quat"].new_tensor((1.0, 0.0, 0.0)).expand(1, 3)
        heading = quat_apply_wxyz(next_state["root_quat"], root_x)
        heading = heading[:, None].expand(-1, 2, -1)
        next_reference_frame = (
            int(terminal["reference_frame"][0].item()) if terminal is not None
            else int(command.time_steps[0].item())
        )
        role = torch.as_tensor(
            role_code[next_reference_frame % len(role_code)] == 2,
            device=unwrapped.device, dtype=torch.bool).unsqueeze(0)
        next_contact = contact_accumulator.finish_control_step(
            body_position_w=next_state["foot_pos"],
            body_quaternion_wxyz=next_state["foot_quat"],
            body_linear_velocity_w=next_state["foot_lin_vel"],
            body_angular_velocity_w=next_state["foot_ang_vel"],
            heading_w=heading,
            friction=torch.full((1,), args.dynamic_friction, device=unwrapped.device),
            active_sliding=role,
        )
        next_root_ang = quat_inverse_apply_wxyz(
            next_state["root_quat"], next_state["root_ang_vel_w"])
        try:
            simple_delta = simple_joint_velocity_delta(
                robot, control_dt_s=.02, joint_ids=action_joint_ids)
        except RuntimeError:
            # Preserve the transition and make the fallback explicit in telemetry.
            simple_delta = torch.zeros_like(current_joint_vel)
            baseline_valid = torch.zeros(1, dtype=torch.bool, device=unwrapped.device)
        else:
            baseline_valid = torch.ones(1, dtype=torch.bool, device=unwrapped.device)
        if done:
            # Auto-reset may replace mass/torque buffers after the terminal
            # physics step; do not present that baseline as terminal evidence.
            simple_delta.zero_()
            baseline_valid.zero_()
        unexpected = next_contact.contact & (next_contact.slip_speed >= .03) & ~role
        recorder.append(
            student_observation_t=student, privileged_current_t=privileged,
            executed_action_t=action, foot_current_t=current_foot,
            episode_start_t=episode_start, next_contact=next_contact,
            joint_velocity_t=current_joint_vel,
            joint_velocity_next=next_state["joint_vel"],
            root_angular_velocity_t=current_root_ang,
            root_angular_velocity_next=next_root_ang,
            simple_physics_delta_joint_velocity=simple_delta,
            active_sliding_next=role, unexpected_slip_next=unexpected,
            friction_t=torch.full((1,), args.dynamic_friction, device=unwrapped.device),
            valid_transition=torch.ones(1, dtype=torch.bool, device=unwrapped.device),
            joint_velocity_residual_valid=baseline_valid,
            extra_tensors={
                "policy_observation": observation,
                "reference_frame": torch.tensor([[reference_frame_t]], device=unwrapped.device),
                "next_reference_frame": torch.tensor([[next_reference_frame]], device=unwrapped.device),
                "terminated": terminated.bool(), "truncated": truncated.bool(),
                "joint_position": current_joint_pos,
                "next_joint_position": next_state["joint_pos"],
            },
        )
        current_foot = pack_foot_current(next_contact)
        episode_start.zero_()
        if done:
            final_reason = termination_reason(
                terminal, truncated=bool(truncated[0].item()), max_steps=False)
            break
        observation = policy_observation(next_observation)
    else:
        final_reason = termination_reason(None, truncated=False, max_steps=True)
    path = recorder.finish(args.output, termination_reason=final_reason)
    print(path, flush=True)
finally:
    contact_accumulator.close()
    environment.close()
    app.close()
