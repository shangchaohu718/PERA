#!/usr/bin/env python3
"""Collect one exact same-state base/plus/minus action branch in Isaac Lab."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--policy-bundle", type=Path, required=True)
parser.add_argument("--checkpoint-id", required=True)
parser.add_argument("--env-config-class", required=True)
parser.add_argument("--phase-labels", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--probe-frame", type=int, required=True)
parser.add_argument("--joint-name", required=True)
parser.add_argument("--delta-rad", type=float, default=.02)
parser.add_argument("--context-steps", type=int, default=32)
parser.add_argument("--static-friction", type=float, default=.6)
parser.add_argument("--dynamic-friction", type=float, default=.5)
parser.add_argument("--branch-state-mode", choices=("clone", "deterministic_replay"),
                    default="clone", help="clone exposed state at the probe, or preserve "
                    "per-environment PhysX contact caches after identical reset-to-probe replay")
parser.add_argument("--manual-seed-without-replicator", action="store_true",
                    help="seed Python/NumPy/Torch directly and leave cfg.seed unset; "
                         "avoids the Isaac 5.1 headless Replicator seed import")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import k1_moonwalk.project_task  # noqa: F401,E402
from unified_motion.foot_ground_teacher.features import (  # noqa: E402
    FOOT_CURRENT_DIM, pack_foot_current, pack_privileged_current,
    pack_student_observation,
)
from unified_motion.foot_ground_teacher.isaac_collection import (  # noqa: E402
    ISAAC_RAW_SENSOR_NAMES, IsaacFootContactAccumulator,
    configure_isaac_teacher_contact_sensors,
)
from unified_motion.foot_ground_teacher.reference import (  # noqa: E402
    REFERENCE_INTENT_NAMES, reference_intent_from_frames,
)
from unified_motion.rsl_k1_policy import (  # noqa: E402
    load_frozen_rsl_k1_bundle, verify_frozen_rsl_k1_provenance,
)


def config_class(path: str):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def policy_observation(observation):
    value = observation["policy"] if isinstance(observation, dict) else observation
    if value.shape != (4, 125):
        raise RuntimeError(f"source policy observation is {value.shape}, expected [4,125]")
    return value


def joint_ids_for(action_term, robot):
    if isinstance(action_term._joint_ids, slice):
        return torch.arange(robot.num_joints, device=robot.data.joint_pos.device)[
            action_term._joint_ids]
    return torch.as_tensor(action_term._joint_ids, device=robot.data.joint_pos.device,
                           dtype=torch.long)


def quat_inverse_apply_wxyz(quaternion, vector):
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    w, xyz = quaternion[..., :1], -quaternion[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    return vector + 2 * (w * uv + torch.cross(xyz, uv, dim=-1))


def quat_apply_wxyz(quaternion, vector):
    conjugate = quaternion.clone()
    conjugate[..., 1:] *= -1
    return quat_inverse_apply_wxyz(conjugate, vector)


def clone_env0(unwrapped, action_term, command, robot, targets=(1, 2, 3)):
    """Clone exposed physical, controller, actuator-delay, and command state."""
    target_ids = torch.as_tensor(targets, device=unwrapped.device, dtype=torch.long)
    root = robot.data.root_state_w[0:1].repeat(len(targets), 1)
    root[:, :3] += (unwrapped.scene.env_origins[target_ids]
                    - unwrapped.scene.env_origins[0])
    robot.write_root_state_to_sim(root, env_ids=target_ids)
    robot.write_joint_state_to_sim(
        robot.data.joint_pos[0:1].repeat(len(targets), 1),
        robot.data.joint_vel[0:1].repeat(len(targets), 1), env_ids=target_ids)
    for target in targets:
        unwrapped.action_manager._action[target].copy_(unwrapped.action_manager._action[0])
        unwrapped.action_manager._prev_action[target].copy_(
            unwrapped.action_manager._prev_action[0])
        action_term._raw_actions[target].copy_(action_term._raw_actions[0])
        action_term._processed_actions[target].copy_(action_term._processed_actions[0])
    for actuator in robot.actuators.values():
        for buffer_name in ("positions_delay_buffer", "velocities_delay_buffer",
                            "efforts_delay_buffer"):
            delay = getattr(actuator, buffer_name)
            circular = delay._circular_buffer
            for target in targets:
                circular._buffer[:, target].copy_(circular._buffer[:, 0])
                circular._num_pushes[target].copy_(circular._num_pushes[0])
                delay._time_lags[target].copy_(delay._time_lags[0])
        for target in targets:
            if hasattr(actuator, "_joint_vel"):
                actuator._joint_vel[target].copy_(actuator._joint_vel[0])
            actuator.computed_effort[target].copy_(actuator.computed_effort[0])
            actuator.applied_effort[target].copy_(actuator.applied_effort[0])
    for target in targets:
        command.time_steps[target].copy_(command.time_steps[0])
        command.motion_ended[target].copy_(command.motion_ended[0])
        unwrapped.episode_length_buf[target].copy_(unwrapped.episode_length_buf[0])


def pair_state_error(unwrapped, robot):
    root_local = robot.data.root_pos_w - unwrapped.scene.env_origins
    values = []
    for target in (1, 2, 3):
        values.extend((
            (root_local[0] - root_local[target]).abs().max(),
            (robot.data.root_quat_w[0] - robot.data.root_quat_w[target]).abs().max(),
            (robot.data.joint_pos[0] - robot.data.joint_pos[target]).abs().max(),
            (robot.data.joint_vel[0] - robot.data.joint_vel[target]).abs().max(),
        ))
    return float(torch.stack(values).max())


if not 0 < args.delta_rad <= .1 or not 1 <= args.context_steps <= 128:
    raise ValueError("delta-rad must be in (0,.1] and context-steps in [1,128]")
if not 1 <= args.probe_frame <= 498:
    raise ValueError("probe-frame must be in [1,498]")
if not 0 < args.dynamic_friction <= args.static_friction:
    raise ValueError("require 0 < dynamic friction <= static friction")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
cfg = config_class(args.env_config_class)()
cfg.scene.num_envs = 4
cfg.episode_length_s = (args.probe_frame + 2) / 50.0
cfg.commands.motion.play = True
cfg.commands.motion.start_frame_reset_ratio = 1.0
cfg.events.add_joint_default_pos = None
cfg.events.base_com = None
cfg.events.push_robot = None
cfg.events.physics_material.params.update({
    "static_friction_range": (args.static_friction, args.static_friction),
    "dynamic_friction_range": (args.dynamic_friction, args.dynamic_friction),
    "restitution_range": (0.0, 0.0), "num_buckets": 1,
})
cfg.observations.policy.enable_corruption = False
cfg.seed = None if args.manual_seed_without_replicator else args.seed
configure_isaac_teacher_contact_sensors(cfg.scene, max_contact_points=8)

policy = load_frozen_rsl_k1_bundle(args.policy_bundle)
verify_frozen_rsl_k1_provenance(
    policy, checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id)
environment = gym.make("Booster-K1-Moonwalk-v0", cfg=cfg)
unwrapped = environment.unwrapped
policy = policy.to(unwrapped.device).eval().requires_grad_(False)
robot = unwrapped.scene["robot"]
command = unwrapped.command_manager.get_term("motion")
action_term = unwrapped.action_manager.get_term("joint_pos")
action_joint_ids = joint_ids_for(action_term, robot)
if action_joint_ids.numel() != 22 or args.joint_name not in robot.joint_names:
    raise ValueError("counterfactual joint is not in the 22-DoF action contract")
robot_joint_id = robot.joint_names.index(args.joint_name)
matches = (action_joint_ids == robot_joint_id).nonzero().flatten()
if matches.numel() != 1:
    raise ValueError(f"{args.joint_name} is not controlled exactly once")
action_index = int(matches.item())
feet_actuator = robot.actuators["feet"]
feet_robot_joint_ids = [index for index, name in enumerate(robot.joint_names)
                        if "ankle" in name.lower()]
if robot_joint_id not in feet_robot_joint_ids:
    raise ValueError("counterfactual joint is not controlled by the feet actuator")
feet_local_joint_id = feet_robot_joint_ids.index(robot_joint_id)
foot_ids = [robot.body_names.index(name) for name in (
    "left_ankle_roll_link", "right_ankle_roll_link")]
sensors = tuple(unwrapped.scene[name] for name in ISAAC_RAW_SENSOR_NAMES)
accumulator = IsaacFootContactAccumulator(sensors, batch=4, slots=8)
accumulator.subscribe()
with np.load(args.phase_labels, allow_pickle=False) as phase_archive:
    role_code = phase_archive["role_code"]
    cycle_start = int(phase_archive["cycle_start_50hz"])
    cycle_frames = int(phase_archive["cycle_frames"])

student_history, privileged_history, foot_history = [], [], []
action_history, episode_history, reference_history = [], [], []
current_foot = torch.zeros((4, 2, FOOT_CURRENT_DIM), device=unwrapped.device)
observation, _ = environment.reset()
observation = policy_observation(observation)

try:
    while int(command.time_steps[0]) <= args.probe_frame:
        frame = int(command.time_steps[0])
        if frame == args.probe_frame:
            preclone_error = pair_state_error(unwrapped, robot)
            if args.branch_state_mode == "clone":
                with torch.inference_mode():
                    clone_env0(unwrapped, action_term, command, robot)
                    if accumulator._previous_active is not None:
                        accumulator._previous_active[1:].copy_(
                            accumulator._previous_active[0:1].expand(3, -1))
            clone_error = pair_state_error(unwrapped, robot)
        q = robot.data.joint_pos[:, action_joint_ids].clone()
        dq = robot.data.joint_vel[:, action_joint_ids].clone()
        student = pack_student_observation(
            joint_position=q, joint_velocity=dq,
            imu_angular_velocity_body=robot.data.root_ang_vel_b.clone(),
            imu_projected_gravity_body=robot.data.projected_gravity_b.clone(),
            estimated_joint_torque=robot.data.applied_torque[:, action_joint_ids].clone(),
            previous_executed_action=unwrapped.action_manager.prev_action.clone())
        zero = torch.zeros((4, 3), device=unwrapped.device)
        privileged = pack_privileged_current(
            root_linear_velocity_body=robot.data.root_lin_vel_b.clone(),
            root_height=robot.data.root_pos_w[:, 2:3].clone(),
            friction=torch.full((4, 1), args.dynamic_friction, device=unwrapped.device),
            mass_scale=torch.ones((4, 1), device=unwrapped.device),
            applied_external_force_body=zero, applied_external_torque_body=zero)
        with torch.inference_mode():
            base_action = policy.act(observation[0:1])
        intent_np = reference_intent_from_frames(
            np.asarray([[frame]], dtype=np.int64), role_code,
            cycle_start=cycle_start, cycle_frames=cycle_frames)
        student_history.append(student[0].cpu())
        privileged_history.append(privileged[0].cpu())
        foot_history.append(current_foot[0].cpu())
        action_history.append(base_action[0].cpu())
        episode_history.append(frame == 0)
        reference_history.append(torch.from_numpy(intent_np[0]))
        actions = base_action.repeat(4, 1)
        if frame == args.probe_frame:
            scale = action_term._scale[0, action_index]
            if not torch.isfinite(scale) or scale.abs() < 1e-8:
                raise RuntimeError("selected action has zero/non-finite joint-position scale")
            raw_delta = args.delta_rad / float(scale)
            actions[1, action_index] += raw_delta
            actions[2, action_index] -= raw_delta
        accumulator.begin_control_step()
        next_observation, _, terminated, truncated, _ = environment.step(actions)
        done = terminated | truncated
        if done.any():
            raise RuntimeError(f"counterfactual branch terminated at frame {frame}: {done.tolist()}")
        root_x = robot.data.root_quat_w.new_tensor((1.0, 0.0, 0.0)).expand(4, 3)
        heading = quat_apply_wxyz(robot.data.root_quat_w, root_x)[:, None].expand(-1, 2, -1)
        next_frames = command.time_steps.detach().cpu().numpy().astype(np.int64)
        role = torch.as_tensor(
            role_code[np.remainder(next_frames, len(role_code))] == 2,
            device=unwrapped.device, dtype=torch.bool)
        contact = accumulator.finish_control_step(
            body_position_w=robot.data.body_pos_w[:, foot_ids],
            body_quaternion_wxyz=robot.data.body_quat_w[:, foot_ids],
            body_linear_velocity_w=robot.data.body_link_lin_vel_w[:, foot_ids],
            body_angular_velocity_w=robot.data.body_ang_vel_w[:, foot_ids],
            heading_w=heading,
            friction=torch.full((4,), args.dynamic_friction, device=unwrapped.device),
            active_sliding=role)
        if frame == args.probe_frame:
            processed = action_term._processed_actions[:, action_index].detach().cpu()
            delay = feet_actuator.positions_delay_buffer
            # This is the exact stale target selected by Isaac Lab's official
            # DelayBuffer after the last 200 Hz physics substep.  A processed
            # action target is not evidence that the actuator has seen it.
            delayed_targets = delay._circular_buffer._buffer[
                delay.time_lags, torch.arange(4, device=unwrapped.device)]
            delayed_target = delayed_targets[:, feet_local_joint_id].detach().cpu()
            applied_torque = feet_actuator.applied_effort[
                :, feet_local_joint_id].detach().cpu()
            time_lag_steps = delay.time_lags.detach().cpu()
            break
        current_foot = pack_foot_current(contact)
        observation = policy_observation(next_observation)
    else:
        raise RuntimeError("probe frame was not reached")

    start = max(0, len(student_history) - args.context_steps)
    base_actions = torch.stack(action_history[start:])
    branch_actions = base_actions.unsqueeze(0).repeat(3, 1, 1)
    branch_actions[:, -1] = actions[:3].detach().cpu()
    arrays = {
        "student_observation": torch.stack(student_history[start:]).unsqueeze(0).numpy(),
        "privileged_current": torch.stack(privileged_history[start:]).unsqueeze(0).numpy(),
        "foot_current": torch.stack(foot_history[start:]).unsqueeze(0).numpy(),
        "episode_start": torch.tensor(episode_history[start:], dtype=torch.bool).unsqueeze(0).numpy(),
        "reference_intent": torch.stack(reference_history[start:]).unsqueeze(0).numpy(),
        "executed_action": branch_actions.unsqueeze(0).numpy(),
        "target_contact": contact.contact[:3].detach().cpu().unsqueeze(0).numpy(),
        "target_contact_mode": contact.contact_mode[:3].detach().cpu().unsqueeze(0).numpy(),
        "target_tangent_velocity":
            contact.relative_velocity_tangent[:3].detach().cpu().unsqueeze(0).numpy(),
        "target_normal_force":
            contact.normal_force[:3].detach().cpu().unsqueeze(0).numpy(),
        "processed_target_delta_rad": np.asarray([[
            0.0, float(processed[1] - processed[0]),
            float(processed[2] - processed[0])]], dtype=np.float32),
        "delayed_actuator_target_delta_rad": np.asarray([[
            0.0, float(delayed_target[1] - delayed_target[0]),
            float(delayed_target[2] - delayed_target[0])]], dtype=np.float32),
        "applied_torque_delta_nm": np.asarray([[
            0.0, float(applied_torque[1] - applied_torque[0]),
            float(applied_torque[2] - applied_torque[0])]], dtype=np.float32),
        "actuator_time_lag_physics_steps": time_lag_steps.unsqueeze(0).numpy(),
        "null_target_contact": contact.contact[3:4].detach().cpu().numpy(),
        "null_target_tangent_velocity":
            contact.relative_velocity_tangent[3:4].detach().cpu().numpy(),
        "null_target_normal_force": contact.normal_force[3:4].detach().cpu().numpy(),
        "delta_rad": np.asarray([args.delta_rad], dtype=np.float32),
    }
    metadata = {
        "schema": 1, "branches": ["base", "plus", "minus"],
        "null_replicate_branch": "base_duplicate",
        "checkpoint_id": args.checkpoint_id, "seed": args.seed,
        "simulator_seed_via_cfg": not args.manual_seed_without_replicator,
        "probe_frame": args.probe_frame, "joint_name": args.joint_name,
        "action_index": action_index, "requested_delta_rad": args.delta_rad,
        "raw_action_delta": raw_delta, "action_scale": float(scale),
        "processed_target_delta_rad": [
            0.0, float(processed[1] - processed[0]), float(processed[2] - processed[0])],
        "delayed_actuator_target_delta_rad": [
            0.0, float(delayed_target[1] - delayed_target[0]),
            float(delayed_target[2] - delayed_target[0])],
        "applied_torque_delta_nm": [
            0.0, float(applied_torque[1] - applied_torque[0]),
            float(applied_torque[2] - applied_torque[0])],
        "actuator_time_lag_physics_steps": time_lag_steps.tolist(),
        "pre_branch_state_max_abs_error": clone_error,
        "preclone_state_max_abs_error": preclone_error,
        "branch_state_mode": args.branch_state_mode,
        "dynamic_friction": args.dynamic_friction,
        "context_steps": branch_actions.shape[1],
        "reference_intent_names": list(REFERENCE_INTENT_NAMES),
        "same_state_branching": True, "optimizer_updates": 0,
        "actor_updates": 0, "ppo_updates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(args.output)
    print(json.dumps(metadata, indent=2))
finally:
    accumulator.close()
    environment.close()
    app.close()
