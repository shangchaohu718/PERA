"""Run an exported official HoloMotion K1 actor in the real mjlab backend.

This is a simulation evaluator, not a trainer and not a robot deployment tool.
It requires a named HoloMotion/MJLab reference NPZ as accepted by
``MotionLibrary.from_named_mjlab_file``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.holomotion_mjlab import (
    HoloMotionK1MJLabAdapter,
    HoloMotionOnnxContract,
    make_holomotion_k1_mjlab_cfg,
)
from unified_motion.holomotion_mjlab_rewards import (
    HoloMotionMJLabState,
    gravity_projection_distance,
    holomotion_k1_reward_terms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--provider",
        choices=("CUDAExecutionProvider", "CPUExecutionProvider"),
        default="CUDAExecutionProvider",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not args.device.startswith("cuda"):
        parser.error("mjlab evaluation requires a CUDA device")
    return args


def _shape(session, name: str) -> tuple[int, ...]:
    item = next((value for value in session.get_inputs() if value.name == name), None)
    if item is None or any(not isinstance(dim, int) or dim < 1 for dim in item.shape):
        raise ValueError(f"ONNX input {name!r} must have a static positive shape")
    return tuple(item.shape)


def _initialize_physics_from_reference(
    bridge: K1MJLabBridge, sampler: MotionSampler
) -> None:
    reference = sampler.reference_window()
    ids = torch.arange(bridge.num_envs, device=bridge.device, dtype=torch.long)
    root_pose = torch.cat(
        (reference.root_pos_w[:, 0], reference.root_quat_wxyz[:, 0]), -1
    )
    root_velocity = torch.cat(
        (reference.root_lin_vel_w[:, 0], reference.root_ang_vel_w[:, 0]), -1
    )
    joint_pos = bridge.robot.to_simulator_order(
        reference.joint_pos[:, 0], bridge.joint_names
    )
    joint_vel = bridge.robot.to_simulator_order(
        reference.joint_vel[:, 0], bridge.joint_names
    )
    bridge.entity.write_root_link_pose_to_sim(root_pose, env_ids=ids)
    bridge.entity.write_root_link_velocity_to_sim(root_velocity, env_ids=ids)
    bridge.entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    bridge.env.sim.forward()
    bridge.env.sim.sense()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no mjlab simulation was started")
    if args.provider == "CUDAExecutionProvider":
        device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        providers = [("CUDAExecutionProvider", {"device_id": device_index})]
    else:
        providers = ["CPUExecutionProvider"]

    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv

    contract = HoloMotionOnnxContract.from_onnx(args.onnx)
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    if session.get_providers()[0] != args.provider:
        raise RuntimeError(
            f"ONNX Runtime fell back to {session.get_providers()}; expected {args.provider}"
        )
    if _shape(session, "obs") != (1, contract.obs_dim):
        raise ValueError("this evaluator currently requires the exported batch-1 actor")
    kv = np.zeros(_shape(session, "past_key_values"), dtype=np.float32)
    step_idx = np.zeros(_shape(session, "step_idx"), dtype=np.int64)

    device = torch.device(args.device)
    library = MotionLibrary.from_named_mjlab_file(
        args.motion, robot_id="k1_22dof", device=device
    )
    offsets = tuple(index * 0.02 for index in range(contract.future_frames + 2))
    sampler = MotionSampler(library, num_envs=1, step_dt=0.02, future_offsets_s=offsets)
    cfg = make_holomotion_k1_mjlab_cfg(args.asset_root, contract, num_envs=1)
    env = ManagerBasedRlEnv(cfg, device=str(device))
    bridge = K1MJLabBridge(env, reference_steps=contract.future_frames + 2)
    adapter = HoloMotionK1MJLabAdapter(bridge, contract)
    records = []
    initial_alignment = None
    try:
        adapter.reset()
        data = bridge.entity.data
        sampler.reset(
            clip_ids=torch.zeros(1, dtype=torch.long, device=device),
            start_times_s=torch.zeros(1, device=device),
            current_root_pos_w=data.root_link_pos_w,
            current_root_quat_wxyz=data.root_link_quat_w,
        )
        _initialize_physics_from_reference(bridge, sampler)
        initial_reference = sampler.reference_window()
        initial_body_reference = sampler.body_reference_window()
        zero_action = adapter.last_action_onnx.clone()
        initial_state = HoloMotionMJLabState.from_adapter(
            adapter, action=zero_action, previous_action=zero_action
        )
        initial_terms = holomotion_k1_reward_terms(
            initial_state,
            initial_reference,
            initial_body_reference,
            reference_index=0,
        )
        initial_alignment = {
            name: float(value[0].item()) for name, value in initial_terms.items()
        }
        max_steps = min(args.steps, int(library.clip_lengths[0].item()) - 1)
        for index in range(max_steps):
            reference = sampler.reference_window()
            body_reference = sampler.body_reference_window()
            observation = adapter.observation(reference)
            outputs = session.run(
                None,
                {
                    "obs": observation.detach().cpu().numpy(),
                    "past_key_values": kv,
                    "step_idx": step_idx,
                },
            )
            action = torch.as_tensor(outputs[0], device=device, dtype=observation.dtype)
            kv = np.asarray(outputs[1], dtype=np.float32)
            previous_action = adapter.last_action_onnx.clone()
            transition = adapter.step(action)
            state = HoloMotionMJLabState.from_adapter(
                adapter, action=action, previous_action=previous_action
            )
            reward_terms = holomotion_k1_reward_terms(
                state, reference, body_reference, reference_index=1
            )
            reference_joint_sim = bridge.robot.to_simulator_order(
                reference.joint_pos[:, 1], bridge.joint_names
            )
            joint_rmse = torch.sqrt(
                torch.square(state.joint_pos - reference_joint_sim).mean(-1)
            )
            action_term = bridge.action_term
            unclipped_target = (
                action_term.raw_action * action_term.scale + action_term.offset
            )
            clipped_targets = (
                (unclipped_target - action_term._processed_actions).abs() > 1.0e-6
            ).sum(-1)
            torque = bridge.entity.data.qfrc_actuator
            effort_onnx = (
                contract.tensor("action_scale", action)
                * contract.tensor("joint_stiffness", action)
                * 4.0
            )
            effort_policy = adapter.builder.onnx_action_to_policy(effort_onnx[None])[0]
            effort_sim = bridge.robot.to_simulator_order(
                effort_policy[None], bridge.joint_names
            )
            saturated_torques = (torque.abs() >= 0.99 * effort_sim).sum(-1)
            gravity_distance = gravity_projection_distance(
                state.root_quat_wxyz,
                reference.root_quat_wxyz[:, 1],
            )
            sampler.advance()
            official_terminated = gravity_distance > 0.8
            done = bool(
                (official_terminated | transition.terminated | transition.truncated)[
                    0
                ].item()
            )
            records.append(
                {
                    "step": index,
                    "root_height_m": float(
                        bridge.entity.data.root_link_pos_w[0, 2].item()
                    ),
                    "action_l2": float(torch.linalg.vector_norm(action[0]).item()),
                    "joint_rmse_rad": float(joint_rmse[0].item()),
                    "gravity_projection_distance": float(gravity_distance[0].item()),
                    "max_abs_actuator_torque_nm": float(torque[0].abs().max().item()),
                    "clipped_position_targets": int(clipped_targets[0].item()),
                    "saturated_actuator_torques": int(saturated_torques[0].item()),
                    "reward_terms": {
                        name: float(value[0].item())
                        for name, value in reward_terms.items()
                    },
                    "official_gravity_termination": bool(official_terminated[0].item()),
                    "done": done,
                }
            )
            step_idx += 1
            if done or bool(sampler.finished()[0].item()):
                break
    finally:
        adapter.bridge.close()

    summary = {
        "backend": "mjlab",
        "policy": str(args.onnx.resolve()),
        "motion": str(args.motion.resolve()),
        "provider": session.get_providers()[0],
        "initial_alignment": initial_alignment,
        "completed_steps": len(records),
        "terminated": bool(records and records[-1]["done"]),
        "minimum_root_height_m": min(
            (item["root_height_m"] for item in records), default=None
        ),
        "records": records,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
