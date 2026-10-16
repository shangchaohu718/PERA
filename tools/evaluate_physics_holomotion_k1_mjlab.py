"""Frozen evaluation of every complete K1 Actor bundle, with or without Physics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from unified_motion.k1_policy_runtime import (
    frozen_k1_policy, evaluate_k1_policy, load_k1_checkpoint, runtime_configuration,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--onnx", type=Path, help="metadata only; needed for a bundle without task_contract")
    source.add_argument("--task-contract", type=Path, help="JSON joint/PD contract, alternative to ONNX")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", choices=("full", "no_physics", "no_latent", "shuffle_latent", "no_history"), default="full")
    parser.add_argument("--video", type=Path,
                        help="optional H.264 MP4 from the exact evaluated rollout")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--no-reference-ghost", action="store_true",
                        help="omit the translucent green reference robot")
    parser.add_argument("--stop-on-first-done", action="store_true",
                        help="end evaluation at the first failure, timeout, or reference completion")
    args = parser.parse_args()
    if (args.steps < 1 or args.num_envs < 1 or args.seed < 0
            or args.video_width < 1 or args.video_height < 1):
        parser.error("steps/num-envs must be positive and seed non-negative")
    if args.video is not None and args.num_envs != 1:
        parser.error("video evaluation requires --num-envs 1")
    return args


def install_reference_ghost(env: Any, task: Any) -> None:
    """Draw the current reference pose as a translucent green robot."""
    original = getattr(env, "update_visualizers", None)
    robot = task.bridge.robot
    bridge = task.bridge

    def update_visualizers(visualizer) -> None:
        if original is not None:
            original(visualizer)
        reference = task.sampler.reference_window()
        root = reference.root_pos_w[0, 0]
        quat = reference.root_quat_wxyz[0, 0]
        joint = robot.to_simulator_order(
            reference.joint_pos[0, 0], bridge.joint_names)
        qpos = torch.cat((root, quat, joint)).detach().cpu().numpy()
        model = visualizer.mj_model
        rgba = model.geom_rgba.copy()
        try:
            model.geom_rgba[:, :3] = np.array((0.15, 1.0, 0.25))
            model.geom_rgba[:, 3] = 0.32
            visualizer.add_ghost_mesh(qpos, model, label="REFERENCE")
        finally:
            model.geom_rgba[:] = rgba

    env.update_visualizers = update_visualizers


def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real mjlab evaluation requires CUDA; no diagnostic fallback")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
    from unified_motion.envs.mjlab_k1 import K1MJLabBridge
    from unified_motion.holomotion_mjlab import (
        HoloMotionK1MJLabAdapter, HoloMotionOnnxContract, make_holomotion_k1_mjlab_cfg,
    )
    from unified_motion.holomotion_mjlab_task import (
        HoloMotionK1MJLabTask, HoloMotionPhysicsRandomizationConfig,
    )
    from mjlab.envs import ManagerBasedRlEnv

    loaded = load_k1_checkpoint(args.checkpoint, device=args.device)
    config = runtime_configuration(loaded.payload)
    saved = config.get("task_contract")
    contract = HoloMotionOnnxContract(**saved) if saved is not None else None
    if args.onnx is not None or args.task_contract is not None:
        supplied = (HoloMotionOnnxContract.from_onnx(args.onnx) if args.onnx is not None else
                    HoloMotionOnnxContract.from_json(args.task_contract))
        if contract is not None and contract != supplied:
            raise ValueError("ONNX metadata disagrees with saved Physics task contract")
        contract = supplied
    if contract is None:
        raise ValueError("checkpoint has no task contract; provide the training ONNX metadata")
    library = MotionLibrary.from_named_mjlab_files(args.motion, robot_id="k1_22dof", device=args.device)
    sampler = MotionSampler(library, num_envs=args.num_envs, step_dt=.02,
        future_offsets_s=tuple(i*.02 for i in range(contract.future_frames+2)))
    cfg = make_holomotion_k1_mjlab_cfg(str(args.asset_root), contract, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.video is not None:
        cfg.viewer.width = args.video_width
        cfg.viewer.height = args.video_height
        cfg.viewer.distance = 3.0
        cfg.viewer.elevation = -15.0
        cfg.viewer.azimuth = 135.0
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"
    env = ManagerBasedRlEnv(
        cfg, device=args.device,
        render_mode="rgb_array" if args.video is not None else None,
    )
    writer = None
    try:
        task = HoloMotionK1MJLabTask(
            HoloMotionK1MJLabAdapter(K1MJLabBridge(env,
                reference_steps=contract.future_frames+2), contract), sampler,
            physics_randomization=(None if config.get("physics_randomization") is None else
                HoloMotionPhysicsRandomizationConfig(**config["physics_randomization"])))
        if args.video is not None:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            if args.video.exists():
                raise FileExistsError(f"refusing to overwrite video: {args.video}")
            if not args.no_reference_ghost:
                install_reference_ghost(env, task)
            writer = imageio.get_writer(
                str(args.video), fps=round(1.0 / env.step_dt),
                codec="libx264", quality=8, macro_block_size=1,
            )
            original_step = task.step

            def recorded_step(action):
                transition = original_step(action)
                frame = env.render()
                if frame is None:
                    raise RuntimeError("offscreen renderer returned no frame")
                writer.append_data(frame)
                return transition

            task.step = recorded_step
        policy = frozen_k1_policy(loaded, ablation=args.ablation)
        report = evaluate_k1_policy(
            policy, task, steps=args.steps, stop_on_first_done=args.stop_on_first_done
        )
        from unified_motion.llm.advisor import fingerprint
        from unified_motion.llm.validation import file_sha256
        protocol = dict(schema=1, steps=args.steps, num_envs=args.num_envs, seed=args.seed,
            stop_on_first_done=args.stop_on_first_done,
            task_contract=saved if saved is not None else asdict(contract),
            physics_randomization=config.get('physics_randomization'),
            motion_sha256=[file_sha256(path) for path in args.motion],
            asset_root=str(args.asset_root.resolve()), ablation=args.ablation)
        report.update(evaluation_protocol=protocol, evaluation_protocol_sha256=fingerprint(protocol),
            training_trial=loaded.payload.get('training_trial'),
            training_trial_sha256=(fingerprint(loaded.payload['training_trial'])
                                   if 'training_trial' in loaded.payload else None),
            trial_adjustment_sha256=(fingerprint(loaded.payload['training_adjustment'])
                                     if 'training_adjustment' in loaded.payload else None),
            training_source_checkpoint_sha256=loaded.payload.get('training_source_checkpoint_sha256'),
            training_protocol=loaded.payload.get('training_protocol'),
            training_seed=loaded.payload.get('training_seed'), training_cost=loaded.payload.get('training_cost'))
        report.update(checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                      backend="mjlab_cuda", robot_skill_verified=False,
                      actor_backend=config.get("network_config", loaded.payload.get("network_config", {})).get("backend", "holomotion"))
        if args.video is not None:
            report["video"] = str(args.video.resolve())
            report["video_fps"] = round(1.0 / env.step_dt)
            report["video_frames"] = report["steps"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
