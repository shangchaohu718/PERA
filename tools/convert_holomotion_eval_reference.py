"""把 HoloMotion 冻结评测 NPZ 中的 ref_* 参考轨迹导出为本项目带名字格式。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pose_gradient(values: np.ndarray, fps: float) -> np.ndarray:
    """Match HoloMotion ``torch_utils.grad_t`` for a complete clip.

    Offline evaluation dumps can contain a stale ``ref_dof_vel`` at the first
    pre-step sample after reset.  Training and policy observations derive DoF
    velocity from pose context instead, so the portable reference must do the
    same rather than treating the diagnostic dump as ground truth.
    """

    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if values.ndim < 2:
        raise ValueError(
            f"values must include frame and feature axes, got {values.shape}"
        )
    velocity = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return velocity
    velocity[0] = (values[1] - values[0]) * fps
    velocity[-1] = (values[-1] - values[-2]) * fps
    if values.shape[0] > 2:
        velocity[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return velocity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    contract = parser.add_mutually_exclusive_group(required=True)
    contract.add_argument(
        "--contract-source",
        type=Path,
        help="仅读取已审计 K1 NPZ 的 joint_names/body_names，不读取其动作",
    )
    contract.add_argument(
        "--robot-config",
        type=Path,
        help="读取 HoloMotion robot YAML 中的 robot.dof_names/body_names",
    )
    parser.add_argument(
        "--source-quaternion-order",
        choices=("xyzw", "wxyz"),
        required=True,
        help="官方离线评测 ref_global_rotation_quat 当前使用 xyzw；必须显式确认",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{args.output}")

    if args.contract_source is not None:
        with np.load(args.contract_source, allow_pickle=False) as contract_data:
            joint_names = np.array(contract_data["joint_names"], copy=True)
            body_names = np.array(contract_data["body_names"], copy=True)
    else:
        import yaml

        config = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))
        robot = config.get("robot", {}) if isinstance(config, dict) else {}
        joint_names = np.asarray(robot.get("dof_names", ()))
        body_names = np.asarray(robot.get("body_names", ()))
        if (
            joint_names.ndim != 1
            or body_names.ndim != 1
            or not len(joint_names)
            or not len(body_names)
        ):
            raise ValueError(
                "robot config must define robot.dof_names and robot.body_names"
            )
    with np.load(args.input, allow_pickle=False) as source:
        required = (
            "ref_dof_pos",
            "ref_dof_vel",
            "ref_global_translation",
            "ref_global_rotation_quat",
            "ref_global_velocity",
            "ref_global_angular_velocity",
            "metadata",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"HoloMotion评测文件缺少参考字段：{missing}")
        metadata = json.loads(np.asarray(source["metadata"]).item())
        fps = float(metadata["motion_fps"])
        joint_pos = np.array(source["ref_dof_pos"], dtype=np.float32, copy=True)
        dumped_joint_vel = np.array(source["ref_dof_vel"], dtype=np.float32, copy=True)
        body_pos = np.array(
            source["ref_global_translation"], dtype=np.float32, copy=True
        )
        body_quat = np.array(
            source["ref_global_rotation_quat"], dtype=np.float32, copy=True
        )
        if args.source_quaternion_order == "xyzw":
            body_quat = body_quat[..., (3, 0, 1, 2)]
        body_lin = np.array(source["ref_global_velocity"], dtype=np.float32, copy=True)
        body_ang = np.array(
            source["ref_global_angular_velocity"], dtype=np.float32, copy=True
        )

    if (
        dumped_joint_vel.shape != joint_pos.shape
        or not np.isfinite(dumped_joint_vel).all()
    ):
        raise ValueError(
            "ref_dof_vel must be a finite array matching ref_dof_pos; "
            f"got {dumped_joint_vel.shape} versus {joint_pos.shape}"
        )

    # This mirrors the current HoloMotion training/observation path.  In
    # particular it repairs the stale first ref_dof_vel sample present in some
    # offline-evaluation rollouts.
    joint_vel = _pose_gradient(joint_pos, fps)
    dumped_joint_vel_rmse = float(
        np.sqrt(np.mean(np.square(dumped_joint_vel - joint_vel)))
    )
    dumped_joint_vel_first_frame_rmse = float(
        np.sqrt(np.mean(np.square(dumped_joint_vel[0] - joint_vel[0])))
    )

    frames = joint_pos.shape[0]
    expected = {
        "joint_pos": (frames, len(joint_names)),
        "joint_vel": (frames, len(joint_names)),
        "body_pos_w": (frames, len(body_names), 3),
        "body_quat_w": (frames, len(body_names), 4),
        "body_lin_vel_w": (frames, len(body_names), 3),
        "body_ang_vel_w": (frames, len(body_names), 3),
    }
    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
    }
    for name, shape in expected.items():
        value = arrays[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} 应为有限数组 {shape}，实际 {value.shape}")
    quat_norm = np.linalg.norm(body_quat, axis=-1)
    if np.max(np.abs(quat_norm - 1.0)) > 1e-2:
        raise ValueError("参考身体四元数不是单位四元数")

    exported_metadata = json.dumps(
        {
            "schema": "holomotion_eval_reference_named_v2",
            "source": str(args.input.resolve()),
            "source_motion_key": metadata.get(
                "raw_motion_key", metadata.get("motion_key")
            ),
            "source_quaternion_order": args.source_quaternion_order,
            "quaternion_order": "wxyz_body_to_world",
            "content": "ref_fields_only_no_policy_rollout",
            "joint_velocity_source": "derived_from_joint_pos_holomotion_grad_t",
            "dumped_joint_velocity_rmse_rad_s": dumped_joint_vel_rmse,
            "dumped_joint_velocity_first_frame_rmse_rad_s": (
                dumped_joint_vel_first_frame_rmse
            ),
        },
        ensure_ascii=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        fps=np.asarray(fps),
        joint_names=joint_names,
        body_names=body_names,
        metadata=np.asarray(exported_metadata),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": frames,
                "fps": fps,
                "duration_s": (frames - 1) / fps,
                "motion": metadata.get("raw_motion_key"),
                "dumped_joint_velocity_rmse_rad_s": dumped_joint_vel_rmse,
                "dumped_joint_velocity_first_frame_rmse_rad_s": (
                    dumped_joint_vel_first_frame_rmse
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
