"""Convert original HoloMotion HDF5 windows into named project NPZ clips.

The original Hdf5RootDofDataset and TrainingReferenceFK remain responsible for
window enumeration, world normalization, forward kinematics, and velocities.
This tool only serializes their output into MotionLibrary's explicit contract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from unified_motion.data.motion_sampler import MotionClip

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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--robot-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frame-length", type=int, default=300)
    parser.add_argument("--min-window-length", type=int, default=50)
    parser.add_argument(
        "--motion-name",
        action="append",
        dest="motion_names",
        help=(
            "convert only this manifest motion key; repeat to select multiple "
            "motions (default: convert every motion)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="convert only the first N windows for a bounded validation run",
    )
    args = parser.parse_args()
    if args.max_frame_length < 1 or args.min_window_length < 1:
        parser.error("frame lengths must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def _xyzw_to_wxyz(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., 3:], value[..., :3]), dim=-1)


def main() -> None:
    args = parse_args()
    root = args.holomotion_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from holomotion.src.training.h5_dataloader import Hdf5RootDofDataset
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError(
            "conversion requires the original HoloMotion runtime and "
            "pytorch_kinematics"
        ) from error

    config = OmegaConf.load(args.config.expanduser().resolve())
    joint_names = tuple(str(name) for name in config.robot.dof_names)
    motion_config = config.robot.motion
    excluded = motion_config.get("excluded_motion_names", None)
    handpicked = (
        args.motion_names
        if args.motion_names is not None
        else motion_config.get("handpicked_motion_names", None)
    )
    dataset = Hdf5RootDofDataset(
        manifest_path=str(args.manifest.expanduser().resolve()),
        max_frame_length=args.max_frame_length,
        min_window_length=args.min_window_length,
        handpicked_motion_names=(
            None if handpicked is None else [str(name) for name in handpicked]
        ),
        excluded_motion_names=(
            None if excluded is None else [str(name) for name in excluded]
        ),
        fk_robot_file_path=str(args.robot_file.expanduser().resolve()),
        expected_dof_names=joint_names,
        allowed_prefixes=("ref_",),
    )
    body_names = tuple(dataset._fk_transform._fk.body_names)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    count = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    try:
        for index in range(count):
            sample = dataset[index]
            arrays = sample.tensors
            destination = output / f"{sample.motion_key}.npz"
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite {destination}")
            fps = float(arrays["motion_fps"][0, 0])
            clip = MotionClip(
                name=sample.motion_key,
                robot_id="k1_22dof",
                root_name="trunk",
                joint_names=joint_names,
                fps=fps,
                joint_pos=arrays["ref_dof_pos"],
                joint_vel=arrays["ref_dof_vel"],
                root_pos_w=arrays["ref_root_pos"],
                root_quat_wxyz=_xyzw_to_wxyz(arrays["ref_root_rot"]),
                root_lin_vel_w=arrays["ref_root_vel"],
                root_ang_vel_w=arrays["ref_root_ang_vel"],
                body_names=body_names,
                body_pos_w=arrays["ref_rg_pos"],
                body_quat_wxyz=_xyzw_to_wxyz(arrays["ref_rb_rot"]),
                body_lin_vel_w=arrays["ref_body_vel"],
                body_ang_vel_w=arrays["ref_body_ang_vel"],
            )
            clip.save_project_npz(destination)
    finally:
        dataset.close()
    print(f"converted {count} windows into {output}")


if __name__ == "__main__":
    main()
