"""Private experiment paths and immutable G1 policy/reference contracts."""

import hashlib
import json
import os
from pathlib import Path
import sys

from pera.paths import ROOT

CODE = Path(__file__).resolve().parent
REPORT = Path(os.environ.get("PERA_OUTPUT", ROOT / "runs/martial"))
BASE = Path(os.environ.get("PERA_BASE_CHECKPOINT", ROOT / "checkpoints/base/best.pt"))
BASE_SHA = "fc5df7ad6ec83cce71853a30314e6c4c0109baab144308084279ded0679cb2eb"
NATIVE = ROOT / "src"
SOURCE = ROOT / "reports/kungfubot_original_20261105/references_h5"
OLD_CONFIG = ROOT / "configs/martial_isaac.yaml"


def bootstrap():
    # Use the active training Python; importing native robot.py would pull in
    # the unrelated MuJoCo viewer stack, so g1_spec reads static metadata.
    for path in (ROOT, ROOT / "integrations/HoloMotion", NATIVE, CODE):
        sys.path.insert(0, str(path))
    import site

    for package in site.getsitepackages():
        isaac = Path(package) / "isaaclab/source/isaaclab"
        if isaac.is_dir():
            sys.path.insert(0, str(isaac))


def g1_spec():
    """Use the one saved SDK/actuator contract; no source-code execution."""
    from pera.robot import get_robot

    return get_robot("g1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def motion_library(robot, device):
    import h5py
    import numpy as np
    from blinkmimic.kinematics import Kinematics
    from blinkmimic.motion import MotionLib
    from holomotion.src.training.h5_dataloader import _WorldFrameNormalizeTransform

    manifest = json.loads((SOURCE / "manifest.json").read_text())
    names = list(manifest["clips"])
    assert len(names) == 7

    def load(name):
        key = Path(name).stem
        clip = manifest["clips"][key]
        assert clip["metadata"]["dof_names"] == list(robot.joint_names)
        assert clip["metadata"]["quaternion_convention"] == "xyzw"
        with h5py.File(SOURCE / manifest["hdf5_shards"][clip["shard"]]["file"], "r") as f:
            q = f["ref_dof_pos"][:].astype(np.float32)
            p = f["ref_root_pos"][:].astype(np.float32)
            quat = f["ref_root_rot"][:][:, [3, 0, 1, 2]].astype(np.float32)
        assert len(q) == clip["length"] and np.isfinite(q).all()
        # HoloMotion's hdf5_v2 loader always centers XY/removes initial yaw,
        # even if a legacy YAML says world_frame_normalization:false. Use
        # the very same transform for policy goals and initialization.
        import torch

        normalized = dict(
            ref_rg_pos=torch.from_numpy(p.copy())[:, None],
            ref_rb_rot=torch.from_numpy(quat[:, [1, 2, 3, 0]].copy())[:, None],
            ref_body_vel=torch.zeros(len(p), 1, 3),
            ref_body_ang_vel=torch.zeros(len(p), 1, 3),
        )
        _WorldFrameNormalizeTransform()(normalized)
        p = normalized["ref_rg_pos"][:, 0].numpy()
        quat = normalized["ref_rb_rot"][:, 0][:, [3, 0, 1, 2]].numpy()
        return p, quat, q, 50.0

    kin = Kinematics(robot, device)
    motion = MotionLib([n + ".npz" for n in names], kin, device, load)
    assert motion.clip_len.tolist() == [manifest["clips"][n]["length"] for n in names]
    return motion, kin, manifest
