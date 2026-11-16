"""Pure metadata for the unchanged G1 policy/FK, without a second simulator."""

import json
from types import SimpleNamespace
import numpy as np
from blinkmimic.sensing import SensingCfg

from pera.paths import ROOT


def load_motion(path):
    with np.load(path, allow_pickle=False) as d:
        if d["joint_pos"].shape[-1] != 29 or float(d["fps"]) != 50:
            raise ValueError("G1 motion contract mismatch")
        return (
            d["root_pos"].copy(),
            d["root_quat_wxyz"].copy(),
            d["joint_pos"].copy(),
            float(d["fps"]),
        )


def get_robot(name):
    if name != "g1":
        raise ValueError("Only G1 supported")
    d = json.loads((ROOT / "configs/robot.json").read_text())
    q0 = d.pop("default_joint_pos_list")
    alpha = d.pop("action_scale_list")
    d["mjcf_path"] = str(ROOT / d["mjcf_path"])
    d["motion_dir"] = str(ROOT / d["motion_dir"])
    d["load_motion"] = load_motion
    d["sensing"] = SensingCfg(**d["sensing"])
    return SimpleNamespace(
        **d,
        default_joint_pos_list=lambda: list(q0),
        action_scale_list=lambda: list(alpha),
    )
