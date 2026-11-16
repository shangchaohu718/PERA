"""Enforce the user's official-HoloMotion-only demonstration selection."""

import json

from pera.paths import ROOT


def enforce_demo_policy(config):
    policy = json.loads((ROOT / "configs/demo_policy.json").read_text())
    if (
        config.get("base_family") != policy["base_family"]
        or config.get("public_sha256") != policy["public_sha256"]
    ):
        raise ValueError(
            "Demonstrations require the official HoloMotion G1 v1.4 model_14000 "
            "Actor and its matching experts. This archived showcase targets the "
            "self-trained base. Use the current official showcase config. "
            "See configs/demo_policy.json. No simulation was started."
        )
