"""Explicit dependencies; never inject packages from a different environment."""

import importlib.util


def ensure_fk_parser():
    if importlib.util.find_spec("mujoco") is None:
        raise ImportError(
            "MuJoCo is required for G1 FK. Install the PERA native extra in this interpreter; no packages are borrowed from another venv."
        )
