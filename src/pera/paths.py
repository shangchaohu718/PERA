"""Portable release paths. No developer environment is inferred."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PERA_ROOT", Path(__file__).resolve().parents[2])).resolve()


def resource(name):
    return ROOT / name
