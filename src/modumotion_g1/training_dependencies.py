"""Explicit prerequisites for optional native teacher/training adapters.

The scene replay vendors only its inference subset; it never calls these checks.
No source, Python package, model or environment is downloaded here.
"""

import importlib.util
import sys

from pera.paths import ROOT


def require_instinct_teacher_sources():
    sources = (
        ("mjlab", "src/mjlab/__init__.py", "src"),
        ("InstinctMJ", "src/instinct_mj/tasks/registry.py", "src"),
        ("instinct_rl", "instinct_rl/modules/__init__.py", ""),
    )
    missing = [
        name for name, marker, _ in sources if not (ROOT / "third_party" / name / marker).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Native teacher training needs the full pinned upstream sources: "
            + ", ".join(missing)
            + ". vendor/ is a scene-inference subset. See docs/TRAINING.md; "
            "preview with python tools/fetch_upstreams.py --plan. Nothing was downloaded."
        )
    for package in ("mjlab", "instinct_mj", "instinct_rl"):
        imported = sys.modules.get(package)
        filename = getattr(imported, "__file__", None)
        if filename and not str(filename).startswith(str(ROOT / "third_party") + "/"):
            raise RuntimeError(
                f"{package} was already imported from another runtime. "
                "Use a separate training process; see docs/TRAINING.md."
            )
    paths = [str(ROOT / "third_party" / name / subdir) for name, _, subdir in sources]
    sys.path[:0] = [path for path in paths if path not in sys.path]


def require_isaaclab():
    if importlib.util.find_spec("isaaclab") is None:
        raise RuntimeError(
            "This optional PPO adapter needs an externally installed Isaac Lab runtime. "
            "It is not part of the MuJoCo scene replay dependencies. See docs/TRAINING.md."
        )
