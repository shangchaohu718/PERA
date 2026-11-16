"""Record project imports and reject accidental research-tree dependencies."""

import importlib.metadata
from pathlib import Path
import sys

from pera.paths import ROOT

PROJECT_MODULES = (
    "pera",
    "blinkmimic",
    "modumotion_g1",
    "holomotion",
    "experiments",
    "athlete",
    "instinct_mj",
    "instinct_rl",
    "mjlab",
    "runtime",
    "sim2sim_teacher",
    "learning",
    "utils",
)


def imported_sources(strict=True):
    local, external = set(), []
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in PROJECT_MODULES):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        filename = Path(source).resolve()
        if filename.is_relative_to(ROOT):
            local.add(str(filename.relative_to(ROOT)))
        else:
            external.append({"module": name, "file": str(filename)})
    if strict and external:
        raise RuntimeError(f"Project source loaded outside the release checkout: {external}")
    return {
        "checkout_only_project_sources": not external,
        "files": sorted(local),
        "external": external,
    }


def versions():
    names = [
        "torch",
        "torchvision",
        "numpy",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "onnx",
        "onnxruntime-gpu",
        "onnxruntime",
        "tyro",
        "trimesh",
        "rl-games",
        "tensordict",
        "imageio",
        "imageio-ffmpeg",
        "Pillow",
        "scipy",
        "h5py",
        "omegaconf",
        "hydra-core",
        "safetensors",
        "mediapy",
        "einops",
        "loguru",
        "gymnasium",
        "torchrunx",
    ]
    result = {"python": sys.version.split()[0]}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
