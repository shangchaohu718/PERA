"""Portable command line; importing/help never starts a simulator."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from .paths import ROOT


def environment():
    env = os.environ.copy()
    paths = [
        ROOT / "src",
        ROOT / "reproduction",
        ROOT / "integrations/HoloMotion",
        ROOT / "vendor/mjlab/src",
        ROOT / "vendor/InstinctMJ/src",
        ROOT / "vendor/instinct_rl",
    ]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def doctor(resources=False):
    packages = ["torch", "numpy", "mujoco", "mjlab", "omegaconf", "h5py"]
    result = {
        "root": str(ROOT),
        "python": sys.executable,
        "CUDA_initialized": False,
        "packages": {p: importlib.util.find_spec(p) is not None for p in packages},
    }
    missing = []
    if resources:
        manifest = json.loads((ROOT / "resources/holomotion_weights_manifest.json").read_text())
        missing = [x["path"] for x in manifest["resources"] if not (ROOT / x["path"]).is_file()]
        result["base_family"] = manifest["base_family"]
        result["missing_base_resources"] = missing
        result["resource_check_scope"] = (
            "Model file presence only; not a runtime/reproduction check"
        )
    print(json.dumps(result, indent=2))
    return 2 if missing else 0


def main():
    if sys.argv[1:2] == ["demo"]:
        env = environment()
        if "--video" in sys.argv and "--viewer" not in sys.argv:
            env.setdefault("MUJOCO_GL", "egl")
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("--config", type=Path, default=ROOT / "configs/showcase.json")
        config_args, _ = config_parser.parse_known_args(sys.argv[2:])
        config_path = config_args.config
        config = json.loads(config_path.read_text())
        if not any(arg in ("--help", "-h") for arg in sys.argv[2:]):
            from .demo_policy import enforce_demo_policy

            try:
                enforce_demo_policy(config)
            except ValueError as exc:
                config_parser.error(str(exc))
        module = (
            "pera.interaction_showcase"
            if config.get("runtime") == "interaction"
            else "pera.showcase"
        )
        return subprocess.call([sys.executable, "-m", module, *sys.argv[2:]], cwd=ROOT, env=env)
    p = argparse.ArgumentParser(
        description="PERA official HoloMotion scene replay and expert inspection."
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="Inspect dependencies; no CUDA initialization")
    d.add_argument("--resources", action="store_true")
    sub.add_parser("check", help="Run CPU routing/freeze compatibility checks; no skill evaluation")
    sub.add_parser(
        "demo", help="Run shared-world multi-robot inference; use demo --help for options"
    )
    e = sub.add_parser("experts", help="List the experts in the official HoloMotion bank")
    e.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/holomotion_experts/pera_25skills.experts.pt",
    )
    a = p.parse_args()
    if a.command == "doctor":
        return doctor(a.resources)
    if a.command == "check":
        from .smoke import main as check

        return check()
    if a.command == "experts":
        import torch

        payload = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format") != "pera_official_holomotion_bank_v1":
            p.error(
                "Expected the official HoloMotion expert bank; historical banks are unsupported"
            )
        print(
            json.dumps(
                {
                    "max_active_experts": payload["max_active_experts_per_inference"],
                    "experts": [
                        {"name": name, "context": entry["record"].get("context")}
                        for name, entry in payload["experts"].items()
                    ],
                    "inspection_only": True,
                },
                indent=2,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
