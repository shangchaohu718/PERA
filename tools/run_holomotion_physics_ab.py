"""Launch a matched from-scratch K1 Actor baseline and full-Physics run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLOMOTION_CONFIG = ROOT / "configs/actors/holomotion_k1_16e.yaml"
DEFAULT_CUSTOM_ACTOR_CONFIG = ROOT / "configs/actors/k1_small.yaml"
TRAINER = ROOT / "tools/train_holomotion_k1_mjlab.py"
ACTORS = ("holomotion", "mlp", "transformer", "moe", "gru", "lstm", "lru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    motion = parser.add_mutually_exclusive_group(required=True)
    motion.add_argument("--motion", type=Path, action="append")
    motion.add_argument("--motion-dir", type=Path)
    contract = parser.add_mutually_exclusive_group(required=True)
    contract.add_argument("--onnx", type=Path)
    contract.add_argument("--task-contract", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--actor", choices=ACTORS, default="holomotion")
    parser.add_argument("--holomotion-config", type=Path, default=DEFAULT_HOLOMOTION_CONFIG)
    parser.add_argument("--actor-config", type=Path, default=DEFAULT_CUSTOM_ACTOR_CONFIG)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    common = [
        sys.executable,
        str(TRAINER),
        "--actor", args.actor,
        "--asset-root", str(args.asset_root),
        "--iterations", str(args.iterations),
        "--steps-per-env", str(args.steps_per_env),
        "--num-envs", str(args.num_envs),
        "--save-interval", str(args.save_interval),
        "--log-interval", str(args.log_interval),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.actor == "holomotion":
        common.extend(("--holomotion-config", str(args.holomotion_config)))
    else:
        common.extend(("--actor-config", str(args.actor_config)))
    if args.onnx is not None:
        common.extend(("--onnx", str(args.onnx)))
    else:
        common.extend(("--task-contract", str(args.task_contract)))
    if args.motion_dir is not None:
        common.extend(("--motion-dir", str(args.motion_dir)))
    else:
        for path in args.motion:
            common.extend(("--motion", str(path)))
    root = args.output_root.expanduser().resolve()
    return {
        "baseline": [
            *common,
            "--matched-physics-task",
            "--output-dir", str(root / "baseline_matched"),
        ],
        "physics": [
            *common,
            "--physics",
            *(["--finetune-base-actor"] if args.actor == "holomotion" else []),
            "--output-dir", str(root / "physics_full"),
        ],
    }


def _plan_signature(plan: dict) -> dict:
    """Ignore only the Python executable when deciding whether a retry matches."""
    signature = dict(plan)
    signature["commands"] = {
        name: command[1:] for name, command in plan["commands"].items()
    }
    return signature


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("mjlab") is None:
        raise RuntimeError(
            f"mjlab is not installed in {sys.executable}; run this launcher with "
            "a Python environment that contains mjlab"
        )
    commands = build_commands(args)
    plan = {
        "comparison": f"matched {args.actor} from-scratch baseline vs full Physics",
        "same_initialization": True,
        "seed": args.seed,
        "actor": args.actor,
        "actor_config": str((args.holomotion_config if args.actor == "holomotion"
                             else args.actor_config).expanduser().resolve()),
        "physics": {
            "scene_encoder": "set",
            "physics_width": 64,
            "memory_width": 64,
            "latent_width": 8,
            "hidden_dim": 128,
            "teacher": False,
            "base_actor_trainable": True,
        },
        "commands": commands,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "ab_plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if _plan_signature(previous) != _plan_signature(plan):
            raise FileExistsError(
                f"A different A/B plan already exists: {plan_path}"
            )
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else os.pathsep.join((str(ROOT), existing_pythonpath))
    )
    for command in commands.values():
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
