"""Portable command line; importing/help never starts a simulator."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from .paths import ROOT

WORKFLOWS = {
    "distill": "experiments.g1_competitive_suite31_v1.pipeline",
    "ppo": "experiments.blink_competitive_martial_ppo_v1_20261106.train",
    "stairs-ab": "experiments.blink_stairs_routing_ab_20261106.train",
    "consolidate": "experiments.blink_stairs_full_finetune_20261106.finetune",
    "collect-old": "experiments.blink_stairs_full_finetune_20261106.collect_old",
    "retain": "experiments.g1_competitive_suite31_v1.evaluate_retention",
    "retain-ppo": "experiments.blink_competitive_martial_ppo_v1_20261106.evaluate_retention",
    "retain-ab": "experiments.blink_stairs_routing_ab_20261106.evaluate_retention",
    "retain-full": "experiments.blink_stairs_full_finetune_20261106.evaluate_retention",
    "transfer-prepare": "experiments.martial_transfer_v1_20261106.prepare",
    "transfer-eval": "experiments.martial_transfer_v1_20261106.evaluate",
    "evaluate-base": "blinkmimic.evaluate",
}


def environment():
    env = os.environ.copy()
    paths = [
        ROOT / "src",
        ROOT / "reproduction",
        ROOT / "integrations/HoloMotion",
        ROOT / "third_party/mjlab/src",
        ROOT / "third_party/InstinctMJ/src",
        ROOT / "third_party/instinct_rl",
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
        description="PERA HoloMotion release preparation. No default training or downloads."
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="Inspect dependencies; no CUDA initialization")
    d.add_argument("--resources", action="store_true")
    sub.add_parser("check", help="Run historical self-trained CPU routing/freeze contract checks")
    sub.add_parser(
        "demo", help="Run shared-world multi-robot inference; use demo --help for options"
    )
    c = sub.add_parser("compose", help="Compose historical self-trained competitive checkpoints")
    c.add_argument("--config", type=Path, default=ROOT / "configs/expert_bank.json")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--device", default="cpu")
    e = sub.add_parser("experts", help="List installed experts in a composed Actor")
    e.add_argument("--checkpoint", type=Path, required=True)
    t = sub.add_parser("train-base", help="Run the historical self-trained base recipe")
    t.add_argument("--config", type=Path, default=ROOT / "configs/base_train.json")
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--resume", type=Path)
    t.add_argument("--dry-run", action="store_true")
    r = sub.add_parser(
        "reproduce", help="Run a historical v2.4 workflow; pass workflow arguments after --"
    )
    r.add_argument("workflow", choices=WORKFLOWS)
    r.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command == "doctor":
        return doctor(a.resources)
    if a.command == "check":
        from .smoke import main as check

        return check()
    if a.command in {"compose", "experts"}:
        from .switching import SwitchedActor, compose_config, file_digest

        if a.command == "compose":
            bank = compose_config(a.config, device=a.device)
            bank.save_bundle(a.output)
            print(
                json.dumps(
                    dict(
                        checkpoint=str(a.output),
                        sha256=file_digest(a.output),
                        bytes=a.output.stat().st_size,
                        expert_count=len(bank.expert_names),
                        max_active_experts=1,
                        active_expert=bank.active_expert,
                    ),
                    indent=2,
                )
            )
        else:
            bank = SwitchedActor.load_bundle(a.checkpoint)
            print(
                json.dumps(
                    dict(
                        max_active_experts=1,
                        active_expert=bank.active_expert,
                        experts=[
                            dict(
                                name=n,
                                context_dim=bank.expert_specs[n]["context_dim"],
                                enabled=bank.switches[n],
                            )
                            for n in bank.expert_names
                        ],
                    ),
                    indent=2,
                )
            )
        return 0
    if a.command == "train-base":
        cfg = json.loads(a.config.read_text())
        cfg["log_dir"] = str(a.output.resolve())
        if a.resume:
            cfg["resume"] = str(a.resume.resolve())
        for k in ["motion_dir", "resume"]:
            if cfg.get(k) and not Path(cfg[k]).is_absolute():
                cfg[k] = str(ROOT / cfg[k])
        cmd = [sys.executable, "-m", "blinkmimic.train"]
        for k, v in cfg.items():
            if v is True:
                cmd.append("--" + k)
            elif v is not None and v is not False:
                cmd.extend(["--" + k, str(v)])
        if a.dry_run:
            print(json.dumps(cmd, indent=2))
            return 0
        if not Path(cfg["motion_dir"]).is_dir():
            p.error("Base motions are absent; see docs/RESOURCES.md")
        if (a.output / "args.json").exists():
            p.error(
                "Output already contains a run; use a new directory or an explicit resume workflow"
            )
        return subprocess.call(cmd, cwd=ROOT, env=environment())
    args = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
    if not any(x in args for x in ["--help", "-h"]) and a.workflow not in ["transfer-prepare"]:
        if not (ROOT / "checkpoints/base/best.pt").is_file():
            p.error(
                "Missing checkpoints/base/best.pt; see docs/RESOURCES.md or run doctor --resources"
            )
    if a.workflow == "distill" and "--config" not in args:
        args = ["--config", str(ROOT / "configs/distillation31.json"), *args]
    return subprocess.call(
        [sys.executable, "-m", WORKFLOWS[a.workflow], *args], cwd=ROOT, env=environment()
    )


if __name__ == "__main__":
    raise SystemExit(main())
