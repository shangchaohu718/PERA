"""在同一种子、任务和预算下运行五组物理适应结构消融。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_motion.experiment import load_config, run_evaluation, run_training
from tools.wait_and_train_feedback_ablation import wait_until_idle


VARIANTS = {
    "asymmetric_only": {
        "actor_dynamics": "none", "system_identification_weight": 0.0,
        "teacher_distillation_weight": 0.0, "teacher_reconstruction_weight": 0.0,
        "deployable": True, "meaning": "特权Critic；Actor不读取历史动力学latent",
    },
    "predictive_online": {
        "actor_dynamics": "online", "system_identification_weight": 0.0,
        "teacher_distillation_weight": 0.0, "teacher_reconstruction_weight": 0.0,
        "deployable": True, "meaning": "Actor读取仅由可观测历史和预测误差形成的latent",
    },
    "direct_system_id": {
        "actor_dynamics": "online", "system_identification_weight": 0.1,
        "teacher_distillation_weight": 0.0, "teacher_reconstruction_weight": 0.0,
        "deployable": True, "meaning": "在线latent额外直接回归仿真物理真值",
    },
    "teacher_student": {
        "actor_dynamics": "online", "system_identification_weight": 0.1,
        "teacher_distillation_weight": 0.05, "teacher_reconstruction_weight": 0.1,
        "deployable": True, "meaning": "在线学生对齐stop-gradient Teacher物理latent",
    },
    "oracle_upper_bound": {
        "actor_dynamics": "oracle", "system_identification_weight": 0.0,
        "teacher_distillation_weight": 0.0, "teacher_reconstruction_weight": 0.0,
        "deployable": False, "meaning": "Actor直接读取仿真真值；仅作不可部署上限",
    },
}


def configured(base: dict, label: str, args: argparse.Namespace) -> dict:
    config = deepcopy(base)
    config["asset_root"] = str(args.asset_root.resolve())
    config["device"] = f"cuda:{args.gpu}"
    config["num_envs"] = args.num_envs
    config["ppo"]["minibatch_envs"] = args.minibatch_envs
    config["run"].update(updates=args.updates, horizon=args.horizon,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_max_steps=args.evaluation_max_steps)
    variant = VARIANTS[label]
    config["adaptation"] = {
        "actor_observation": "deployable", "actor_dynamics": variant["actor_dynamics"],
        "privileged_critic": True,
        "system_identification_weight": variant["system_identification_weight"],
        "teacher_distillation_weight": variant["teacher_distillation_weight"],
        "teacher_reconstruction_weight": variant["teacher_reconstruction_weight"],
    }
    config["output"] = str((args.output_root / label).resolve())
    return config


def metric_delta(result: dict, baseline: dict) -> dict:
    names = ("success_rate", "mean_position_error_m", "mean_relative_speed_error_m_s",
             "mean_stage_index", "maximum_stage_index_observed", "push_recovery_rate",
             "mean_push_recovery_steps")
    return {name: result[name]-baseline[name] for name in names
            if name in result and name in baseline}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/k1_lateral_step.yaml"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/k1_adaptation_ablation"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--minibatch-envs", type=int, default=32)
    parser.add_argument("--evaluation-episodes", type=int, default=2)
    parser.add_argument("--evaluation-max-steps", type=int, default=1200)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    for name in ("poll_seconds", "num_envs", "updates", "horizon", "minibatch_envs",
                 "evaluation_episodes", "evaluation_max_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_','-')}必须是正整数")
    if args.gpu < 0 or args.minibatch_envs > args.num_envs:
        parser.error("GPU编号必须非负，且minibatch-envs不能大于num-envs")
    if not args.asset_root.is_dir():
        parser.error(f"资产目录不存在：{args.asset_root}")

    base = load_config(args.config)
    if base["backend"] != "mjlab_contact":
        parser.error("消融必须使用真实mjlab_contact后端")
    configs = {label: configured(base, label, args) for label in VARIANTS}
    if not args.no_wait and not args.evaluate_only:
        wait_until_idle(args.gpu, args.poll_seconds)

    results = {}
    for label, config in configs.items():
        checkpoint = Path(config["output"])/"checkpoint.pt"
        if args.evaluate_only:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"没有可复评checkpoint：{checkpoint}")
            print(f"复评 {label}", flush=True)
        else:
            print(f"训练 {label}：{VARIANTS[label]['meaning']}", flush=True)
            checkpoint = run_training(config)
        results[label] = run_evaluation(config, checkpoint)

    baseline = results["asymmetric_only"]
    comparison = {
        "schema": 1, "same_seed": len({c["seed"] for c in configs.values()}) == 1,
        "baseline": "asymmetric_only",
        "variants": {label: {
            "deployable": VARIANTS[label]["deployable"],
            "meaning": VARIANTS[label]["meaning"], "evaluation": result,
            "minus_baseline": metric_delta(result, baseline),
        } for label, result in results.items()},
        "interpretation": ("先比较四个可部署组；oracle_upper_bound只衡量真值信息可能带来的上限。"
                           "单种子短训练只验证链路，不足以确定算法优劣。"),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root/"comparison.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2,
                                 allow_nan=False), encoding="utf-8")
    print(json.dumps({"status":"PASS", "comparison":str(output.resolve())},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
