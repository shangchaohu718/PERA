"""等待指定 GPU 空闲，再运行 K1 物理反馈 A/B 训练与冻结评测。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_motion.experiment import load_config, run_evaluation, run_training


def gpu_compute_pids(index: int) -> tuple[int, ...]:
    """返回指定 GPU 上的计算进程 PID；查询失败时拒绝误判为空闲。"""
    result = subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "未知错误"
        raise RuntimeError(f"nvidia-smi 查询失败：{detail}")
    pids = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value and value != "[N/A]":
            try:
                pids.append(int(value))
            except ValueError as exc:
                raise RuntimeError(f"无法解析 nvidia-smi PID：{value!r}") from exc
    return tuple(pids)


def wait_until_idle(index: int, poll_seconds: int) -> None:
    """只等待，不结束或暂停任何已有进程。"""
    while True:
        pids = gpu_compute_pids(index)
        if not pids:
            print(f"GPU cuda:{index} 已空闲，开始 A/B 训练。", flush=True)
            return
        print(f"GPU cuda:{index} 仍有计算进程 {list(pids)}；{poll_seconds}s 后重查。",
              flush=True)
        time.sleep(poll_seconds)


def configured(base: dict, *, feedback: bool, args: argparse.Namespace) -> dict:
    """从同一基础配置生成只差反馈开关与输出目录的实验配置。"""
    config = deepcopy(base)
    config["asset_root"] = str(args.asset_root.resolve())
    config["device"] = f"cuda:{args.gpu}"
    config["num_envs"] = args.num_envs
    config["dynamics"]["interaction_feedback"] = feedback
    config["ppo"]["minibatch_envs"] = args.minibatch_envs
    config["run"]["updates"] = args.updates
    config["run"]["horizon"] = args.horizon
    config["run"]["evaluation_episodes"] = args.evaluation_episodes
    config["run"]["evaluation_max_steps"] = args.evaluation_max_steps
    if args.push_interval_steps is not None:
        config["push_interval_steps"] = args.push_interval_steps
    if args.train_push_force is not None:
        for name in config["train_domains"]:
            config["domains"][name]["push_force_n"] = list(args.train_push_force)
    if args.eval_push_force is not None:
        for name in config["evaluation_domains"]:
            config["domains"][name]["push_force_n"] = list(args.eval_push_force)
    if args.hold_steps is not None:
        for stage in config["plan"]:
            stage["hold_steps"] = args.hold_steps
    if args.max_steps is not None:
        for stage in config["plan"]:
            stage["max_steps"] = args.max_steps
    if "evaluation_plan" in config:
        if args.hold_steps is not None:
            for stage in config["evaluation_plan"]:
                stage["hold_steps"] = args.hold_steps
        if args.max_steps is not None:
            for stage in config["evaluation_plan"]:
                stage["max_steps"] = args.max_steps
    label = "feedback_on" if feedback else "feedback_off"
    config["output"] = str((args.output_root / label).resolve())
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/k1_contact.yaml"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("runs/k1_feedback_ablation"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--minibatch-envs", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--evaluation-max-steps", type=int, default=2000)
    parser.add_argument("--push-interval-steps", type=int,
                        help="每隔多少控制步施加一次侧推")
    parser.add_argument("--train-push-force", type=float, nargs=2,
                        metavar=("LOW", "HIGH"), help="训练域侧推力范围（N）")
    parser.add_argument("--eval-push-force", type=float, nargs=2,
                        metavar=("LOW", "HIGH"), help="评测域侧推力范围（N）")
    parser.add_argument("--hold-steps", type=int,
                        help="成功前必须连续满足目标的步数")
    parser.add_argument("--max-steps", type=int, help="每阶段最大控制步数")
    parser.add_argument("--no-wait", action="store_true",
                        help="立即运行；仅用于已经人工确认 GPU 空闲的情况")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="复用输出目录中的checkpoint，只重跑冻结评测与对比")
    args = parser.parse_args(argv)

    for name in ("gpu",):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} 不能为负数")
    for name in ("poll_seconds", "num_envs", "updates", "horizon",
                 "minibatch_envs", "evaluation_episodes", "evaluation_max_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} 必须是正整数")
    if args.minibatch_envs > args.num_envs:
        parser.error("--minibatch-envs 不能大于 --num-envs")
    for name in ("push_interval_steps", "hold_steps", "max_steps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须是正整数")
    if (args.hold_steps is not None and args.max_steps is not None
            and args.hold_steps > args.max_steps):
        parser.error("--hold-steps 不能大于 --max-steps")
    for name in ("train_push_force", "eval_push_force"):
        value = getattr(args, name)
        if value is not None and (value[0] < 0 or value[0] > value[1]):
            parser.error(f"--{name.replace('_', '-')} 必须是非负递增范围")
    if not args.asset_root.is_dir():
        parser.error(f"资产目录不存在：{args.asset_root}")

    base = load_config(args.config)
    if base["backend"] != "mjlab_contact":
        parser.error("基础配置必须使用 mjlab_contact，不能用诊断环境冒充")
    configs = {
        "feedback_on": configured(base, feedback=True, args=args),
        "feedback_off": configured(base, feedback=False, args=args),
    }
    if not args.no_wait and not args.evaluate_only:
        wait_until_idle(args.gpu, args.poll_seconds)

    results = {}
    for label, config in configs.items():
        if args.evaluate_only:
            checkpoint=Path(config["output"])/"checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"没有可复评的checkpoint：{checkpoint}")
            print(f"复评 {label}：{checkpoint}", flush=True)
        else:
            print(f"开始 {label}：{config['output']}", flush=True)
            checkpoint = run_training(config)
        results[label] = run_evaluation(config, checkpoint)

    on, off = results["feedback_on"], results["feedback_off"]
    comparison = {
        "schema": 1,
        "same_seed": configs["feedback_on"]["seed"] == configs["feedback_off"]["seed"],
        "feedback_on": on,
        "feedback_off": off,
        "on_minus_off": {
            "success_rate": on["success_rate"] - off["success_rate"],
            "mean_position_error_m": on["mean_position_error_m"] - off["mean_position_error_m"],
            "mean_relative_speed_error_m_s": (
                on["mean_relative_speed_error_m_s"]
                - off["mean_relative_speed_error_m_s"]
            ),
            "mean_stage_index": on["mean_stage_index"] - off["mean_stage_index"],
            "maximum_stage_index_observed": (
                on["maximum_stage_index_observed"]
                - off["maximum_stage_index_observed"]
            ),
        },
        "interpretation": (
            "反馈有效应表现为 success_rate 更高，并且位置/相对速度误差更低；"
            "单次种子结果仍需多种子复验。"
        ),
    }
    if "push_recovery_rate" in on and "push_recovery_rate" in off:
        comparison["on_minus_off"].update(
            push_recovery_rate=on["push_recovery_rate"]-off["push_recovery_rate"],
            mean_push_recovery_steps=(
                on["mean_push_recovery_steps"]-off["mean_push_recovery_steps"]
            ),
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "comparison.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2,
                                 allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "comparison": str(output.resolve())},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
