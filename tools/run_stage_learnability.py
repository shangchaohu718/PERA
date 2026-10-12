"""定位 K1 侧移任务的可学习阶段：仅站稳，以及左脚抬高 3/6/9 cm。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_motion.experiment import load_config, run_evaluation, run_training


HEIGHTS = {
    "settle_only": None,
    "lift_03m": 0.03,
    "lift_06m": 0.06,
    "lift_09m": 0.09,
}


def configured(base: dict, label: str, args: argparse.Namespace) -> dict:
    """生成除阶段终点外训练预算和网络均相同的配置。"""
    config=deepcopy(base)
    config["asset_root"]=str(args.asset_root.resolve())
    config["device"]=f"cuda:{args.gpu}"
    config["num_envs"]=args.num_envs
    config["ppo"]["minibatch_envs"]=args.minibatch_envs
    config["run"].update(updates=args.updates,horizon=args.horizon,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_max_steps=args.evaluation_max_steps)
    config["adaptation"]={
        "actor_observation":"deployable", "actor_dynamics":"none",
        "privileged_critic":True, "system_identification_weight":0.0,
        "teacher_distillation_weight":0.0, "teacher_reconstruction_weight":0.0,
    }
    height=HEIGHTS[label]
    if height is None:
        config["plan"]=deepcopy(base["plan"][:1])
    else:
        config["plan"]=deepcopy(base["plan"][:2])
        lift=config["plan"][1]
        lift["name"]=f"lift_left_{int(height*100):02d}cm"
        for target in lift["targets"]:
            if target["pair"]=="left_ground":
                position=list(target["position"]);position[2]=height
                target["position"]=position
                break
        else:
            raise ValueError("lift_left阶段缺少left_ground目标")
    config.pop("evaluation_plan",None)
    config["output"]=str((args.output_root/label).resolve())
    return config


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/k1_lateral_step.yaml"))
    parser.add_argument("--asset-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=Path("runs/k1_stage_learnability"))
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--num-envs",type=int,default=64)
    parser.add_argument("--updates",type=int,default=50)
    parser.add_argument("--horizon",type=int,default=16)
    parser.add_argument("--minibatch-envs",type=int,default=16)
    parser.add_argument("--evaluation-episodes",type=int,default=2)
    parser.add_argument("--evaluation-max-steps",type=int,default=400)
    parser.add_argument("--evaluate-only",action="store_true")
    args=parser.parse_args(argv)
    for name in ("num_envs","updates","horizon","minibatch_envs",
                 "evaluation_episodes","evaluation_max_steps"):
        if getattr(args,name)<1: parser.error(f"--{name.replace('_','-')}必须是正整数")
    if args.gpu<0 or args.minibatch_envs>args.num_envs:
        parser.error("GPU编号必须非负，且minibatch-envs不能大于num-envs")
    if not args.asset_root.is_dir(): parser.error(f"资产目录不存在：{args.asset_root}")
    base=load_config(args.config)
    if base["backend"]!="mjlab_contact": parser.error("必须使用mjlab_contact")
    configs={name:configured(base,name,args) for name in HEIGHTS}
    results={}
    for label,config in configs.items():
        checkpoint=Path(config["output"])/"checkpoint.pt"
        if args.evaluate_only:
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        else:
            print(f"训练阶段诊断 {label}",flush=True)
            checkpoint=run_training(config)
        results[label]=run_evaluation(config,checkpoint)
    comparison={
        "schema":1,"same_seed":len({x["seed"] for x in configs.values()})==1,
        "actor_dynamics":"none",
        "variants":{name:{"target_height_m":HEIGHTS[name],"evaluation":result}
                    for name,result in results.items()},
        "interpretation":"settle_only验证基本闭环；3/6/9cm用于定位左脚离地的学习边界。",
    }
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=args.output_root/"comparison.json"
    output.write_text(json.dumps(comparison,ensure_ascii=False,indent=2,
                                 allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"PASS","comparison":str(output.resolve())},
                     ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
