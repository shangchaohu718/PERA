"""固定 3 cm 抬脚任务，比较动作探索标准差能否跨过离地接触门槛。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from unified_motion.experiment import load_config,run_evaluation,run_training
from tools.run_stage_learnability import configured as stage_configured


STDS={"std_015":0.15,"std_035":0.35,"std_060":0.60}


def configured(base:dict,label:str,args:argparse.Namespace)->dict:
    config=stage_configured(base,"lift_03m",args)
    config["model"]["init_std"]=STDS[label]
    config["output"]=str((args.output_root/label).resolve())
    return config


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/k1_lateral_step.yaml"))
    parser.add_argument("--asset-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=Path("runs/k1_lift_exploration"))
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--num-envs",type=int,default=64)
    parser.add_argument("--updates",type=int,default=50)
    parser.add_argument("--horizon",type=int,default=16)
    parser.add_argument("--minibatch-envs",type=int,default=16)
    parser.add_argument("--evaluation-episodes",type=int,default=2)
    parser.add_argument("--evaluation-max-steps",type=int,default=400)
    args=parser.parse_args(argv)
    for name in ("num_envs","updates","horizon","minibatch_envs",
                 "evaluation_episodes","evaluation_max_steps"):
        if getattr(args,name)<1:parser.error(f"--{name.replace('_','-')}必须是正整数")
    if args.gpu<0 or args.minibatch_envs>args.num_envs:
        parser.error("GPU编号必须非负，且minibatch-envs不能大于num-envs")
    if not args.asset_root.is_dir():parser.error(f"资产目录不存在：{args.asset_root}")
    base=load_config(args.config)
    configs={name:configured(base,name,args) for name in STDS}
    results={}
    for label,config in configs.items():
        print(f"训练 3cm 抬脚探索组 {label}",flush=True)
        checkpoint=run_training(config)
        results[label]=run_evaluation(config,checkpoint)
    comparison={"schema":1,"task":"settle_then_lift_left_3cm",
        "variants":{name:{"initial_action_std":STDS[name],"evaluation":result}
                    for name,result in results.items()},
        "interpretation":"高std若产生非零成功率，说明主要瓶颈是跨越离地接触门槛的探索。"}
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=args.output_root/"comparison.json"
    output.write_text(json.dumps(comparison,ensure_ascii=False,indent=2,
                                 allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"PASS","comparison":str(output.resolve())},
                     ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
