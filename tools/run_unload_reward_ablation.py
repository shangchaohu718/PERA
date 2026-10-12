"""固定 3 cm 抬脚和 std=0.35，比较连续左脚卸载力奖励权重。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from unified_motion.experiment import load_config,run_evaluation,run_training
from tools.run_stage_learnability import configured as stage_configured


WEIGHTS={"unload_000":0.0,"unload_010":0.10,"unload_025":0.25,"unload_050":0.50}


def configured(base:dict,label:str,args:argparse.Namespace)->dict:
    config=stage_configured(base,"lift_03m",args)
    config["model"]["init_std"]=0.35
    lift=config["plan"][1]
    lift["unload_reward_weight"]=WEIGHTS[label]
    lift["unload_force_scale_n"]=25.0
    config["output"]=str((args.output_root/label).resolve())
    return config


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/k1_lateral_step.yaml"))
    parser.add_argument("--asset-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=Path("runs/k1_unload_reward_ablation"))
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
    configs={name:configured(base,name,args) for name in WEIGHTS}
    results={}
    for label,config in configs.items():
        print(f"训练连续卸载奖励组 {label}",flush=True)
        checkpoint=run_training(config)
        results[label]=run_evaluation(config,checkpoint)
    comparison={"schema":1,"task":"settle_then_lift_left_3cm","initial_action_std":.35,
        "force_scale_n":25.,
        "variants":{name:{"unload_reward_weight":WEIGHTS[name],"evaluation":result}
                    for name,result in results.items()},
        "interpretation":"非零权重若产生冻结成功，说明连续法向载荷卸载信号能跨过接触门槛。"}
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=args.output_root/"comparison.json"
    output.write_text(json.dumps(comparison,ensure_ascii=False,indent=2,
                                 allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"PASS","comparison":str(output.resolve())},
                     ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
