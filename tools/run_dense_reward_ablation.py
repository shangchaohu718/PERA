"""K1 3 cm抬脚：分层筛选HoloMotion式目标驱动物理奖励。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from unified_motion.experiment import load_config,run_evaluation,run_training
from tools.run_stage_learnability import configured as stage_configured


WEIGHTS={
    "baseline": {},
    "support_only": {"support_weight": .3},
    "posture_com": {"upright_weight": .2,"com_transfer_weight": .2},
    "balance_core": {"support_weight": .3,"upright_weight": .2,"com_transfer_weight": .2},
    "dense_full": {"support_weight": .3,"upright_weight": .2,"com_transfer_weight": .2,
                   "action_rate_weight": .02,"joint_limit_weight": .1,
                   "undesired_contact_weight": .1,"root_ang_vel_weight": .05,
                   "foot_slide_weight": .05},
}


def configured(base:dict,label:str,args:argparse.Namespace)->dict:
    config=stage_configured(deepcopy(base),"lift_03m",args)
    reward={key:0.0 for key in ("support_weight","upright_weight","com_transfer_weight",
        "action_rate_weight","joint_limit_weight","undesired_contact_weight",
        "root_ang_vel_weight","foot_slide_weight")}
    reward.update({"support_force_scale_n":25.0,"upright_sigma_rad":.35,"com_sigma_m":.08})
    reward.update(WEIGHTS[label]);config["task_reward"]=reward
    config["plan"][1]["unload_reward_weight"]=.25
    config["plan"][1]["unload_force_scale_n"]=25.0
    config["output"]=str((args.output_root/label).resolve())
    return config


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/k1_lateral_step.yaml"))
    parser.add_argument("--asset-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=Path("runs/k1_dense_reward_ablation"))
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--num-envs",type=int,default=64)
    parser.add_argument("--updates",type=int,default=75)
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
    base=load_config(args.config);results={}
    for label in WEIGHTS:
        config=configured(base,label,args)
        print(f"训练密集奖励组 {label}",flush=True)
        checkpoint=Path(config["output"])/"checkpoint.pt"
        if not checkpoint.is_file(): checkpoint=run_training(config)
        results[label]=run_evaluation(config,checkpoint)
    comparison={"schema":1,"task":"settle_then_lift_left_3cm",
        "unload_reward_weight":.25,"variants":{
            name:{"task_reward_weights":WEIGHTS[name],"evaluation":results[name]}
            for name in WEIGHTS},
        "interpretation":"依次判断承重、姿态/质心、三项核心及安全正则是否跨过离地门槛。"}
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=args.output_root/"comparison.json"
    output.write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"PASS","comparison":str(output.resolve())},ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
