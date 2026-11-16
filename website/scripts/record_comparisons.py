#!/usr/bin/env python3
"""Record frozen on/off executions; never construct an optimizer or train."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import torch
from modumotion_g1.official import load_official
from modumotion_g1.domain_moe import install_domains, load_domain, set_active_domains, public_digest

OUT=ROOT/'website/.render-work/comparisons-v5'

def save(path, result):
    path.write_text(json.dumps(result,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('skill',choices=['roll','vault','gaps','steps','skate']);a=p.parse_args()
    torch.set_num_threads(2)
    actor,critic,_=load_official('cuda:0');del critic
    install_domains(actor);before=public_digest(actor);actor.eval()
    if a.skill in ('roll','vault'):
        from modumotion_g1.instinct_dagger import NativeTeacherTask
        from modumotion_g1.perceptive_supermimic import tensor_module_digest
        from scripts.instinct_dagger import TEACHER
        task_name='diveroll_lab_no_wall' if a.skill=='roll' else 'roll_vault_no_wall'
        task=NativeTeacherTask(ROOT/'data/instinct_task_dagger_v1'/task_name,TEACHER,1,'cuda:0',False)
        plugin=ROOT/'reports/instinct_all_tasks_dagger_v2'/task_name/'run/training/round_020.domain.pt'
        context={'context_dim':32,'context_schema':{'kind':'instinct_released_depth_features','depth_shape':[1,18,32],
                 'normalization':'released_policy_normalizer','frozen':True},'encoder_sha256':tensor_module_digest(task.depth)}
        domain=load_domain(actor,plugin,**context)
        original_step=task.vec.step
        for enabled in (True,False):
            out=OUT/(a.skill+('-on' if enabled else '-off'))
            if (out/'report.json').exists():continue
            if out.exists():out.rmdir()  # Empty directory from an interrupted start only.
            out.mkdir(parents=True,exist_ok=False)
            set_active_domains(actor,[domain] if enabled else [],{domain:0.} if enabled else {})
            poses=[];calls=0
            def step(action):
                nonlocal calls
                if calls%2==0:poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
                calls+=1
                return original_step(action)
            task.vec.step=step
            result,_,_=task.rollout(actor=actor,controller='student',seed=7002)
            assert result['teacher_queries']==0
            np.savez_compressed(out/'physics.npz',qpos=poses,fps=25.)
            mujoco.mj_saveModel(task.env.sim.mj_model,str(out/'scene.mjb'))
            result.update(expert_enabled=enabled,training_updates=0,public_sha256=before,
                          source_task=task_name,plugin=str(plugin.relative_to(ROOT)),seed=7002,
                          source='Fresh standalone native task evaluation; same reset and reference for ON/OFF')
            save(out/'report.json',result)
            print(a.skill,enabled,result['episode_seconds'],flush=True)
        task.close()
    elif a.skill in ('gaps','steps'):
        from modumotion_g1.parkour_remaining import RemainingParkourTask
        from modumotion_g1.parkour_remaining_rollout import rollout
        from modumotion_g1.parkour_extra import ExtraParkourTask,rollout as extra_rollout
        task_name='gaps' if a.skill=='gaps' else 'stairs_up_high';seed=51001 if a.skill=='gaps' else 93001
        cls=RemainingParkourTask if a.skill=='gaps' else ExtraParkourTask
        task=cls(task_name,1,'cuda:0')
        run=ROOT/('reports/instinct_parkour_remaining_v3' if a.skill=='gaps' else 'reports/instinct_extra_parkour_v1')/task_name
        domain=load_domain(actor,run/'training/round_020.domain.pt',**task.contract())
        for enabled in (True,False):
            out=OUT/(a.skill+('-on' if enabled else '-off'))
            if (out/'report.json').exists():continue
            out.mkdir(parents=True,exist_ok=False)
            set_active_domains(actor,[domain] if enabled else [],{domain:0.} if enabled else {})
            with torch.no_grad():result,_=(rollout if a.skill=='gaps' else extra_rollout)(task,actor,seed=seed,seconds=10.,recording=out/'recording')
            result.update(expert_enabled=enabled,training_updates=0,public_sha256=before,source_task=task_name,seed=seed,
                          controller='frozen_HoloMotion_public_plus_expert' if enabled else 'frozen_HoloMotion_public_expert_off',teacher_queries=0)
            save(out/'report.json',result);print(a.skill,enabled,result['episodes'][0]['elapsed_s'],flush=True)
    else:
        from modumotion_g1.husky_skateboarding import HuskySkateTask,context_contract,DOMAIN
        from scripts.husky_skateboard_dagger import evaluate
        task=HuskySkateTask(1,'cuda:0')
        plugin=ROOT/'reports/husky_skateboard_dagger_v2/training/round_020.domain.pt'
        load_domain(actor,plugin,**context_contract())
        for enabled in (True,False):
            out=OUT/('skate'+('-on' if enabled else '-off'))
            result=evaluate(task,actor,seconds=30,seed=10102,turn=True,output=out,expert_enabled=enabled)
            result.update(public_sha256=before,training_updates=0,teacher_queries=0,
                          plugin=str(plugin.relative_to(ROOT)))
            save(out/'report.json',result)
            print('skate',enabled,result['episode_seconds'],flush=True)
    assert public_digest(actor)==before==actor.public_sha256
    print('Frozen public weights unchanged; no training.',flush=True)

if __name__=='__main__':main()
