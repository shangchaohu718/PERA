#!/usr/bin/env python3
"""Keep official failure results; record four more seconds with the same policy, without reset.

The continuation is visualization only and never changes native evaluation scores.
No optimizer, teacher queries, fabricated poses or replacement controller.
"""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from modumotion_g1.official import load_official
from modumotion_g1.domain_moe import install_domains,set_active_domains,public_digest,set_domain_context,load_domain

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('skill',choices=['roll','vault','gaps','steps']);a=p.parse_args()
    torch.set_num_threads(2)
    out=ROOT/'website/.render-work/comparisons-v6'/(a.skill+'-off')
    out.mkdir(parents=True,exist_ok=False)
    actor,critic,_=load_official('cuda:0');del critic
    install_domains(actor);actor.eval();before=public_digest(actor);set_active_domains(actor,[],{})
    old=ROOT/'website/.render-work/comparisons-v5'/(a.skill+'-off')
    report=json.loads((old/'report.json').read_text())
    poses=[];calls=0
    if a.skill in ('roll','vault'):
        from modumotion_g1.instinct_dagger import NativeTeacherTask
        from scripts.instinct_dagger import TEACHER
        task=NativeTeacherTask(ROOT/'data/instinct_task_dagger_v1'/report['source_task'],TEACHER,1,'cuda:0',False)
        from modumotion_g1.perceptive_supermimic import tensor_module_digest
        context={'context_dim':32,'context_schema':{'kind':'instinct_released_depth_features','depth_shape':[1,18,32],'normalization':'released_policy_normalizer','frozen':True},'encoder_sha256':tensor_module_digest(task.depth)}
        domain=load_domain(actor,ROOT/report['plugin'],**context)
        set_active_domains(actor,[domain],{domain:0.})
        task.rollout(actor=actor,controller='student',seed=7002)
        set_active_domains(actor,[],{})
        original=task.vec.step
        def step(action):
            nonlocal calls
            if calls%2==0:poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
            calls+=1
            return original(action)
        task.vec.step=step
        result,_,_=task.rollout(actor=actor,controller='student',seed=7002)
        native_end=result['episode_seconds'][0]
        print('Native end',native_end,'previous',report['episode_seconds'][0],flush=True)
        report.update(result)
        obs=task.vec.get_observations()[0]
        for _ in range(round(4/task.env.step_dt)):
            # Diagnostic continuation only: retain all state and termination terms.
            # Clear the step guard, never call reset or enable auto-reset.
            task.env._manual_reset_pending.zero_()
            td,z,_=task.student_inputs(obs)
            set_domain_context(actor,z)
            try: action=task.to_native(actor(td,mode='inference')['actions'])
            finally:set_domain_context(actor,None)
            assert torch.isfinite(action).all()
            obs,_,_,_=task.vec.step(action)
            task.last_action.copy_(task.to_holo(action))
            assert np.isfinite(task.env.sim.data.qpos[0].cpu().numpy()).all()
        model=task.env.sim.mj_model
    else:
        from modumotion_g1.parkour_remaining import RemainingParkourTask
        from modumotion_g1.parkour_extra import ExtraParkourTask
        from modumotion_g1.parkour_remaining_rollout import episode,actor_action
        task=(RemainingParkourTask if a.skill=='gaps' else ExtraParkourTask)(report['source_task'],1,'cuda:0')
        run=ROOT/('reports/instinct_parkour_remaining_v3' if a.skill=='gaps' else 'reports/instinct_extra_parkour_v1')/report['source_task']
        domain=load_domain(actor,run/'training/round_020.domain.pt',**task.contract())
        set_active_domains(actor,[domain],{domain:0.})
        from modumotion_g1.parkour_remaining_rollout import rollout
        from modumotion_g1.parkour_extra import rollout as extra_rollout
        (rollout if a.skill=='gaps' else extra_rollout)(task,actor,seed=report['seed'],seconds=10.)
        set_active_domains(actor,[],{})
        original=task.step
        def step(action):
            nonlocal calls
            if calls%2==0:poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
            calls+=1
            return original(action)
        task.step=step
        result,_,_=episode(task,actor,report['seed'],10.)
        native_end=result['elapsed_s']
        print('Native end',native_end,'previous',report['episodes'][0]['elapsed_s'],flush=True)
        report['prior_four_seed_evaluation']=report['episodes']
        report['episodes']=[result];report['num_episodes']=1;report['successes']=int(result['success']);report['acceptance_passed']=bool(result['success'])
        for _ in range(round(4/task.env.step_dt)):
            task.env._manual_reset_pending.zero_()
            mu,_=actor_action(task,actor)
            task.step(task.holo_to_native(mu))
            assert not task.overflow_flags().any(),'Numerical overflow in visualization'
        model=task.cpu_model
    suffix='recording/physics.npz' if a.skill in ('gaps','steps') else 'physics.npz'
    previous=np.load(old/suffix)['qpos']
    error=float(np.max(abs(previous[:min(len(previous),len(poses))]-np.asarray(poses[:min(len(previous),len(poses))]))))
    assert np.max(abs(previous[0]-np.asarray(poses[0])))<1e-7,'Different initial pose'
    print('Prior recording prefix difference',error,flush=True)
    mujoco.mj_saveModel(model,str(out/'scene.mjb'))
    np.savez_compressed(out/'physics.npz',qpos=np.asarray(poses),fps=25.)
    assert public_digest(actor)==before==actor.public_sha256==report['public_sha256']
    report['visualization']={'native_termination_s':native_end,'post_termination_seconds':4.,
        'recording_seconds':len(poses)/25.,'policy':'same frozen public policy, expert bypassed throughout',
        'resets':0,'teacher_queries':0,'training_updates':0,'native_prefix_max_qpos_difference':error,
        'manual_step_guard_cleared_after_native_termination':True,'native_termination_terms_preserved':True,'evaluation_result_unchanged':True,'scope':'After native termination, continue the identical public controller for four seconds solely to show the physical fall and aftermath. This is outside the scored evaluation.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(a.skill,json.dumps(report['visualization']),flush=True)
    # Avoid third-party Warp environment finalizer races at interpreter shutdown.
    sys.stdout.flush()
    import os
    os._exit(0)

if __name__=='__main__':main()
