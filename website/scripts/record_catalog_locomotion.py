"""Capture one matched legacy-policy episode per mode, never train or auto-reset."""
import json
import hashlib
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
import mujoco
from modumotion_g1.official import load_official
from modumotion_g1.domain_moe import install_domains,load_domain,set_active_domains,public_digest,set_domain_context
from modumotion_g1.parkour_distillation import ParkourTask
from modumotion_g1.parkour_remaining import RemainingParkourTask,configure_hfield_capacity
from modumotion_g1.parkour_extra import ExtraParkourTask
from modumotion_g1.parkour_remaining_rollout import actor_action,endings

WORK=ROOT/'website/.render-work/catalog31-v1'


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    configure_hfield_capacity(256)
    actor,critic,_=load_official('cuda:0')
    del critic
    install_domains(actor)
    actor.eval()
    before=public_digest(actor)
    roster=json.loads((ROOT/'configs/g1_competitive_suite31_v1.json').read_text())['tasks']
    names=['parkour_stairs_up','parkour_stairs_down','parkour_stairs_down_high',
           'parkour_slope','parkour_rough','parkour_obstacles','velocity_walk']
    for name in names:
        if all((WORK/name/m/'report.json').exists() for m in ['on','off']):continue
        item=next(t for t in roster if t['id']==name)
        plugin=ROOT/item['legacy_evidence_plugin']
        velocity=name=='velocity_walk'
        if velocity:
            from modumotion_g1.velocity_walking import VelocityWalkTask,context_contract
            from scripts.velocity_walk_dagger import evaluate
            task=VelocityWalkTask(1,'cuda:0',False)
            contract=context_contract()
            original=task.step_target
        else:
            task_id=name.removeprefix('parkour_')
            cls=(RemainingParkourTask if task_id in ['stairs_up','obstacles'] else
                 ExtraParkourTask if task_id=='stairs_down_high' else ParkourTask)
            task=cls(task_id,1,'cuda:0')
            if not hasattr(task,'overflow_flags'):
                task.overflow_flags=lambda:task.env.sim.data.overflow.clone().to(torch.int64)
            contract=task.contract()
            original=task.step
        domain=load_domain(actor,plugin,**contract)
        starts=[]
        for enabled in [True,False]:
            out=WORK/name/('on' if enabled else 'off')
            if (out/'report.json').exists():
                starts.append(np.load(out/'physics.npz')['qpos'][0]);continue
            out.mkdir(parents=True,exist_ok=True)
            set_active_domains(actor,[domain] if enabled else [],{domain:0.} if enabled else {})
            poses=[]
            calls=0
            def step(action):
                nonlocal calls
                if calls%2==0:poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
                calls+=1
                return original(action)
            if velocity:
                task.step_target=step
                result=evaluate(task,actor,seed=8101)
                native_end=float(task.elapsed[0])
                physical_failure=result['failed_at'][0] is not None
            else:
                task.step=step
                task.reset(8101,actor)
                near=0
                goal_dwell=False
                physical_failure=False
                for _ in range(min(round(20/task.env.step_dt),task.env.max_episode_length)):
                    mu,_=actor_action(task,actor)
                    done=task.step(task.holo_to_native(mu))
                    p,tilt,distance,fired,failed,overflow=endings(task)
                    assert not overflow.any(),'Native collision workspace overflow'
                    near=near+1 if float(distance[0])<=.4 and not bool(failed[0]) else 0
                    goal_dwell|=near>=round(1/task.env.step_dt)
                    if bool(done[0]) or bool(failed[0]):
                        physical_failure=bool(failed[0]);break
                native_end=float(task.elapsed[0])
                result={'physical_failure':physical_failure,'goal_within_40cm_for_one_second':goal_dwell,
                        'final_goal_distance_m':float(distance[0]),
                        'fired':{k:bool(v[0]) for k,v in fired.items()},
                        'outcome_scope':'One illustrative native episode; historical batch metrics are reported separately.'}
            continuation=4 if not enabled and physical_failure else 0
            for _ in range(round(continuation/task.env.step_dt)):
                task.env._manual_reset_pending.zero_()
                if velocity:
                    td,context=task.student_inputs()
                    set_domain_context(actor,{domain:context})
                    try:mu=actor(td,mode='inference')['actions']
                    finally:set_domain_context(actor,None)
                    target=task.action.offset+task.action.scale*task.adapter.action_to_sim(mu,task.action.offset,task.action.scale)
                    task.step_target(target)
                else:
                    mu,_=actor_action(task,actor)
                    task.step(task.holo_to_native(mu))
                    assert not task.overflow_flags().any()
            poses=np.asarray(poses)
            assert len(poses)>0 and np.isfinite(poses).all()
            starts.append(poses[0])
            np.savez_compressed(out/'physics.npz',qpos=poses,fps=25.)
            mujoco.mj_saveModel(task.env.sim.mj_model if velocity else task.cpu_model,str(out/'scene.mjb'))
            result.update(task=name,expert_enabled=enabled,seed=8101,public_sha256=before,
                          plugin=str(plugin.relative_to(ROOT)),plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                          teacher_queries=0,training_updates=0,
                          visualization={'native_termination_s':native_end,'post_termination_seconds':continuation,
                                         'resets_after_initialization':0,'evaluation_result_unchanged':True})
            (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
            print(name,'ON' if enabled else 'OFF','native_s',native_end,
                  'physical_failure',physical_failure,'continuation',continuation,flush=True)
        error=float(np.max(np.abs(starts[0]-starts[1])))
        assert error<1e-7,(name,error)
        (WORK/name/'pair.json').write_text(json.dumps({'initial_qpos_max_difference':error,
            'seed':8101,'public_sha256':before,'training_updates':0,'teacher_queries':0},indent=2)+'\n')
        if velocity:task.step_target=original
        else:task.step=original
        task.env.close()
        assert public_digest(actor)==before==actor.public_sha256
    print('Locomotion recordings complete; public weights unchanged.',flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__=='__main__':main()
