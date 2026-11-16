"""Record legacy HoloMotion ON/OFF task executions without training or resets.

All new outputs live under website/.render-work/catalog31-v1; original reports,
checkpoints and task implementations are read-only.
"""
import json
from pathlib import Path
import sys
import hashlib
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import mujoco
from modumotion_g1.official import load_official
from modumotion_g1.domain_moe import (
    install_domains, load_domain, set_active_domains, public_digest, set_domain_context,
)
from modumotion_g1.instinct_dagger import NativeTeacherTask
from modumotion_g1.perceptive_supermimic import tensor_module_digest
from scripts.instinct_dagger import TEACHER

WORK = ROOT/'website/.render-work/catalog31-v1'


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    roster = json.loads((ROOT/'configs/g1_competitive_suite31_v1.json').read_text())['tasks']
    tasks = [t for t in roster if t['family']=='instinct' and t['id'] not in
             ['diveroll_lab_no_wall', 'roll_vault_no_wall']]
    actor, critic, _ = load_official('cuda:0')
    del critic
    install_domains(actor)
    actor.eval()
    before = public_digest(actor)
    for item in tasks:
        name = item['id']
        if all((WORK/name/mode/'report.json').exists() for mode in ['on','off']):
            continue
        folder = ROOT/'data/instinct_climb_dagger_v1' if name=='knee_climb_no_wall' else ROOT/'data/instinct_task_dagger_v1'/name
        task = NativeTeacherTask(folder, TEACHER, 1, 'cuda:0', False)
        context = {'context_dim':32, 'context_schema':{
            'kind':'instinct_released_depth_features','depth_shape':[1,18,32],
            'normalization':'released_policy_normalizer','frozen':True},
            'encoder_sha256':tensor_module_digest(task.depth)}
        plugin = ROOT/item['legacy_evidence_plugin']
        domain = load_domain(actor, plugin, **context)
        original_step = task.vec.step
        starts = []
        for enabled in [True,False]:
            out = WORK/name/('on' if enabled else 'off')
            if (out/'report.json').exists():
                starts.append(np.load(out/'physics.npz')['qpos'][0])
                continue
            out.mkdir(parents=True, exist_ok=True)
            set_active_domains(actor,[domain] if enabled else [],{domain:0.} if enabled else {})
            poses = []
            calls = 0
            def step(action):
                nonlocal calls
                if calls%2==0:
                    poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
                calls += 1
                return original_step(action)
            task.vec.step = step
            result, _, _ = task.rollout(actor=actor, controller='student', seed=7002)
            assert result['teacher_queries']==0
            native_end = result['episode_seconds'][0]
            continuation = 0
            if not enabled and result['successful_episodes']==0:
                obs = task.vec.get_observations()[0]
                for _ in range(round(4/task.env.step_dt)):
                    task.env._manual_reset_pending.zero_()
                    td, z, _ = task.student_inputs(obs)
                    set_domain_context(actor,z)
                    try:
                        action = task.to_native(actor(td,mode='inference')['actions'])
                    finally:
                        set_domain_context(actor,None)
                    assert torch.isfinite(action).all()
                    obs, _, _, _ = task.vec.step(action)
                    task.last_action.copy_(task.to_holo(action))
                continuation = 4
            poses = np.asarray(poses)
            assert np.isfinite(poses).all()
            starts.append(poses[0])
            np.savez_compressed(out/'physics.npz',qpos=poses,fps=25.)
            mujoco.mj_saveModel(task.env.sim.mj_model,str(out/'scene.mjb'))
            result.update(task=name, expert_enabled=enabled, seed=7002,
                          plugin=str(plugin.relative_to(ROOT)),
                          plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                          public_sha256=before, training_updates=0,
                          visualization={'native_termination_s':native_end,
                                         'post_termination_seconds':continuation,
                                         'resets_after_initialization':0,
                                         'evaluation_result_unchanged':True})
            (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
            print(name, 'ON' if enabled else 'OFF', 'native_s',native_end,
                  'success',result['successful_episodes'],'continuation',continuation,flush=True)
        error = float(np.max(np.abs(starts[0]-starts[1])))
        assert error<1e-7, (name,error)
        (WORK/name/'pair.json').write_text(json.dumps({'initial_qpos_max_difference':error,
            'public_sha256':before,'seed':7002,'training_updates':0,'teacher_queries':0},indent=2)+'\n')
        task.vec.step = original_step
        task.close()
        assert public_digest(actor)==before==actor.public_sha256
    print('Completed legacy Instinct pairs; frozen Actor unchanged.',flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__=='__main__':
    main()
