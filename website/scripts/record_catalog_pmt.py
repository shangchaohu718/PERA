"""Record the existing target-robot policy with its expert bypassed; no training."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args)


def main():
    import torch
    import imageio.v2 as imageio
    import numpy as np
    from pmt_tasks import builder
    from experiments.pmt_dagger_v5.target_robot import configure_target_robot
    from experiments.pmt_dagger_v5.task import OptimizedPlacementTask
    from experiments.pmt_dagger_v2.runtime import load_public
    from experiments.pmt_dagger_v2.rollout import rollout
    from experiments.pmt_dagger_v1.common import configure_sparse_backend,tensor_module_digest
    from modumotion_g1.domain_moe import load_domain,set_active_domains,public_digest,set_trainable_domain
    out=ROOT/'website/.render-work/catalog31-v1/pmt_optimized_scene0/off'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():return
    original_builder=builder.build_env_cfg
    def configure(*values,**options):
        cfg,_=configure_target_robot(original_builder(*values,**options))
        # Reuse the already-converted robot asset; no model or training updates.
        return cfg
    builder.build_env_cfg=configure
    try:task=OptimizedPlacementTask(1,clip_id=0,actor_reference='optimized',full_reference=True,video=True)
    finally:builder.build_env_cfg=original_builder
    actor,_=load_public()
    configure_sparse_backend(actor)
    plugin=ROOT/'reports/pmt_target_recovery_long_v1/training/round_007.domain.pt'
    schema={'kind':'PMT_official_optimized_single_placement_v5','clip_id':0,
            'actor_reference':'official_optimized','observable_context':'frozen_height64_raw_command128_history128_estimated_velocity3_anchor3',
            'optimized_reference_in_context':False,'teacher_action_in_context':False}
    load_domain(actor,plugin,context_dim=326,context_schema=schema,encoder_sha256=tensor_module_digest(task.teacher))
    set_trainable_domain(actor,None)
    set_active_domains(actor,[],{})
    actor.eval()
    before=public_digest(actor)
    original=task.step
    writer=imageio.get_writer(str(out/'native.mp4'),fps=25,codec='libx264',pixelformat='yuv420p')
    calls=0
    def step(action):
        nonlocal calls
        if calls%2==0:writer.append_data(np.asarray(task.env.render())[:,:,:3])
        calls+=1
        return original(action)
    def forbidden(*a,**kw):raise RuntimeError('No teacher action queries allowed')
    task.step=step
    task.teacher_action=forbidden
    try:
        with torch.no_grad():result,_=rollout(task,actor,7604)
    finally:
        writer.close()
        task.step=original
    assert result['teacher_action_queries']==0 and public_digest(actor)==before==actor.public_sha256
    result.update(expert_enabled=False,public_sha256=before,seed=7604,
                  plugin=str(plugin.relative_to(ROOT)),plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                  training_updates=0,teacher_queries=0,
                  scope='First native episode only; stop at its actual outcome; no replay of reset states.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    task.close()
    print('Recorded expert OFF, native seconds',result['mean_seconds'],flush=True)


try:main()
finally:launcher.app.close()
