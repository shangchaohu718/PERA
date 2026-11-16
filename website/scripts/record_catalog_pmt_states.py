"""Capture actual matched policy states for white-stage replay; no learning."""
from pathlib import Path
import argparse
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)


def main():
    import numpy as np
    import torch
    from pmt_tasks import builder
    from experiments.pmt_dagger_v5.target_robot import configure_target_robot
    from experiments.pmt_dagger_v5.task import OptimizedPlacementTask
    from experiments.pmt_dagger_v2.runtime import load_public
    from experiments.pmt_dagger_v2.rollout import rollout
    from experiments.pmt_dagger_v1.common import configure_sparse_backend, tensor_module_digest
    from modumotion_g1.domain_moe import load_domain, set_active_domains, public_digest, set_trainable_domain

    # Match the original evaluation's bounded CPU parallelism. Large default
    # OpenMP pools severely slow the small per-step feature operations.
    torch.set_num_threads(2)
    torch.manual_seed(7)
    out = ROOT / 'website/.render-work/catalog31-white-v2/pmt_optimized_scene0'
    out.mkdir(parents=True, exist_ok=True)
    if all((out / mode / 'report.json').exists() for mode in ['on', 'off']):
        return
    original_builder = builder.build_env_cfg
    def configure(*values, **options):
        cfg, _ = configure_target_robot(original_builder(*values, **options))
        return cfg
    builder.build_env_cfg = configure
    try:
        task = OptimizedPlacementTask(1, clip_id=0, actor_reference='optimized', full_reference=True, video=False)
    finally:
        builder.build_env_cfg = original_builder
    print('Native task ready; loading frozen actor', flush=True)
    actor, _ = load_public()
    print('Frozen actor loaded; loading existing expert', flush=True)
    configure_sparse_backend(actor)
    plugin = ROOT / 'reports/pmt_target_recovery_long_v1/training/round_007.domain.pt'
    schema = {'kind': 'PMT_official_optimized_single_placement_v5', 'clip_id': 0,
              'actor_reference': 'official_optimized',
              'observable_context': 'frozen_height64_raw_command128_history128_estimated_velocity3_anchor3',
              'optimized_reference_in_context': False, 'teacher_action_in_context': False}
    domain = load_domain(actor, plugin, context_dim=326, context_schema=schema, encoder_sha256=tensor_module_digest(task.teacher))
    print('Existing expert loaded', flush=True)
    set_trainable_domain(actor, None)
    actor.eval()
    before = public_digest(actor)
    print('Frozen public digest verified', before, flush=True)
    def forbidden(*a, **kw):
        raise RuntimeError('No teacher action queries allowed')
    task.teacher_action = forbidden
    original = task.step
    initial = []
    for mode in ['on', 'off']:
        folder = out / mode
        folder.mkdir(exist_ok=True)
        if (folder / 'report.json').exists():
            continue
        set_active_domains(actor, [domain] if mode == 'on' else [])
        print('Starting native mode', mode, flush=True)
        states = []
        def step(action):
            data = task.robot.data
            states.append(torch.cat([data.root_pos_w[0], data.root_quat_w[0], data.joint_pos[0, task.to_holo]]).detach().cpu().numpy())
            if len(states) <= 3 or len(states) % 250 == 0:
                print('Capturing', mode, len(states), 'native steps', flush=True)
            return original(action)
        task.step = step
        try:
            with torch.no_grad():
                result, _ = rollout(task, actor, 7604)
        finally:
            task.step = original
        qpos = np.stack(states)
        initial.append(qpos[0].copy())
        np.savez_compressed(folder / 'physics.npz', qpos=qpos, fps=50,
                            joint_names=np.array(task.contract['joint_names']))
        assert result['teacher_action_queries'] == 0 and public_digest(actor) == before == actor.public_sha256
        result.update(expert_enabled=mode == 'on', public_sha256=before, seed=7604,
                      plugin=str(plugin.relative_to(ROOT)), plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                      terrain=str(Path(task.env.cfg.pmt_mesh_path).resolve()),
                      training_updates=0, teacher_queries=0,
                      scope='Actual native pre-step root and joint states. Stop at first native outcome; no reset poses, teacher actions or pose interpolation.')
        (folder / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print('Captured', mode, len(states), 'states;', result['mean_seconds'], 'native seconds', flush=True)
    if len(initial) == 2:
        assert np.max(np.abs(initial[0] - initial[1])) < 1e-6
    task.close()


try:
    main()
except BaseException:
    # Isaac Lab's stop callback can hang while closing a live failed scene.
    # Print the cause first, then exit this isolated recording process.
    import traceback
    import os
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    launcher.app.close()
