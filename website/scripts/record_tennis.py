#!/usr/bin/env python3
"""Record the existing frozen tennis student, without training or teacher actions.

Visual hands are added only to the separate replay model, after evaluation.
"""
import os
from pathlib import Path
import sys
import json
import hashlib
import inspect
import argparse

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'website'
OUT = WEB / '.render-work/tennis-v3'
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/athlete_tennis_v1'),
               str(ROOT/'third_party/ATHLETE/athlete/src')]
import distill as d
import mujoco
import numpy as np
import torch
import athlete.scripts.play_student_onnx as native

RUN = ROOT/'reports/athlete_tennis_v1/body_window_continue_10k_v1'
PLUGIN = RUN/'round_10.domain.pt'

def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

class Recorder:
    def __init__(self, sim, path):
        self.sim = sim
        self.poses, self.times = [], []
        mujoco.mj_saveModel(sim.model, str(OUT/'original-scene.mjb'))

    def frame(self, t, weight, controller):
        assert controller == 'student'
        self.poses.append(self.sim.data.qpos.copy())
        self.times.append(t)

    def close(self):
        np.savez_compressed(OUT/'physics.npz', qpos=self.poses, time_s=self.times, fps=25.)

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--expert',choices=['on','off'],default='on');p.add_argument('--seconds',type=float,default=40.)
    args=p.parse_args();OUT=args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(97319)
    normalization_path = RUN/'context_normalization.json'
    normalization = json.loads(normalization_path.read_text())
    payload = torch.load(PLUGIN, map_location='cpu', weights_only=True)
    context = payload['context']
    assert context['encoder_sha256'] == digest(normalization_path)
    print('Loading frozen public actor and selected tennis plugin', flush=True)
    actor, critic, _ = d.load_official(device='cpu')
    del critic
    d.load_domain(actor, PLUGIN, context_dim=context['dim'], context_schema=context['schema'],
                  encoder_sha256=context['encoder_sha256'])
    d.set_trainable_domain(actor, d.DOMAIN)  # Required domain selector; rollout is no_grad.
    actor.eval()
    before = d.public_digest(actor)
    d.Video = Recorder
    # MjSpec models do not support mj_saveLastXML. Export the very same spec
    # immediately after its final compile, with no change to model construction.
    source = inspect.getsource(native._build_standalone_model)
    assert source.count('  model = spec.compile()\n  return model') == 1
    capture = source.replace('  model = spec.compile()\n  return model',
        '  model = spec.compile()\n  _capture_path.write_text(spec.to_xml())\n  return model')
    scope = dict(native.__dict__, _capture_path=OUT/'original-scene.xml')
    exec(compile(capture, '<read-only tennis spec export>', 'exec'), scope)
    native._build_standalone_model = scope['_build_standalone_model']
    result, _ = d.Runner(actor, 'cpu', normalization).rollout(97001, args.seconds, collect=False,
                   video=OUT/'capture',public_only=args.expert=='off',stop_on_fall=args.expert=='off')
    assert result['teacher_queries'] == result['teacher_executed_steps'] == 0
    assert d.public_digest(actor) == before == actor.public_sha256
    result.update(plugin=str(PLUGIN.relative_to(ROOT)), plugin_sha256=digest(PLUGIN),
                  normalization_sha256=digest(normalization_path), public_actor_sha256=before,
                  training_updates=0, expert_enabled=args.expert=='on',visual_hands_during_simulation=False,
                  recorded_first_frame_s=.02, source_report=str((RUN/'report.json').relative_to(ROOT)))
    (OUT/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
