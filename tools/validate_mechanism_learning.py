"""CPU learning probes on constructed data, never a robot/simulator benchmark.

Run from any directory: python tools/validate_mechanism_learning.py --output <json>.
The same public modules as training are used; no API calls or test fixtures.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from unified_motion.holomotion_physics_actor import HoloMotionPhysicsActorConfig, PhysicsActionResidual
from unified_motion.physics_confidence import PhysicsConfidence, PhysicsConfidenceConfig
from unified_motion.privileged_teacher import (PrivilegedTeacherConfig, PrivilegedTransformerTeacher,
    teacher_supervision, assess_teacher)
from unified_motion.robot_contract import get_robot_contract


def residual_probe(seed):
    torch.manual_seed(seed)
    features = torch.randn(512, 6)
    target = .02*torch.tanh(features[:, :1]-features[:, 1:2])
    rows = {}
    for mode in ('calibrated', 'bounded'):
        torch.manual_seed(seed+100)
        model = PhysicsActionResidual(HoloMotionPhysicsActorConfig(
            physics_feature_dim=6, hidden_dim=16, action_dim=1))
        gate = PhysicsConfidence(2, PhysicsConfidenceConfig(mode=mode))
        confidence = gate(torch.zeros(512, 2), torch.zeros(512, dtype=torch.long), 128)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        base = torch.zeros_like(target)
        for _ in range(150):
            loss = ((confidence[:384]*model(features[:384], base[:384])-target[:384])**2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            predictions = confidence[384:]*model(features[384:], base[384:])
            rows[mode] = dict(gate=float(confidence[0]),
                heldout_mse=float((predictions-target[384:]).square().mean()),
                maximum_abs_correction=float(predictions.abs().max()))
    rows['mse_reduction_fraction'] = 1-rows['bounded']['heldout_mse']/rows['calibrated']['heldout_mse']
    return rows


def teacher_probe(seed):
    torch.manual_seed(seed)
    robot = get_robot_contract('k1_22dof')
    config = PrivilegedTeacherConfig(critic_dim=8, object_slots=0, width=32,
        heads=2, layers=1, context_steps=4, learning_rate=.003)
    teacher = PrivilegedTransformerTeacher(robot, 8, 2, config)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=config.learning_rate)

    def sample(batch):
        gain = .2+1.4*torch.rand(batch, 4, 1)
        actions = torch.zeros(batch, 4, 22)
        actions[..., :1] = 2*torch.rand(batch, 4, 1)-1
        inputs = torch.zeros(batch, 4, config.input_dim(22, 2))
        inputs[..., 0] = gain.squeeze(-1); inputs[..., 1] = 1.
        current = torch.zeros(batch, 4, 53)
        current[..., 8] = -1.
        starts = torch.zeros(batch, 4, dtype=torch.bool); starts[:, 0] = True
        truth = (.2*gain*actions[..., :1]).expand_as(current).clone()
        gravity = torch.cat((.2*gain*actions[..., :1], torch.zeros_like(gain), -torch.ones_like(gain)), -1)
        truth[..., 6:9] = torch.nn.functional.normalize(gravity, dim=-1)
        contacts = torch.zeros(batch, 4, 2)
        return inputs, starts, actions, current, truth, contacts

    test = sample(128)
    def evaluate():
        inputs, starts, actions, current, truth, contacts = test
        with torch.no_grad():
            out = teacher(inputs, starts, actions, current)
            shuffled = teacher.decode(out['latent'], actions.roll(1, 0), current)
            # Mean-gain baseline; gravity lies on the same unit sphere.
            delta = .2*.9*actions[..., :1]
            student = delta.expand_as(truth).clone()
            gravity = torch.cat((delta, torch.zeros_like(delta), -torch.ones_like(delta)), -1)
            student[..., 6:9] = torch.nn.functional.normalize(gravity, dim=-1)
            admitted, diagnostics = assess_teacher(out, student, contacts, shuffled,
                truth, current, torch.ones_like(truth, dtype=torch.bool), contacts, contacts,
                torch.zeros_like(contacts, dtype=torch.bool), config)
        return dict(core_mse=float((out['next_core']-truth).square().mean()),
            persistence_mse=float((current-truth).square().mean()),
            hidden_gain_student_mse=float((student-truth).square().mean()),
            shuffled_action_mse=float((shuffled['next_core']-truth).square().mean()),
            admitted_groups=admitted.tolist(), diagnostics=diagnostics)

    before = evaluate()
    for _ in range(500):
        inputs, starts, actions, current, truth, contacts = sample(16)
        output = teacher(inputs, starts, actions, current)
        loss = teacher_supervision(output, truth, torch.ones_like(truth, dtype=torch.bool),
            contacts, torch.zeros_like(contacts, dtype=torch.bool), torch.zeros(16, 4, 2),
            torch.ones(16, 4, dtype=torch.bool), config)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.)
        optimizer.step()
    return dict(config=asdict(config), initial=before, trained=evaluate())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    rows = [dict(seed=seed, residual=residual_probe(seed), teacher=teacher_probe(seed)) for seed in (7, 19, 42)]
    report = dict(scope='Constructed regression and action-conditioned linear-system probes; NOT robot physics.',
        torch=str(torch.__version__), device='cpu', seeds=rows, elapsed_seconds=time.perf_counter()-start,
        teacher_equation='d = 0.2 * gain * action[0], gain uniform [0.2,1.6]; non-gravity core = d; gravity = unit([d,0,-1])',
        caveat='Teacher has gain as privileged input; an uninformed student cannot recover it. No sim2real claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for row in rows:
        print(json.dumps(dict(seed=row['seed'], residual=row['residual'],
            teacher_initial=row['teacher']['initial']['core_mse'],
            teacher_trained=row['teacher']['trained']['core_mse'],
            teacher_admitted=row['teacher']['trained']['admitted_groups'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
