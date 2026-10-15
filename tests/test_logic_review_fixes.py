"""Regression counterexamples for initialization, geometry and trial admission."""
from types import SimpleNamespace

import pytest
import torch

from test_holomotion_rollout_boundaries import _real_trainer
from test_training_advisor import trial_reports
from unified_motion.holomotion_physics_training import HoloMotionPhysicsMJLabPPO
from unified_motion.k1_training import create_k1_trainer
from unified_motion.models.physics import DynamicsModel
from unified_motion.physics_update import PhysicsUpdateConfig
from unified_motion.privileged_teacher import PrivilegedTeacherConfig, PrivilegedTransformerTeacher
from unified_motion.robot_contract import get_robot_contract
from unified_motion.llm.validation import TRACKING_METRICS, compare_trials


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('pretrained,override,expected', [
    (False, None, False), (True, None, True), (True, False, False), (False, True, True)])
def test_actor_freezing_follows_weight_provenance_and_explicit_override(monkeypatch, direct, pretrained, override, expected):
    source = _real_trainer(False, monkeypatch)
    algorithm = SimpleNamespace(actor=source.actor, critic=source.critic,
        network_config=source._network_config, pretrained=pretrained)
    options = dict(physics_width=4, memory_width=4, latent_width=2, hidden_dim=8)
    update = None if override is None else PhysicsUpdateConfig(freeze_base_actor=override)
    if direct:
        dynamics = DynamicsModel(source.task.bridge.robot, system_id_dim=2, uncertainty=True, **options)
        trainer = HoloMotionPhysicsMJLabPPO(algorithm, source.task, dynamics,
            config=source.config, update_config=update)
    else:
        trainer = create_k1_trainer(algorithm, source.task, physics=True,
            ppo_config=source.config, dynamics_config=options, update_config=update)
    assert trainer.update_config.freeze_base_actor is expected
    assert all(p.requires_grad is not expected for p in trainer.actor.base_actor.parameters())
    optimized = {id(p) for group in trainer.actor_optimizer.param_groups for p in group['params']}
    assert all((id(p) in optimized) is not expected for p in trainer.actor.base_actor.parameters())


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('current_gravity,delta', [
    ([0., 0., -1.], [2., 0., 0.]), ([0., 0., -1.], [0., 0., 1.]),
    ([0., 0., 0.], [0., 0., 0.])])
def test_teacher_gravity_geometry_and_legacy_compatibility(version, current_gravity, delta):
    model = PrivilegedTransformerTeacher(get_robot_contract('k1_22dof'), 4, 2,
        PrivilegedTeacherConfig(width=16, heads=2, layers=1, objective_version=version))
    with torch.no_grad():
        for p in model.transition.parameters():
            p.zero_()
        model.transition[-1].bias[6:9] = torch.tensor(delta)
    current = torch.zeros(2, 3, 53)
    current[..., 6:9] = torch.tensor(current_gravity)
    output = model.decode(torch.zeros(2, 3, 4), torch.zeros(2, 3, 22), current)['next_core']
    assert torch.isfinite(output).all()
    if version == 2:
        torch.testing.assert_close(output[..., 6:9].norm(dim=-1), torch.ones(2, 3))
        if current_gravity == [0., 0., -1.] and delta == [0., 0., 1.]:
            torch.testing.assert_close(output[..., 6:9], current[..., 6:9])
    else:
        expected = torch.tensor(current_gravity)+torch.tensor(delta)
        torch.testing.assert_close(output[..., 6:9], expected.expand(2, 3, 3))


def test_teacher_unit_gravity_has_useful_finite_gradients():
    torch.manual_seed(81)
    model = PrivilegedTransformerTeacher(get_robot_contract('k1_22dof'), 4, 2,
        PrivilegedTeacherConfig(width=16, heads=2, layers=1))
    current = torch.zeros(2, 3, 53); current[..., 8] = -1
    args = (torch.zeros(2, 3, 4), torch.zeros(2, 3, 22), current)
    target = torch.tensor([.6, 0., -.8])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    before = (model.decode(*args)['next_core'][..., 6:9]-target).square().mean()
    optimizer.zero_grad(); before.backward()
    assert model.transition[-1].bias.grad[6:9].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    after = (model.decode(*args)['next_core'][..., 6:9]-target).square().mean()
    assert after < before


def test_one_winning_case_cannot_hide_overall_regression():
    candidate, baseline, trial = trial_reports()
    for base, new in zip(baseline, trial):
        new['tracking'] = {name: value*1.019 for name, value in base['tracking'].items()}
    trial[0]['tracking'][TRACKING_METRICS[0]] = baseline[0]['tracking'][TRACKING_METRICS[0]]*.97
    result = compare_trials(candidate, baseline, trial)
    assert all(case['passed'] for case in result['cases'])
    assert not result['accepted'] and not result['aggregate_nonregression']


@pytest.mark.parametrize('case', ['small_primary', 'secondary_only', 'primary', 'outcome', 'zero'])
def test_trial_requires_aggregate_primary_or_outcome_gain(case):
    candidate, baseline, trial = trial_reports()
    for base, new in zip(baseline, trial):
        new['tracking'] = dict(base['tracking'])
    if case == 'small_primary':
        trial[0]['tracking'][TRACKING_METRICS[0]] *= .97
    elif case == 'secondary_only':
        for new in trial:
            new['tracking'][TRACKING_METRICS[1]] *= .9
    elif case == 'primary':
        for new in trial:
            new['tracking'][TRACKING_METRICS[0]] *= .97
    elif case == 'outcome':
        trial[0]['physical_failures'] -= 1
    else:
        for report in baseline+trial:
            report['tracking'] = {name: 0. for name in TRACKING_METRICS}
    result = compare_trials(candidate, baseline, trial)
    assert result['accepted'] is (case in ('primary', 'outcome'))
    assert result['aggregate_nonregression']
    if case == 'secondary_only':
        assert compare_trials(candidate, baseline, trial, primary_metric=TRACKING_METRICS[1])['accepted']
    if case == 'zero':
        assert all(row['relative_change'] == 0 for row in result['aggregate_tracking'].values())


def test_trial_rejects_unknown_primary_metric():
    with pytest.raises(ValueError, match='primary metric'):
        compare_trials(*trial_reports(), primary_metric='reward_mean')
