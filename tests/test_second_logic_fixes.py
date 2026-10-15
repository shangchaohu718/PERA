"""Regression evidence for statistics, stochastic replay and experiment identity."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from test_holomotion_rollout_boundaries import _real_trainer
from test_k1_actor_switching import trainer as actor_trainer, task, assert_replay
from test_privileged_transformer_teacher import make_trainer
from test_training_advisor import trial_reports, FakeClient, RECORDS
from test_data_rewards_bridge import clip_fixture, ROBOT
from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.holomotion_original_algorithm import build_original_holomotion_networks
from unified_motion.holomotion_mjlab_ppo import HoloMotionMJLabPPO
from unified_motion.k1_training import create_k1_trainer
from unified_motion.k1_policy_runtime import resume_k1_trainer, load_k1_checkpoint, frozen_k1_policy, evaluate_k1_policy
from unified_motion.holomotion_physics_runtime import state_fingerprint
from unified_motion.privileged_teacher import assess_teacher, normalized_core
from unified_motion.llm.advisor import fingerprint, apply_trial_adjustment, propose, tuning_snapshot
from unified_motion.llm.validation import compare_trials


@pytest.mark.parametrize('physics', [False, True])
def test_normalization_updates_after_ppo_and_survives_resume(monkeypatch, tmp_path, physics):
    trainer = actor_trainer('holomotion', physics, monkeypatch)
    actor = getattr(trainer.actor, 'base_actor', trainer.actor)
    norms = (actor.obs_normalizer, trainer.critic.obs_normalizer)
    assert all(int(n.count) == 0 for n in norms)
    rollout = trainer.collect(4)
    assert all(int(n.count) == 0 for n in norms)
    if physics:
        assert_replay(trainer, rollout)
    # Observe every normalizer count on every optimizer replay, not just at entry.
    original = trainer._actor_forward
    counts = []
    def capture(*args, **kwargs):
        counts.append([int(n.count) for n in norms])
        return original(*args, **kwargs)
    trainer._actor_forward = capture
    trainer.update(rollout)
    assert counts and all(row == [0, 0] for row in counts)
    assert all(int(n.count) == 12 for n in norms)
    assert all(torch.isfinite(n._std).all() and n._mean.abs().max() > 0 for n in norms)
    path = trainer.save_checkpoint(tmp_path/'stats.pt', iteration=0)
    resumed = resume_k1_trainer(path, task())
    restored_actor = getattr(resumed.actor, 'base_actor', resumed.actor)
    assert state_fingerprint(*norms) == state_fingerprint(restored_actor.obs_normalizer, resumed.critic.obs_normalizer)
    resumed.update(resumed.collect(4))
    assert int(restored_actor.obs_normalizer.count) == 24
    assert int(resumed.critic.obs_normalizer.count) == 24
    loaded = load_k1_checkpoint(path)
    assert evaluate_k1_policy(frozen_k1_policy(loaded), task(), steps=4)['weights_unchanged']


def test_frozen_actor_and_pretrained_normalization_are_preserved(monkeypatch):
    trainer = _real_trainer(True, monkeypatch)
    before = state_fingerprint(trainer.actor.base_actor)
    trainer.update(trainer.collect(4))
    assert state_fingerprint(trainer.actor.base_actor) == before
    assert int(trainer.critic.obs_normalizer.count) == 12
    source = _real_trainer(False, monkeypatch)
    algorithm = SimpleNamespace(actor=source.actor, critic=source.critic, network_config=source._network_config, pretrained=True)
    pretrained = HoloMotionMJLabPPO(algorithm, source.task, config=source.config)
    before = state_fingerprint(pretrained.actor.obs_normalizer, pretrained.critic.obs_normalizer)
    pretrained.update(pretrained.collect(4))
    assert state_fingerprint(pretrained.actor.obs_normalizer, pretrained.critic.obs_normalizer) == before


def test_old_checkpoint_without_normalization_mode_preserves_frozen_stats(monkeypatch, tmp_path):
    trainer = _real_trainer(False, monkeypatch)
    path = trainer.save_checkpoint(tmp_path/'original.pt', iteration=0)
    payload = torch.load(path, weights_only=True)
    del payload['observation_normalization']
    legacy = tmp_path/'legacy.pt'; torch.save(payload, legacy)
    resumed = resume_k1_trainer(legacy, task())
    resumed.update(resumed.collect(4))
    assert int(resumed.actor.obs_normalizer.count) == int(resumed.critic.obs_normalizer.count) == 0
    migrated = resume_k1_trainer(legacy, task())
    migrated.configure_observation_normalization(enabled=True)
    migrated.update(migrated.collect(4))
    assert int(migrated.actor.obs_normalizer.count) == 12
    with pytest.raises(ValueError, match='before collection'):
        migrated.configure_observation_normalization(enabled=False)


@pytest.mark.parametrize('physics', [False, True])
def test_nonzero_dropout_has_exact_preupdate_replay(monkeypatch, physics):
    source = _real_trainer(False, monkeypatch)
    config = deepcopy(source._network_config)
    config['actor'].update(attn_dropout=.3, mlp_dropout=.3)
    actor, critic = build_original_holomotion_networks(config)
    algorithm = SimpleNamespace(actor=actor, critic=critic, network_config=config)
    options = dict(dynamics_config=dict(physics_width=4, memory_width=4, latent_width=2, hidden_dim=8)) if physics else {}
    trainer = create_k1_trainer(algorithm, task(), physics=physics, ppo_config=source.config, **options)
    trainer.actor.train()  # Even a caller changing mode cannot corrupt sampling.
    rollout = trainer.collect(5)
    assert not trainer.actor.training
    assert_replay(trainer, rollout)
    trainer.actor.train()
    metric = trainer.update(rollout)
    assert not trainer.actor.training and metric.num_updates > 0


def test_teacher_contact_persistence_uses_current_truth_not_student_placeholder():
    trainer = make_trainer(); trainer.task.boundaries = False
    read = trainer.task.privileged_physics_observation
    def truth():
        frame = read(); contacts = frame.contact_features.clone(); contacts[..., 10] = 1.
        return replace(frame, contact_features=contacts)
    trainer.task.privileged_physics_observation = truth
    rollout = trainer.collect(32)
    assert rollout.physics_observation.contact_features[..., 10].count_nonzero() == 0
    assert rollout.privileged_teacher_contacts.eq(1).all()
    ids, targets, _, _ = trainer._update_privileged_teacher([rollout], rollout.next_physics_observation,
        rollout.physics_episode_start, rollout.executed_actions, rollout.physics_teacher_target, 2, 16, validation_end=24)
    assert targets['current_contacts'].eq(1).all()
    core = normalized_core(rollout.next_physics_observation, trainer.dynamics.robot)[ids, 16:24]
    contacts = rollout.next_physics_observation.contact_features[ids, 16:24, :, 10]
    mediocre = dict(targets['validation'], contact_logits=torch.zeros_like(contacts))
    flags, details = assess_teacher(mediocre, torch.zeros_like(core), torch.full_like(contacts, math.log(.1/.9)),
        targets['shuffled'], core, targets['current'], rollout.next_physics_observation.core_valid[ids, 16:24],
        contacts, targets['current_contacts'], targets['current_contact_mask'], trainer.privileged_teacher.config)
    assert flags[-1] == 0
    assert details[-1]['persistence'] == pytest.approx(-math.log(.99), abs=1e-6)


def sampler(names, *, changed=False):
    clips = [clip_fixture(name=name) for name in names]
    if changed:
        clips[0] = replace(clips[0], joint_pos=clips[0].joint_pos+.01)
    return MotionSampler(MotionLibrary(ROBOT, clips), num_envs=3, step_dt=.02, future_offsets_s=(0.,))


@pytest.mark.parametrize('mutation', ['order', 'content', 'legacy_identity'])
def test_weighted_resume_rejects_mismatched_motion_identity(tmp_path, mutation):
    trainer = make_trainer(); trainer.task.sampler = sampler(['moonwalk', 'dance'])
    trainer.task.sampler.set_sampling_weights([1.2, .8])
    path = trainer.save_checkpoint(tmp_path/'weighted.pt', iteration=0)
    other_task = task()
    other_task.sampler = sampler(['dance', 'moonwalk'] if mutation == 'order' else ['moonwalk', 'dance'], changed=mutation == 'content')
    if mutation == 'legacy_identity':
        payload = torch.load(path, weights_only=True)
        del payload['physics_runtime_config']['motion_sampling_manifest']
        torch.save(payload, path)
    with pytest.raises(ValueError, match='motion (identity|library)'):
        resume_k1_trainer(path, other_task)
    assert other_task.sampler.sampling_weights is None


@pytest.mark.parametrize('mutation', ['different_proposal', 'changed_settings', 'missing_evidence'])
def test_trial_rejects_modified_or_unverifiable_baseline(mutation):
    candidate, baseline, trial = trial_reports()
    for report in baseline:
        if mutation == 'different_proposal':
            report['trial_adjustment_sha256'] = 'c'*64
            report['training_trial']['applied_adjustment_sha256'] = 'c'*64
        elif mutation == 'changed_settings':
            report['training_trial']['initial_settings']['ppo.entropy_coef'] *= 1.1
        else:
            del report['training_trial']
        if 'training_trial' in report:
            report['training_trial_sha256'] = fingerprint(report['training_trial'])
    with pytest.raises(ValueError, match='(baseline|starting trial)'):
        compare_trials(candidate, baseline, trial)


def test_inherited_advice_is_distinct_from_a_new_baseline_adjustment():
    candidate, baseline, trial = trial_reports()
    for report in baseline+trial:
        report['training_trial']['source_adjustment_sha256'] = 'd'*64
        report['training_trial_sha256'] = fingerprint(report['training_trial'])
    for report in baseline:
        report['trial_adjustment_sha256'] = 'd'*64
    assert compare_trials(candidate, baseline, trial)['accepted']


def test_trial_accepts_materialized_float32_settings_but_not_changed_values():
    candidate, baseline, trial = trial_reports()
    for report in trial:
        actual = report['training_trial']['initial_settings']
        actual['ppo.entropy_coef'] = float(torch.tensor(actual['ppo.entropy_coef']))
        report['training_trial_sha256'] = fingerprint(report['training_trial'])
    assert compare_trials(candidate, baseline, trial)['accepted']
    trial[0]['training_trial']['initial_settings']['ppo.entropy_coef'] *= 1.01
    trial[0]['training_trial_sha256'] = fingerprint(trial[0]['training_trial'])
    with pytest.raises(ValueError, match='baseline must be unchanged'):
        compare_trials(candidate, baseline, trial)


def test_resume_starts_new_trial_preserving_inherited_advice(tmp_path):
    trainer = make_trainer()
    before = tuning_snapshot(trainer)
    proposal = propose(FakeClient({'ppo.entropy_coef': before['ppo.entropy_coef']*.9}), before, RECORDS)
    apply_trial_adjustment(trainer, proposal)
    path = trainer.save_checkpoint(tmp_path/'first.pt', iteration=0)
    restored = resume_k1_trainer(path, task())
    assert restored.training_trial['applied_adjustment_sha256'] is None
    assert restored.training_trial['source_adjustment_sha256'] == fingerprint(proposal)
    assert restored.training_trial['source_settings'] == tuning_snapshot(restored)
    next_proposal = propose(FakeClient({'ppo.entropy_coef': restored.config.entropy_coef*.9}), tuning_snapshot(restored), RECORDS)
    apply_trial_adjustment(restored, next_proposal)
    assert restored.training_trial['applied_adjustment_sha256'] == fingerprint(next_proposal)
    assert restored.training_trial['initial_settings'] == next_proposal['settings_after']
