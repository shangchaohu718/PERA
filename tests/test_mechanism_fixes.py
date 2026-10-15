"""Counterexamples for feedback starvation, unqualified Teachers and blind tuning."""
import copy
import json
from dataclasses import asdict, replace

import pytest
import torch

from unified_motion.physics_confidence import PhysicsConfidence, PhysicsConfidenceConfig
from unified_motion.holomotion_physics_runtime import state_fingerprint
from unified_motion.privileged_teacher import (PrivilegedTeacherConfig, assess_teacher,
    predictive_distillation, teacher_supervision)
from unified_motion.llm.advisor import (apply_trial_adjustment, tuning_snapshot, propose,
    make_request, summarize, ProposalValidationError)
from test_privileged_transformer_teacher import make_trainer
from test_physics_algorithm_corrections import _training
from test_k1_actor_switching import assert_replay, task
from test_training_advisor import FakeClient, RECORDS, SETTINGS, CLIP_RECORDS


def test_bounded_gate_trains_residual_with_uninformative_id_using_real_ppo(monkeypatch):
    fingerprints = []
    for mode in ('calibrated', 'bounded'):
        torch.manual_seed(51)
        trainer = _training(monkeypatch)
        trainer.confidence = PhysicsConfidence(2, PhysicsConfidenceConfig(mode=mode))
        residual = trainer.actor.physics_residual.net[-1]
        before = residual.weight.detach().clone()
        rollout = trainer.collect(8)
        assert_replay(trainer, rollout)
        trainer.update(rollout)
        movement = (residual.weight-before).abs().max().item()
        fingerprints.append(movement)
        if mode == 'bounded':
            assert rollout.physics_confidence.min() == pytest.approx(.1)
            assert residual.weight.grad.abs().sum() > 0
    assert fingerprints[0] == 0 and fingerprints[1] > 1e-7


def test_one_unidentifiable_parameter_does_not_suppress_all_feedback():
    module = PhysicsConfidence(2, PhysicsConfidenceConfig(mode='bounded'))
    module.samples.fill_(100)
    module.mse.copy_(torch.tensor([1/3, .01]))
    assert module(torch.full((1, 2), -8.), torch.tensor([100]), 10).item() > .5
    assert module(torch.full((1, 2), -8.), torch.tensor([0]), 10).item() == pytest.approx(.1)


def test_old_gate_state_missing_floor_keeps_exact_calibrated_behavior():
    old = PhysicsConfidence(2)
    old.samples.fill_(32); old.mse.fill_(.1)
    state = copy.deepcopy(old.state_dict())
    del state['_extra_state']['feedback_floor']
    restored = PhysicsConfidence(2, PhysicsConfidenceConfig(**state['_extra_state']))
    restored.load_state_dict(state)
    x, age = torch.full((4, 2), -3.), torch.tensor([0, 4, 16, 32])
    torch.testing.assert_close(restored(x, age, 32), old(x, age, 32), rtol=0, atol=0)


def _quality_fixture():
    truth = torch.ones(2, 16, 53)
    contacts = torch.ones(2, 16, 2)
    teacher = dict(next_core=truth.clone(), contact_logits=contacts*8, latent=torch.randn(2, 16, 3))
    student, student_logits = torch.zeros_like(truth), torch.zeros_like(contacts)
    shuffled = dict(next_core=truth+1, contact_logits=-contacts*8)
    args = (teacher, student, student_logits, shuffled, truth, student,
        torch.ones_like(truth, dtype=torch.bool), contacts, torch.zeros_like(contacts),
        torch.ones_like(contacts, dtype=torch.bool), PrivilegedTeacherConfig())
    return args


def test_predictive_teacher_is_admitted_without_requiring_immediate_action_response():
    args = _quality_fixture()
    admitted, details = assess_teacher(*args)
    assert admitted.tolist() == [1.]*6
    assert all(d['samples'] == 32 for d in details)
    indifferent = list(args); indifferent[3] = args[0]
    indifferent_flags, indifferent_details = assess_teacher(*indifferent)
    assert indifferent_flags.tolist() == [1.]*6
    assert all(not row['action_sensitive'] for row in indifferent_details)
    bad = list(args); bad[0] = args[3]
    assert assess_teacher(*bad)[0].sum() == 0
    insufficient = list(args); insufficient[-1] = replace(args[-1], minimum_validation_samples=33)
    assert assess_teacher(*insufficient)[0].sum() == 0
    masked = list(args); masked[6] = args[6].clone(); masked[6][..., :3] = False
    assert assess_teacher(*masked)[0].tolist() == [0., 1., 1., 1., 1., 1.]


def test_physical_distillation_ignores_latent_coordinates_and_stops_teacher_gradients():
    teacher, student, logits, _, truth, _, valid, contacts, _, contact_valid, _ = _quality_fixture()
    student.requires_grad_(); logits.requires_grad_()
    teacher = {k: v.clone().requires_grad_() for k, v in teacher.items()}
    loss = predictive_distillation(student, logits, teacher, valid, contact_valid,
        torch.ones(2, 16, dtype=torch.bool), torch.ones(6))
    rotated = {**teacher, 'latent': -teacher['latent']}
    again = predictive_distillation(student, logits, rotated, valid, contact_valid,
        torch.ones(2, 16, dtype=torch.bool), torch.ones(6))
    torch.testing.assert_close(loss, again, rtol=0, atol=0)
    loss.backward()
    assert student.grad.abs().sum() > 0 and logits.grad.abs().sum() > 0
    assert all(v.grad is None for v in teacher.values())
    assert predictive_distillation(student, logits, teacher, valid, contact_valid,
        torch.ones(2, 16, dtype=torch.bool), torch.zeros(6)) == 0


def test_teacher_v2_does_not_learn_to_copy_privileged_parameter_labels():
    trainer = make_trainer()
    rollout = trainer.collect(8)
    trainer._update_privileged_teacher([rollout], rollout.next_physics_observation,
        rollout.physics_episode_start, rollout.executed_actions, rollout.physics_teacher_target, 2, 4,
        validation_end=6)
    assert all(p.grad is None for p in trainer.privileged_teacher.identification.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in trainer.privileged_teacher.transition.parameters())


def test_rejected_or_masked_teacher_nan_cannot_poison_student_gradients():
    teacher, student, logits, _, _, _, valid, _, _, contact_valid, _ = _quality_fixture()
    student.requires_grad_(); logits.requires_grad_()
    teacher['next_core'].fill_(float('nan'))
    teacher['contact_logits'].fill_(float('nan'))
    loss = predictive_distillation(student, logits, teacher, valid, contact_valid,
        torch.ones(2, 16, dtype=torch.bool), torch.zeros(6))
    loss.backward()
    assert loss == 0 and torch.isfinite(student.grad).all() and torch.isfinite(logits.grad).all()
    assert student.grad.count_nonzero() == logits.grad.count_nonzero() == 0


def test_teacher_quality_holdout_and_calibration_are_excluded_from_both_optimizers():
    torch.manual_seed(21); a = make_trainer()
    torch.manual_seed(21); b = make_trainer()
    torch.manual_seed(22); ra = a.collect(16)
    torch.manual_seed(22); rb = b.collect(16)
    # burn=2, calibration=[12,16), Teacher validation=[8,12), training=[2,8).
    rb.privileged_teacher_input[:, 8:] += 1000
    rb.privileged_teacher_core[:, 8:] += 1000
    rb.next_physics_observation.body_obs[:, 8:] += 1000
    rb.physics_teacher_target[:, 8:] += 1000
    ma, mb = a.update_dynamics(ra), b.update_dynamics(rb)
    # Eight Teacher validation transitions cannot pass the minimum of 16.
    assert ma.teacher_admitted_fraction == mb.teacher_admitted_fraction == 0
    assert state_fingerprint(a.privileged_teacher) == state_fingerprint(b.privileged_teacher)
    assert state_fingerprint(a.dynamics) == state_fingerprint(b.dynamics)


def test_qualified_teacher_targets_reach_student_optimizer(monkeypatch):
    """Inject an explicit oracle fixture to exercise the admitted training path."""
    from unified_motion.privileged_teacher import normalized_core
    torch.manual_seed(123)
    trainer = make_trainer()
    rollout = trainer.collect(32)
    rollout.next_physics_observation.base_lin_vel_b.add_(2.)
    original = trainer._update_privileged_teacher
    def oracle(batches, next_frame, starts, actions, parameters, burn, train_end, validation_end=None):
        ids, targets, loss, ramp = original(batches, next_frame, starts, actions, parameters,
            burn, train_end, validation_end=validation_end)
        truth = normalized_core(next_frame, trainer.dynamics.robot)[ids]
        targets['next_core'] = truth[:, burn:train_end].clone()
        targets['validation']['next_core'] = truth[:, train_end:validation_end].clone()
        targets['shuffled']['next_core'] = truth[:, train_end:validation_end]+1.
        return ids, targets, loss, ramp
    monkeypatch.setattr(trainer, '_update_privileged_teacher', oracle)
    metrics = trainer.update_dynamics(rollout)
    assert metrics.teacher_validation_samples >= 16
    assert metrics.teacher_admitted_fraction > 0 and metrics.teacher_distillation_loss > 0
    assert trainer.task.full_resets == 1 and trainer.task.steps == 32


def test_multistep_head_receives_training_gradients_without_crossing_resets():
    torch.manual_seed(28)
    trainer = make_trainer()
    trainer.task.boundaries = False
    rollout = trainer.collect(20)
    captured = []
    original = trainer.dynamics.predict_future_core
    def capture(obs, sequence, latent, **kwargs):
        captured.append(sequence.detach().clone())
        return original(obs, sequence, latent, **kwargs)
    trainer.dynamics.predict_future_core = capture
    before = state_fingerprint(trainer.dynamics.future_state_predictors)
    metric = trainer.update_dynamics(rollout)
    assert metric.multistep_prediction_loss > 0 and captured
    assert state_fingerprint(trainer.dynamics.future_state_predictors) != before
    # The implementation passes recorded executed actions, in their original order.
    torch.testing.assert_close(captured[0][:, 0], rollout.executed_actions[:, 2:6])
    other = make_trainer()
    r = other.collect(16)
    r.physics_episode_start.fill_(True)
    assert other.update_dynamics(r).multistep_prediction_loss == 0


def test_old_teacher_checkpoint_restores_old_objective_and_new_checkpoint_keeps_v2(tmp_path):
    from unified_motion.k1_policy_runtime import resume_k1_trainer
    trainer = make_trainer()
    path = trainer.save_checkpoint(tmp_path/'new.pt', iteration=0)
    assert resume_k1_trainer(path, task()).privileged_teacher.config.objective_version == 2
    payload = torch.load(path, weights_only=True)
    config = payload['privileged_teacher']['config']
    for key in ('objective_version', 'validation_fraction', 'minimum_validation_samples',
                'minimum_prediction_gain', 'minimum_action_gain'):
        config.pop(key)
    payload['physics_runtime_config']['privileged_teacher_config'] = dict(config)
    torch.save(payload, tmp_path/'old.pt')
    restored = resume_k1_trainer(tmp_path/'old.pt', task())
    assert restored.privileged_teacher.config.objective_version == 1
    assert restored.update_dynamics(restored.collect(6)).teacher_distillation_loss > 0


def test_advisor_distinguishes_opposite_trends_and_opposite_clip_difficulty():
    ascending = [dict(iteration=i*10, reward_terms={'root_position_error_m': x})
                 for i, x in enumerate((.1, .2, .3, .4))]
    descending = [dict(iteration=i*10, reward_terms={'root_position_error_m': x})
                  for i, x in enumerate((.4, .3, .2, .1))]
    a, b = summarize(ascending), summarize(descending)
    assert a['tracking']['root_position_error_m']['slope_per_iteration'] > 0
    assert b['tracking']['root_position_error_m']['slope_per_iteration'] < 0
    assert make_request(SETTINGS, ascending) != make_request(SETTINGS, descending)
    clips = copy.deepcopy(CLIP_RECORDS)
    values = list(clips[0]['clips'].values())
    values[0]['tracking'], values[1]['tracking'] = values[1]['tracking'], values[0]['tracking']
    assert make_request(SETTINGS, CLIP_RECORDS) != make_request(SETTINGS, clips)
    assert 'PRIVATE_' not in make_request(SETTINGS, CLIP_RECORDS)
    with pytest.raises(ValueError, match='increasing'):
        summarize(ascending[::-1])
    with pytest.raises(ValueError, match='finite'):
        summarize([dict(iteration=0, reward_mean=float('nan'))])


def test_curriculum_requires_evidence_for_every_modified_clip():
    client = FakeClient({'curriculum.clip_weights': [1.2, .8]})
    with pytest.raises(ProposalValidationError):
        propose(client, SETTINGS, RECORDS)
    candidate = propose(client, SETTINGS, CLIP_RECORDS)
    assert len(candidate['evidence']['clips']) == 2
    missing = copy.deepcopy(CLIP_RECORDS)
    missing[0]['clips']['PRIVATE_CLIP_B']['sampled_steps'] = 10
    with pytest.raises(ProposalValidationError):
        propose(client, SETTINGS, missing)


def test_advisor_learning_rates_reach_optimizers_and_survive_resume(tmp_path):
    from unified_motion.k1_policy_runtime import resume_k1_trainer
    trainer = make_trainer()
    settings = tuning_snapshot(trainer)
    assert 'ppo.value_loss_coef' not in settings and 'loss.teacher_reconstruction_weight' not in settings
    changes = {key: settings[key]*1.2 for key in ('optimizer.critic_learning_rate', 'optimizer.teacher_learning_rate')}
    candidate = propose(FakeClient(changes), settings, RECORDS)
    apply_trial_adjustment(trainer, candidate)
    assert trainer.config.critic_learning_rate == changes['optimizer.critic_learning_rate']
    assert trainer.privileged_teacher.config.learning_rate == changes['optimizer.teacher_learning_rate']
    path = trainer.save_checkpoint(tmp_path/'rates.pt', iteration=0)
    restored = resume_k1_trainer(path, task())
    for key, optimizer in [('optimizer.critic_learning_rate', restored.critic_optimizer),
                           ('optimizer.teacher_learning_rate', restored.privileged_teacher_optimizer)]:
        assert all(group['lr'] == changes[key] for group in optimizer.param_groups)
    assert tuning_snapshot(restored) == tuning_snapshot(trainer)


def test_advisor_rate_change_changes_real_critic_update():
    torch.manual_seed(71); baseline = make_trainer()
    torch.manual_seed(71); candidate = make_trainer()
    for trainer in (baseline, candidate):
        trainer.config = replace(trainer.config, desired_kl=None, num_mini_batches=1)
    before = tuning_snapshot(candidate)
    proposal = propose(FakeClient({'optimizer.critic_learning_rate': before['optimizer.critic_learning_rate']*1.25}),
                       before, RECORDS)
    apply_trial_adjustment(candidate, proposal)
    torch.manual_seed(82); a = baseline.collect(6)
    torch.manual_seed(82); b = candidate.collect(6)
    origin = [p.detach().clone() for p in baseline.critic.parameters()]
    baseline.update(a); candidate.update(b)
    da = torch.cat([(p-x).flatten() for p, x in zip(baseline.critic.parameters(), origin)])
    db = torch.cat([(p-x).flatten() for p, x in zip(candidate.critic.parameters(), origin)])
    assert float((db.norm()/da.norm()).detach()) == pytest.approx(1.25, abs=.005)


def test_offline_advisor_uses_latest_logged_optimizer_settings(tmp_path, monkeypatch):
    from tools.advise_training import main
    stale = {**SETTINGS, 'optimizer.critic_learning_rate': 9e-5}
    metrics, metadata, output = (tmp_path/name for name in ('metrics.jsonl', 'runner.json', 'preview.json'))
    metrics.write_text(json.dumps({**RECORDS[0], 'training_tuning': SETTINGS})+'\n', encoding='utf-8')
    metadata.write_text(json.dumps({'training_tuning': stale}), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['advise_training.py', '--provider', 'glm', '--model', 'test',
        '--metrics', str(metrics), '--runner-metadata', str(metadata), '--output', str(output), '--dry-run'])
    main()
    assert json.loads(output.read_text(encoding='utf-8'))['user']['current_settings'] == SETTINGS


def test_runner_logs_stable_clip_ids_fixed_errors_and_separate_outcomes(tmp_path, monkeypatch):
    from unified_motion.holomotion_mjlab_runner import HoloMotionMJLabRunner, HoloMotionRunnerConfig
    trainer = make_trainer()
    trainer.task.sampler.step_dt = .02
    trainer.task.sampler.library = type('Library', (), {'clip_names': ['PRIVATE_A', 'PRIVATE_B', 'PRIVATE_C']})()
    original = trainer.task.step
    def with_metrics(action):
        transition = original(action)
        transition.terms.update(root_position_error_m=torch.tensor([.3, .2, .1]),
            physical_failure=torch.tensor([1., 0., 0.]), reference_completion=torch.tensor([0., 1., 0.]))
        return transition
    monkeypatch.setattr(trainer.task, 'step', with_metrics)
    records = HoloMotionMJLabRunner(trainer, tmp_path, HoloMotionRunnerConfig(
        num_iterations=1, steps_per_env=2, random_start=False)).run()
    record = records[0]
    assert record['physical_failures'] == record['reference_completions'] == 2
    assert record['clips']['PRIVATE_A']['clip_id'] == 0
    assert record['clips']['PRIVATE_A']['tracking']['root_position_error_m'] == pytest.approx(.3)
    assert record['training_tuning'] == tuning_snapshot(trainer)
    evidence = summarize(records)
    assert len(evidence['clips']) == 3 and 'PRIVATE_' not in json.dumps(evidence)
