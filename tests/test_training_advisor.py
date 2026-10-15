"""No network calls: exercise provider boundaries and experiment controls with explicit fixtures."""
from copy import deepcopy
from dataclasses import asdict
import json
from threading import Event
from urllib.error import HTTPError, URLError

import pytest
import torch

from unified_motion.llm.client import ChatClient, ChatConfig, AdvisorAPIError, PROVIDERS
from unified_motion.llm.advisor import (PeriodicTrainingAdvisor, apply_trial_adjustment,
    fingerprint, make_request, propose, tuning_snapshot, validate_changes)
from unified_motion.llm.validation import compare_trials
from test_privileged_transformer_teacher import make_trainer


SETTINGS = {'ppo.entropy_coef': .005, 'optimizer.critic_learning_rate': 5e-5,
            'loss.teacher_distillation_weight': 0., 'curriculum.clip_weights': [1., 1.]}
RECORDS = [{'iteration': 99, 'reward_mean': 3., 'done_count': 2,
            'physics_confidence_mean': 0., 'reward_terms': {'root_position_error_m': .1},
            'path': 'PRIVATE_PATH', 'API_KEY': 'PRIVATE_KEY'}]
CLIP_RECORDS = [{**RECORDS[0], 'clips': {
    'PRIVATE_CLIP_A': dict(clip_id=0, sampled_steps=100, physical_failures=5,
        reference_completions=2, tracking={'root_position_error_m': .4}),
    'PRIVATE_CLIP_B': dict(clip_id=1, sampled_steps=100, physical_failures=1,
        reference_completions=8, tracking={'root_position_error_m': .1})}}]


def response(content='{"reason":"Test fixture, not empirical evidence.","changes":{}}', finish='stop'):
    return dict(choices=[dict(finish_reason=finish, message=dict(content=content))],
                usage=dict(prompt_tokens=12, completion_tokens=8, total_tokens=20, private='omit'))


class FakeClient:
    def __init__(self, changes=None):
        self.changes = changes or {}

    def complete(self, system, user):
        return json.dumps(dict(reason='Synthetic test hypothesis.', changes=self.changes)), dict(usage={}, attempts=1)


@pytest.mark.parametrize('provider', ['deepseek', 'glm'])
def test_provider_payload_and_account_key_stay_separate(monkeypatch, provider):
    endpoint, variable = PROVIDERS[provider]
    monkeypatch.setenv(variable, 'TEST_ONLY_SECRET')
    calls = []
    def transport(url, payload, key, timeout):
        calls.append((url, payload, key, timeout))
        return response()
    config = ChatConfig(provider, 'account-model', thinking=False)
    answer, cost = ChatClient(config, transport=transport).complete('Return JSON.', '{}')
    assert calls[0][0] == endpoint and calls[0][2] == 'TEST_ONLY_SECRET'
    assert calls[0][1]['response_format'] == {'type': 'json_object'}
    assert calls[0][1]['thinking'] == {'type': 'disabled'}
    assert 'TEST_ONLY_SECRET' not in json.dumps((calls[0][1], asdict(config), cost))
    assert cost['usage'] == dict(prompt_tokens=12, completion_tokens=8, total_tokens=20)
    assert json.loads(answer)['changes'] == {}


@pytest.mark.parametrize('code,expected_calls', [(429, 3), (503, 3), (401, 1), (302, 1)])
def test_http_retry_budget_and_sanitized_errors(monkeypatch, code, expected_calls):
    monkeypatch.setenv('GLM_API_KEY', 'TEST_SECRET')
    calls = []
    def transport(*args):
        calls.append(1)
        raise HTTPError('https://example.invalid/TEST_SECRET', code, 'TEST_SECRET', {}, None)
    with pytest.raises(AdvisorAPIError) as error:
        ChatClient(ChatConfig('glm', 'test', retries=2), transport=transport, sleep=lambda _: None).complete('JSON', '{}')
    assert len(calls) == expected_calls and 'TEST_SECRET' not in str(error.value)


def test_timeouts_are_not_retried_and_server_json_mode_is_optional(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'TEST_SECRET')
    calls = []
    def transport(url, payload, *args):
        assert 'response_format' not in payload
        calls.append(1)
        raise URLError('TEST_SECRET')
    with pytest.raises(AdvisorAPIError, match='connection or timeout'):
        ChatClient(ChatConfig('deepseek', 'test', retries=2, json_mode=False), transport=transport).complete('JSON', '{}')
    assert len(calls) == 1


@pytest.mark.parametrize('value', [response('', 'stop'), response('{}', 'length'), {}, {'choices': []}])
def test_bad_provider_answers_cannot_become_proposals(monkeypatch, value):
    monkeypatch.setenv('GLM_API_KEY', 'TEST_SECRET')
    client = ChatClient(ChatConfig('glm', 'test'), transport=lambda *args: value)
    with pytest.raises(AdvisorAPIError):
        propose(client, SETTINGS, RECORDS)


@pytest.mark.parametrize('changes', [
    {'actor.weight': [1.]}, {'physics.gate': 1.}, {'ppo.entropy_coef': .009},
    {'ppo.entropy_coef': float('nan')}, {'ppo.value_loss_coef': True},
    {'loss.teacher_distillation_weight': .01},
    {'curriculum.clip_weights': [1.5, .5]}, {'curriculum.clip_weights': [0., 2.]},
    {'curriculum.clip_weights': [1.]}, {'curriculum.clip_weights': [1., 1.1]},
])
def test_invalid_or_out_of_budget_adjustments_are_rejected(changes):
    with pytest.raises(ValueError):
        validate_changes(SETTINGS, changes)


def test_prompt_contains_only_whitelisted_numeric_evidence():
    request = make_request(SETTINGS, RECORDS)
    assert 'PRIVATE_' not in request and 'root_position_error_m' in request
    assert json.loads(request)['evidence']['gate_nonzero_fraction'] == 0
    for settings in ({**SETTINGS, 'API_KEY': 'SECRET'}, {'ppo.entropy_coef': 'SECRET'}):
        with pytest.raises(ValueError):
            make_request(settings, RECORDS)
    assert validate_changes(SETTINGS, {'curriculum.clip_weights': [1.2, .8]})['curriculum.clip_weights'] == [1.2, .8]


def test_explicit_trial_is_applied_before_collection_and_persisted(tmp_path):
    trainer = make_trainer()
    before = tuning_snapshot(trainer)
    proposal = propose(FakeClient({'ppo.entropy_coef': before['ppo.entropy_coef']*.9}), before, RECORDS)
    apply_trial_adjustment(trainer, proposal)
    assert trainer.config.entropy_coef == before['ppo.entropy_coef']*.9
    with pytest.raises(ValueError, match='stale'):
        apply_trial_adjustment(trainer, proposal)
    path = trainer.save_checkpoint(tmp_path/'trial.pt', iteration=0)
    from unified_motion.k1_policy_runtime import resume_k1_trainer
    from test_k1_actor_switching import task
    resumed = resume_k1_trainer(path, task())
    assert resumed.config.entropy_coef == trainer.config.entropy_coef
    assert resumed.training_adjustment == proposal
    resumed.collect(1)
    with pytest.raises(ValueError, match='before collecting'):
        apply_trial_adjustment(resumed, proposal)


def test_background_advisor_never_changes_training_and_obeys_call_budget(tmp_path):
    trainer = make_trainer()
    before = tuning_snapshot(trainer)
    entered, release = Event(), Event()
    class BlockingClient(FakeClient):
        def complete(self, *args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test synchronization timeout')
            return super().complete(*args)
    advisor = PeriodicTrainingAdvisor(dict(interval=1, window=1, max_calls=1), tmp_path,
        client=BlockingClient({'ppo.entropy_coef': before['ppo.entropy_coef']*.9}))
    try:
        advisor.on_iteration(RECORDS[0], trainer)
        assert entered.wait(5)
        advisor.on_iteration({**RECORDS[0], 'iteration': 100}, trainer)
        assert advisor.calls == 1 and tuning_snapshot(trainer) == before
    finally:
        release.set()
        advisor.close()
    candidate = json.loads((tmp_path/'proposal_99.json').read_text())
    assert candidate['status'] == 'unvalidated_candidate'
    assert tuning_snapshot(trainer) == before


def test_advisor_failure_records_no_secret_and_leaves_training_alone(tmp_path):
    class BrokenClient:
        def complete(self, *args):
            raise RuntimeError('TEST_SECRET')
    trainer = make_trainer()
    before = tuning_snapshot(trainer)
    advisor = PeriodicTrainingAdvisor(dict(interval=1, window=1, max_calls=1), tmp_path, client=BrokenClient())
    advisor.on_iteration(RECORDS[0], trainer)
    advisor.close()
    log = (tmp_path/'events.jsonl').read_text()
    assert 'failed_no_change' in log and 'TEST_SECRET' not in log
    assert tuning_snapshot(trainer) == before and not list(tmp_path.glob('proposal*'))


def test_rejected_proposal_still_records_successful_api_call_cost(tmp_path):
    trainer = make_trainer()
    advisor = PeriodicTrainingAdvisor(dict(interval=1, window=1, max_calls=1), tmp_path,
        client=FakeClient({'actor.weight': 1.}))
    advisor.on_iteration(RECORDS[0], trainer)
    advisor.close()
    outcome = json.loads((tmp_path/'events.jsonl').read_text())
    assert outcome['status'] == 'failed_no_change' and outcome['api']['attempts'] == 1


def trial_reports():
    """Synthetic admission inputs only, NOT a real MJLab run or validation result."""
    candidate = propose(FakeClient({'ppo.entropy_coef': .0045}), SETTINGS, RECORDS)
    baseline, trial = [], []
    for seed in range(3):
        protocol = dict(seed=seed, ablation='full', motion_sha256=['a'*64], steps=100)
        report = dict(evaluation_protocol=protocol, evaluation_protocol_sha256=fingerprint(protocol),
            deterministic=True, weights_unchanged=True, backend='mjlab_cuda',
            training_source_checkpoint_sha256='b'*64, training_seed=42,
            training_protocol=dict(motion_sha256=['a'*64], num_envs=10, steps_per_env=10),
            training_cost=dict(compute_seconds=10., transitions=1000),
            tracking=dict(root_position_error_m=.1, root_orientation_error_rad=.1,
                root_linear_velocity_error_mps=.2, keybody_linear_velocity_rmse_mps=.2),
            physical_failures=3, reference_completions=5, reward_mean=10.)
        report['training_trial'] = dict(schema=1, source_settings=deepcopy(SETTINGS),
            initial_settings=deepcopy(SETTINGS), source_adjustment_sha256=None, applied_adjustment_sha256=None)
        report['training_trial_sha256'] = fingerprint(report['training_trial'])
        baseline.append(report)
        improved = deepcopy(report)
        improved['trial_adjustment_sha256'] = fingerprint(candidate)
        improved['training_trial']['initial_settings'] = deepcopy(candidate['settings_after'])
        improved['training_trial']['applied_adjustment_sha256'] = fingerprint(candidate)
        improved['training_trial_sha256'] = fingerprint(improved['training_trial'])
        improved['tracking']['root_position_error_m'] = .09
        improved['reward_mean'] = -100.  # Shaped reward must not determine admission.
        trial.append(improved)
    return candidate, baseline, trial


def test_admission_uses_fixed_metrics_and_cost_not_changed_shaped_reward():
    candidate, baseline, trial = trial_reports()
    assert compare_trials(candidate, baseline, trial)['accepted']
    assert not compare_trials(candidate, baseline, trial)['deployment_changed']
    trial[0]['physical_failures'] = 4
    assert not compare_trials(candidate, baseline, trial)['accepted']
    trial[0]['physical_failures'] = 3
    trial[1]['training_cost']['compute_seconds'] = 13.
    assert not compare_trials(candidate, baseline, trial)['accepted']
    trial[1]['training_cost']['compute_seconds'] = 10.
    trial[1]['tracking']['keybody_linear_velocity_rmse_mps'] = .3
    assert not compare_trials(candidate, baseline, trial)['accepted']


@pytest.mark.parametrize('mutation', ['seed_count', 'training_budget', 'source', 'candidate', 'frozen', 'missing_outcome', 'training_motions'])
def test_admission_rejects_unmatched_or_incomplete_evidence(mutation):
    candidate, baseline, trial = trial_reports()
    if mutation == 'seed_count':
        baseline, trial = baseline[:2], trial[:2]
    elif mutation == 'training_budget':
        trial[0]['training_cost']['transitions'] += 1
    elif mutation == 'source':
        trial[0]['training_source_checkpoint_sha256'] = 'c'*64
    elif mutation == 'candidate':
        trial[0]['trial_adjustment_sha256'] = 'c'*64
    elif mutation == 'frozen':
        trial[0]['weights_unchanged'] = False
    elif mutation == 'training_motions':
        trial[0]['training_protocol']['motion_sha256'] = ['c'*64]
    else:
        trial[0]['physical_failures'] = None
    with pytest.raises(ValueError):
        compare_trials(candidate, baseline, trial)


def test_curriculum_only_affects_resets_and_respects_eligible_clips():
    from test_data_rewards_bridge import clip_fixture, ROBOT
    from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
    library = MotionLibrary(ROBOT, [clip_fixture(name='short', frames=2),
        clip_fixture(name='a'), clip_fixture(name='b')])
    sampler = MotionSampler(library, num_envs=4000, step_dt=.02, future_offsets_s=(0.,))
    sampler.reset(clip_ids=[1]*4000)
    sampler.advance()
    before = (sampler.clip_ids.clone(), sampler.step_counts.clone())
    sampler.set_sampling_weights([100., 1., 3.])
    torch.testing.assert_close(sampler.clip_ids, before[0])
    torch.testing.assert_close(sampler.step_counts, before[1])
    torch.manual_seed(12)
    sampler.reset(min_remaining_s=.5)
    assert not (sampler.clip_ids == 0).any()
    assert .70 < float((sampler.clip_ids == 2).float().mean()) < .80
    for weights in ([1., 0., 1.], [1., float('nan'), 1.], [1.]):
        with pytest.raises(ValueError):
            sampler.set_sampling_weights(weights)


def test_curriculum_checkpoint_restores_sampling_without_reset(tmp_path):
    from test_data_rewards_bridge import clip_fixture, ROBOT
    from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
    from unified_motion.k1_policy_runtime import resume_k1_trainer
    from test_k1_actor_switching import task
    library = MotionLibrary(ROBOT, [clip_fixture(name='a'), clip_fixture(name='b')])
    trainer = make_trainer()
    trainer.task.sampler = MotionSampler(library, num_envs=3, step_dt=.02, future_offsets_s=(0.,))
    before = tuning_snapshot(trainer)
    proposal = propose(FakeClient({'curriculum.clip_weights': [1.2, .8]}), before, CLIP_RECORDS)
    apply_trial_adjustment(trainer, proposal)
    path = trainer.save_checkpoint(tmp_path/'curriculum.pt', iteration=0)
    new_task = task()
    new_task.sampler = MotionSampler(library, num_envs=3, step_dt=.02, future_offsets_s=(0.,))
    resumed = resume_k1_trainer(path, new_task)
    torch.testing.assert_close(resumed.task.sampler.sampling_weights, torch.tensor([1.2, .8]))
    assert (resumed.task.sampler.clip_ids == -1).all()
