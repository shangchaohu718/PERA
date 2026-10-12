"""Dynamics weight changes must not shorten physical episodes."""
import copy

import pytest
import torch

from physics_fixtures import make_models
from unified_motion.algorithms.dynamics_history import DynamicsHistory, snapshot_physics
from unified_motion.algorithms.sequence_training import (
    PhysicsSequenceCollector, PhysicsSequencePPO, DynamicsUpdater,
)
from unified_motion.models.physics import DynamicsModel


@pytest.mark.parametrize("feedback", [False, True])
def test_replay_matches_full_history_after_asynchronous_resets_and_weight_change(feedback):
    task, _, _ = make_models(endings=True)
    model = DynamicsModel(task.robot, physics_width=8, memory_width=8, latent_width=4,
                          hidden_dim=16, interaction_feedback=feedback).eval()
    history = DynamicsHistory(max_steps=12)
    full = []
    for step in range(15):
        frame, starts = snapshot_physics(task.read()), task.episode_start
        action = torch.full((2, 22), .01*(step+1))
        history.append(frame, starts, action)
        full.append((frame, starts.clone(), action.clone()))
        transition = task.step(action)
        done = transition.terminated | transition.truncated
        if done.any():
            task.reset(done.nonzero().flatten())
    assert len(history.steps) < len(full)
    with torch.no_grad():
        model.gru.weight_hh.add_(.03)
        model.state_predictor[-1].bias.add_(.1)
    model.invalidate_memories()
    before = task.ticks.clone()
    rebuilt, _ = history.rebuild(model, task.env_ids)
    memory = None
    with torch.no_grad():
        for frame, starts, action in full:
            encoding, memory = model.step(frame, episode_start=starts,
                env_ids=task.env_ids, memory=memory)
            prediction = model.predict_next(frame, action, encoding.latent,
                                             physics_encoding=encoding.physics_encoding())
            memory = model.remember_prediction(memory, prediction)
    torch.testing.assert_close(rebuilt.hidden, memory.hidden)
    torch.testing.assert_close(rebuilt.predicted_core, memory.predicted_core)
    torch.testing.assert_close(task.ticks, before)
    if feedback:
        torch.testing.assert_close(rebuilt.interaction_prediction.next_contact_logits,
                                   memory.interaction_prediction.next_contact_logits)


@pytest.mark.parametrize("kind", ["mlp", "dense", "moe"])
def test_multi_epoch_dynamics_updates_preserve_running_world_and_old_policy_replay(kind):
    task, model, policy = make_models(kind, endings=False)
    collector = PhysicsSequenceCollector(task, model, policy)
    updater = DynamicsUpdater(model, epochs=2)
    ppo = PhysicsSequencePPO(policy)
    for _ in range(3):
        rollout = collector.collect(6)
        ticks, body, resets = task.ticks.clone(), task.body.clone(), task.reset_calls
        updater.update(rollout)
        collector.on_dynamics_updated()
        torch.testing.assert_close(task.ticks, ticks)
        torch.testing.assert_close(task.body, body)
        assert task.reset_calls == resets
        again = collector.collect(1)
        assert not again.episode_start[:, 0].any()
        assert ppo.verify_old_policy(again) < 1e-4
    assert task.ticks.tolist() == [21, 21]


def test_history_limit_fails_before_world_step_and_snapshot_is_independent():
    task, model, policy = make_models("mlp", endings=False)
    collector = PhysicsSequenceCollector(task, model, policy, dynamics_history_steps=2)
    first = collector.collect(2)
    stored = collector._dynamics_history.steps[0].observation.body_obs.clone()
    task.body.add_(.1)
    torch.testing.assert_close(collector._dynamics_history.steps[0].observation.body_obs, stored)
    ticks, resets = task.ticks.clone(), task.reset_calls
    with pytest.raises(RuntimeError, match="max_steps"):
        collector.collect(1)
    torch.testing.assert_close(task.ticks, ticks)
    assert task.reset_calls == resets
    assert first.dynamics_prefix.steps == []


def test_executed_action_is_used_in_prediction_replay():
    task, model, policy = make_models("mlp", endings=False)
    step = task.step
    def delayed(action):
        task.last_executed_action = torch.zeros_like(action)
        return step(task.last_executed_action)
    task.step = delayed
    collector = PhysicsSequenceCollector(task, model, policy)
    rollout = collector.collect(3)
    assert rollout.action.abs().sum() > 0
    assert rollout.executed_action.count_nonzero() == 0
    assert all(item.executed_action.count_nonzero() == 0 for item in collector._dynamics_history.steps)
    DynamicsUpdater(model, epochs=2).update(rollout)
    collector.on_dynamics_updated()
    assert task.ticks.tolist() == [3, 3]


def test_holomotion_ema_updates_keep_observation_age_and_environment(monkeypatch):
    from test_holomotion_rollout_boundaries import _real_trainer
    trainer = _real_trainer(True, monkeypatch, boundaries=False)
    trainer.dynamics_burn_in_steps = 2
    trainer.dynamics_train_steps = 4
    trainer.dynamics_epochs = 2
    for _ in range(2):
        rollout = trainer.collect(6)
        observation = trainer._observation
        state = trainer.task.state.clone()
        age = trainer._physics_age.clone()
        resets = trainer.task.full_resets
        before = copy.deepcopy(trainer.actor_dynamics.state_dict())
        metrics = trainer.update_dynamics(rollout)
        assert metrics.optimizer_steps == 2
        assert trainer._observation is observation
        torch.testing.assert_close(trainer.task.state, state)
        torch.testing.assert_close(trainer._physics_age, age)
        assert trainer.task.full_resets == resets
        assert not torch.equal(before["gru.weight_ih"], trainer.actor_dynamics.gru.weight_ih)
        assert trainer._physics_memory.owner is trainer.actor_dynamics._owner
        expected, _ = trainer._dynamics_history.rebuild(trainer.actor_dynamics, (0, 1, 2))
        torch.testing.assert_close(trainer._physics_memory.hidden, expected.hidden)
        again = trainer.collect(1)
        assert not again.physics_episode_start[:, 0].any()
    assert trainer.task.state.tolist() == [14., 14., 14.]


def test_holomotion_records_executed_actions_separately_from_ppo_samples(monkeypatch):
    from test_holomotion_rollout_boundaries import _real_trainer
    trainer = _real_trainer(True, monkeypatch, boundaries=False)
    original_step = trainer.task.step
    def step(action):
        trainer.task.last_executed_action = torch.zeros_like(action)
        return original_step(trainer.task.last_executed_action)
    trainer.task.step = step
    rollout = trainer.collect(3)
    assert rollout.actions.abs().sum() > 0
    assert rollout.executed_actions.count_nonzero() == 0
    assert all(item.executed_action.count_nonzero() == 0 for item in trainer._dynamics_history.steps)
