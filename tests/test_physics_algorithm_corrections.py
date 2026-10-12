"""Algorithm counterexamples, calibration holdout and complete-policy update guards."""
import copy
from dataclasses import replace

import pytest
import torch

from physics_fixtures import TensorTask
from test_holomotion_rollout_boundaries import _real_trainer
from unified_motion.algorithms.common import diagonal_gaussian_kl
from unified_motion.algorithms.dynamics_history import DynamicsHistory, snapshot_physics
from unified_motion.algorithms.dynamics_objective import identification_loss
from unified_motion.algorithms.sequence_training import PhysicsSequenceCollector
from unified_motion.holomotion_physics_actor import HoloMotionPhysicsActorConfig, PhysicsActionResidual
from unified_motion.holomotion_physics_runtime import state_fingerprint, FrozenPhysicsPolicy
from unified_motion.models.physics import DynamicsModel, _core_state
from unified_motion.physics_confidence import PhysicsConfidence, PhysicsConfidenceConfig
from unified_motion.physics_data import stack_physics
from unified_motion.physics_update import PhysicsUpdateConfig


def _model():
    task = TensorTask(objects=0, endings=False)
    net = DynamicsModel(task.robot, physics_width=8, memory_width=8, latent_width=4,
        hidden_dim=16, uncertainty=True, system_id_dim=2, prediction_horizons=(1, 2)).eval()
    return task, net


def test_unobserved_velocity_never_becomes_innovation_or_encoder_signal():
    task, net = _model()
    observed = snapshot_physics(task.read())
    observed.core_valid[:, :3] = False
    encoding, memory = net.step(observed, episode_start=task.episode_start, env_ids=task.env_ids)
    prediction = net.predict_next(observed, torch.zeros(2, 22), encoding.latent)
    prediction.next_core = _core_state(observed, task.robot).clone()
    prediction.next_core[:, 0] = 1.
    memory = net.remember_prediction(memory, prediction)
    starts = torch.zeros(2, dtype=torch.bool)
    result, _ = net.step(observed, episode_start=starts, env_ids=task.env_ids, memory=memory)
    assert result.prediction_error.count_nonzero() == 0
    # Arbitrary placeholders for missing values must not influence belief.
    changed = snapshot_physics(observed)
    changed.base_lin_vel_b.fill_(100.)
    again, _ = net.step(changed, episode_start=starts, env_ids=task.env_ids, memory=memory)
    torch.testing.assert_close(again.latent, result.latent)
    action = torch.zeros(2, 22)
    torch.testing.assert_close(net.predict_next(observed, action, result.latent).next_core,
                               net.predict_next(changed, action, result.latent).next_core)
    commands = action[:, None].expand(-1, 2, -1)
    torch.testing.assert_close(net.predict_future_core(observed, commands, result.latent, horizon=2),
                               net.predict_future_core(changed, commands, result.latent, horizon=2))
    sequence = stack_physics([observed, changed])
    encoded = net.forward_sequence(sequence, episode_start=torch.tensor([[True, False]]*2),
        previous_predicted_core=prediction.next_core[:, None].expand(-1, 2, -1),
        previous_prediction_valid=torch.ones(2, 2, dtype=torch.bool))
    assert encoded.prediction_error[:, :, :3].count_nonzero() == 0
    # Unavailable targets contribute no artificial zero-velocity loss.
    prediction.next_core_logvar = None
    loss = net.prediction_loss(prediction, observed, valid_transition=torch.ones(2, dtype=torch.bool))
    assert loss.terms["base_linear_velocity"] == 0


def test_analytic_kl_gradient_matches_normal_kl_and_numeric_derivative():
    old_mu = torch.zeros(3, 4, dtype=torch.float64)
    old_sigma = torch.ones_like(old_mu)
    mu = torch.ones_like(old_mu, requires_grad=True)
    sigma = torch.full_like(old_mu, .7, requires_grad=True)
    actual = diagonal_gaussian_kl(old_mu, old_sigma, mu, sigma)
    expected = torch.distributions.kl_divergence(torch.distributions.Normal(old_mu, old_sigma),
        torch.distributions.Normal(mu, sigma)).sum(-1)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.sum(), mu)[0]
    eps = 1e-5
    numeric = (diagonal_gaussian_kl(old_mu, old_sigma, mu.detach()+eps, sigma.detach())
               - diagonal_gaussian_kl(old_mu, old_sigma, mu.detach()-eps, sigma.detach()))/(2*eps)
    torch.testing.assert_close(grad.sum(-1), numeric)
    assert grad.abs().min() > 1


def test_no_information_and_uncalibrated_models_cannot_open_gate():
    calibration = PhysicsConfidence(2)
    targets = ((torch.arange(1024)+.5)/512-1)[:, None].expand(-1, 2)
    mean = torch.zeros_like(targets)
    logvar = torch.full_like(targets, torch.log(torch.tensor(1/3)))
    age = torch.full((1024,), 10000)
    assert calibration(logvar, age, 128).count_nonzero() == 0
    calibration.update(mean, logvar, targets, torch.ones(1024, dtype=torch.bool))
    assert calibration(logvar, age, 128).count_nonzero() == 0
    # A falsely confident variance cannot override the held-out error floor.
    logvar.fill_(-6)
    calibration.update(mean, logvar, targets, torch.ones(1024, dtype=torch.bool))
    assert calibration(logvar, age, 128).count_nonzero() == 0


def test_informative_heldout_model_can_open_gate_only_after_real_evidence():
    calibration = PhysicsConfidence(2)
    target = torch.rand(128, 2)*2-1
    logvar = torch.full_like(target, -4.)
    calibration.update(target+.05, logvar, target, torch.ones(128, dtype=torch.bool))
    assert calibration(logvar, torch.zeros(128, dtype=torch.long), 128).count_nonzero() == 0
    assert calibration(logvar, torch.full((128,), 128), 128).min() > .8


def test_conditioned_residual_distinguishes_opposite_commands():
    net = PhysicsActionResidual(HoloMotionPhysicsActorConfig(physics_feature_dim=6, hidden_dim=8, action_dim=2))
    with torch.no_grad():
        net.net[1].weight.zero_(); net.net[1].bias.zero_()
        net.action_condition.weight.fill_(1)
        net.net[-1].weight.fill_(1)
    result = net(torch.zeros(2, 6), torch.tensor([[.5, .5], [-.5, -.5]]))
    assert (result[0] > 0).all() and (result[1] < 0).all()


def test_direct_identification_has_no_teacher_dependency():
    _, net = _model()
    latent = torch.randn(2, 3, 4, requires_grad=True)
    terms, _, _ = identification_loss(net, latent, torch.randn(2, 3, 2), torch.ones(2, 3, dtype=torch.bool))
    terms["system_identification"].backward()
    assert net.teacher_encoder is None and latent.grad.abs().sum() > 0
    assert terms["teacher_distillation"] == terms["teacher_reconstruction"] == 0


def _training(monkeypatch):
    trainer = _real_trainer(True, monkeypatch, boundaries=False)
    trainer.dynamics_burn_in_steps = 2
    trainer.dynamics_train_steps = 6
    trainer.dynamics_epochs = 1
    return trainer


def test_calibration_labels_are_excluded_from_optimizer_loss(monkeypatch):
    a, b = _training(monkeypatch), _training(monkeypatch)
    torch.manual_seed(3); ra = a.collect(8)
    torch.manual_seed(3); rb = b.collect(8)
    rb.physics_teacher_target[:, -2:] = 50.
    a.update_dynamics(ra); b.update_dynamics(rb)
    assert state_fingerprint(a.dynamics) == state_fingerprint(b.dynamics)
    assert not torch.equal(a.confidence.mse, b.confidence.mse)


def test_excessive_feature_update_is_rejected_without_world_or_policy_mutation(monkeypatch):
    trainer = _training(monkeypatch)
    trainer.confidence = PhysicsConfidence(2, PhysicsConfidenceConfig(mode="ungated"))
    trainer.update_config = replace(trainer.update_config, maximum_mean_kl=1e-15, maximum_token_kl=1e-15)
    trainer.dynamics_ema_decay = .01
    with torch.no_grad():
        trainer.actor.physics_residual.net[-1].weight.fill_(.3)
        bias = trainer.dynamics.encoder.merge[1].bias
        bias.add_(10*torch.randn_like(bias))
    rollout = trainer.collect(8)
    before = state_fingerprint(trainer.actor, trainer.actor_dynamics, trainer.confidence)
    observation, age, state = trainer._observation, trainer._physics_age.clone(), trainer.task.state.clone()
    metrics = trainer.update_dynamics(rollout)
    assert not metrics.deployment_update_accepted
    assert metrics.deployment_mean_kl > 1e-15
    assert state_fingerprint(trainer.actor, trainer.actor_dynamics, trainer.confidence) == before
    assert trainer._observation is observation
    torch.testing.assert_close(trainer._physics_age, age)
    torch.testing.assert_close(trainer.task.state, state)


def test_base_actor_freeze_is_respected_by_real_ppo_update(monkeypatch):
    trainer = _training(monkeypatch)
    trainer.confidence = PhysicsConfidence(2, PhysicsConfidenceConfig(mode="ungated"))
    base = state_fingerprint(trainer.actor.base_actor)
    residual = state_fingerprint(trainer.actor.physics_residual)
    rollout = trainer.collect(8)
    trainer.update(rollout)
    assert state_fingerprint(trainer.actor.base_actor) == base
    assert state_fingerprint(trainer.actor.physics_residual) != residual


def test_cpu_chunk_replay_matches_individual_frames_and_preserves_missingness():
    task, net = _model()
    history = DynamicsHistory(chunk_steps=3)
    memory = None
    for i in range(11):
        obs = snapshot_physics(task.read())
        obs.core_valid[:, :3] = False
        obs.base_lin_vel_b.fill_(100.)
        starts = task.episode_start
        action = torch.randn(2, 22)*.01
        history.append(obs, starts, action)
        enc, memory = net.step(obs, episode_start=starts, env_ids=task.env_ids, memory=memory)
        memory = net.remember_prediction(memory, net.predict_next(obs, action, enc.latent))
        task.step(action)
    rebuilt, _ = history.rebuild(net, task.env_ids)
    torch.testing.assert_close(rebuilt.hidden, memory.hidden)
    torch.testing.assert_close(rebuilt.predicted_core, memory.predicted_core)
    assert all(item.observation.body_obs.device.type == "cpu" for item in history.steps)


def test_ema_step_half_life_is_independent_of_rollout_partition():
    config = PhysicsUpdateConfig(ema_half_life_steps=1024)
    assert config.ema_decay(32)**32 == pytest.approx(.5)
    assert config.ema_decay(192)**2 == pytest.approx(config.ema_decay(384))


def test_history_byte_budget_is_checked_before_world_step():
    from physics_fixtures import make_models
    task, net, actor = make_models(endings=False)
    collector = PhysicsSequenceCollector(task, net, actor, dynamics_history_bytes=1)
    with pytest.raises(RuntimeError, match="max_bytes"):
        collector.collect(1)
    assert task.ticks.tolist() == [0, 0]


def test_direct_id_full_training_and_checkpoint_without_teacher(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unified_motion.holomotion_physics_training import HoloMotionPhysicsMJLabPPO
    original = _real_trainer(False, monkeypatch, boundaries=False)
    net = DynamicsModel(original.task.bridge.robot, physics_width=4, memory_width=4,
        latent_width=2, hidden_dim=8, system_id_dim=2, teacher_student=False, uncertainty=True)
    trainer = HoloMotionPhysicsMJLabPPO(SimpleNamespace(actor=original.actor, critic=original.critic,
        network_config=original._network_config), original.task, net, config=original.config,
        dynamics_burn_in_steps=2, dynamics_train_steps=6, dynamics_epochs=1,
        confidence_config=PhysicsConfidenceConfig(mode="ungated"))
    rollout = trainer.collect(8)
    trainer.update(rollout)
    metrics = trainer.update_dynamics(rollout)
    assert metrics.teacher_reconstruction_loss == metrics.teacher_distillation_loss == 0
    path = trainer.save_checkpoint(tmp_path/"direct.pt", iteration=1)
    policy = FrozenPhysicsPolicy.from_checkpoint(path)
    assert policy.dynamics.teacher_encoder is None
    assert policy.calibration.config.mode == "ungated"
