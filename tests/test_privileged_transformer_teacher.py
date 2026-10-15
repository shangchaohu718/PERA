from dataclasses import replace

import pytest
import torch

from unified_motion.privileged_teacher import PrivilegedTeacherConfig, PrivilegedTransformerTeacher
from unified_motion.robot_contract import get_robot_contract
from unified_motion.k1_training import create_custom_k1_algorithm, create_k1_trainer
from unified_motion.k1_policy_backends import K1ActorConfig
from unified_motion.holomotion_mjlab_ppo import (
    HoloMotionPPOConfig, build_episode_causal_mask,
)
from unified_motion.holomotion_original_algorithm import flat_observations_to_tensordict
from unified_motion.holomotion_physics_runtime import state_fingerprint
from unified_motion.k1_policy_runtime import load_k1_checkpoint, resume_k1_trainer, frozen_k1_policy, evaluate_k1_policy
from test_k1_actor_switching import task


def test_teacher_is_causal_and_does_not_cross_episode_boundaries():
    torch.manual_seed(13)
    robot = get_robot_contract('k1_22dof')
    cfg = PrivilegedTeacherConfig(width=16, heads=2, layers=2, context_steps=4)
    model = PrivilegedTransformerTeacher(robot, 4, 2, cfg).eval()
    inputs = torch.randn(2, 8, cfg.input_dim(22, 2))
    starts = torch.zeros(2, 8, dtype=torch.bool)
    starts[:, 0] = starts[:, 4] = True
    actions, core = torch.randn(2, 8, 22), torch.randn(2, 8, 53)
    with torch.no_grad():
        reference = model(inputs, starts, actions, core)
        changed = inputs.clone(); changed[:, 3:] += 50
        future_changed = model(changed, starts, actions, core)
        changed = inputs.clone(); changed[:, :4] -= 50
        previous_episode_changed = model(changed, starts, actions, core)
    for name in reference:
        torch.testing.assert_close(reference[name][:, :3], future_changed[name][:, :3])
        torch.testing.assert_close(reference[name][:, 4:], previous_episode_changed[name][:, 4:])


def test_v3_teacher_predicts_deltas_and_action_conditioned_future():
    torch.manual_seed(31)
    robot = get_robot_contract('k1_22dof')
    cfg = PrivilegedTeacherConfig(objective_version=3, width=16, heads=2, layers=1,
        context_steps=4, prediction_horizons=(1, 2), multistep_weight=.5)
    model = PrivilegedTransformerTeacher(robot, 4, 2, cfg)
    inputs = torch.randn(2, 3, cfg.input_dim(22, 2))
    starts = torch.tensor([[True, False, False], [True, False, False]])
    actions = torch.randn(2, 3, 22)
    current = torch.randn(2, 3, 53)
    current[..., 6:9] = torch.nn.functional.normalize(current[..., 6:9], dim=-1)
    output = model(inputs, starts, actions, current)
    torch.testing.assert_close(output['delta_core'], output['next_core']-current)
    future = model.predict_future_core(
        output['latent'][:, :2], torch.stack((actions[:, :2], actions[:, 1:3]), -2),
        current[:, :2], horizon=2)
    assert future.shape == (2, 2, 53)
    torch.testing.assert_close(future[..., 6:9].norm(dim=-1), torch.ones(2, 2))


def test_v3_delta_whitening_decodes_fixed_train_statistics():
    robot = get_robot_contract('k1_22dof')
    width = 53
    statistics = {str(horizon): {
        'mean': [0.01 * horizon] * width,
        'std': [0.2] * width,
    } for horizon in (1, 2)}
    cfg = PrivilegedTeacherConfig(
        objective_version=3, width=16, heads=2, layers=1, context_steps=4,
        prediction_horizons=(1, 2), delta_whitening=True,
        delta_whitening_statistics=statistics)
    model = PrivilegedTransformerTeacher(robot, 4, 2, cfg)
    torch.nn.init.zeros_(model.transition[-1].weight)
    torch.nn.init.zeros_(model.transition[-1].bias)
    torch.nn.init.zeros_(model.future_transitions['2'][-1].weight)
    torch.nn.init.zeros_(model.future_transitions['2'][-1].bias)
    inputs = torch.zeros(1, 2, cfg.input_dim(22, 2))
    starts = torch.tensor([[True, False]])
    actions = torch.zeros(1, 2, 22)
    current = torch.zeros(1, 2, width); current[..., 8] = -1
    output = model(inputs, starts, actions, current)
    non_gravity = torch.tensor([*range(6), *range(9, width)])
    torch.testing.assert_close(
        output['delta_core'].index_select(-1, non_gravity),
        torch.full((1, 2, width-3), .01))
    future = model.predict_future_core(
        output['latent'][:, :1], actions[:, :1].unsqueeze(-2).expand(-1, -1, 2, -1),
        current[:, :1], horizon=2)
    torch.testing.assert_close(
        (future-current[:, :1]).index_select(-1, non_gravity),
        torch.full((1, 1, width-3), .02))
    assert {'delta_whitening_mean', 'delta_whitening_std'} <= set(model.state_dict())


def test_delta_whitening_requires_complete_finite_statistics():
    with pytest.raises(ValueError, match='exactly cover'):
        PrivilegedTeacherConfig(objective_version=3, prediction_horizons=(1, 2),
            delta_whitening=True, delta_whitening_statistics={})
    bad = {str(h): {'mean': [0.], 'std': [0.]} for h in (1, 2)}
    with pytest.raises(ValueError, match='std floor'):
        PrivilegedTeacherConfig(objective_version=3, prediction_horizons=(1, 2),
            delta_whitening=True, delta_whitening_statistics=bad)


def test_v3_teacher_only_trains_multistep_heads():
    torch.manual_seed(37)
    algorithm = create_custom_k1_algorithm(
        K1ActorConfig(backbone='mlp', d_model=16, hidden_dim=24, n_layers=1))
    trainer = create_k1_trainer(algorithm, task(), physics=True, teacher=True,
        privileged_teacher_config=dict(objective_version=3, width=16, heads=2, layers=1,
            context_steps=4, train_envs=2, warmup_updates=1,
            prediction_horizons=(1, 2), multistep_weight=.5),
        dynamics_config=dict(physics_width=4, memory_width=4, latent_width=2, hidden_dim=8),
        ppo_config=HoloMotionPPOConfig(num_learning_epochs=1, num_mini_batches=2,
            aux_state_pred_enabled=False, inactive_expert_margin_enabled=False),
        dynamics_burn_in_steps=2, dynamics_train_steps=4, dynamics_epochs=1)
    trainer.prepare_teacher_only()
    before = state_fingerprint(trainer.privileged_teacher.future_transitions)
    metrics = None
    for _ in range(2):
        metrics = trainer.update_teacher_only(trainer.collect(3))
    assert metrics is not None and metrics['teacher_multistep_loss'] > 0
    assert before != state_fingerprint(trainer.privileged_teacher.future_transitions)


def make_trainer():
    algorithm = create_custom_k1_algorithm(K1ActorConfig(backbone='mlp', d_model=16, hidden_dim=24, n_layers=1))
    return create_k1_trainer(algorithm, task(), physics=True, teacher=True,
        privileged_teacher_config=dict(width=16, heads=2, layers=1, context_steps=4, train_envs=2, warmup_updates=1),
        dynamics_config=dict(physics_width=4, memory_width=4, latent_width=2, hidden_dim=8),
        ppo_config=HoloMotionPPOConfig(num_learning_epochs=1, num_mini_batches=2,
            aux_state_pred_enabled=False, inactive_expert_margin_enabled=False),
        dynamics_burn_in_steps=2, dynamics_train_steps=4, dynamics_epochs=1)


def test_online_teacher_student_update_checkpoint_and_privilege_free_inference(tmp_path):
    torch.manual_seed(7)
    trainer = make_trainer()
    teacher_before = state_fingerprint(trainer.privileged_teacher)
    student_before = state_fingerprint(trainer.dynamics)
    for _ in range(2):
        rollout = trainer.collect(3)
        assert rollout.privileged_teacher_input.shape[:2] == (2, 3)
        trainer.update(rollout)
        metrics = trainer.update_dynamics(rollout)
    assert metrics is not None and metrics.teacher_distillation_loss == 0
    assert metrics.teacher_admitted_fraction == 0  # Insufficient evidence must fail closed.
    assert metrics.teacher_validation_samples == 2
    assert teacher_before != state_fingerprint(trainer.privileged_teacher)
    assert student_before != state_fingerprint(trainer.dynamics)
    assert trainer.task.full_resets == 1 and trainer.task.steps == 6
    assert trainer.privileged_teacher_updates == 1
    assert not trainer.dynamics.teacher_student
    path = trainer.save_checkpoint(tmp_path/'teacher.pt', iteration=1)
    loaded = load_k1_checkpoint(path)
    assert 'privileged_teacher' in loaded.payload
    assert all('teacher' not in name for name, _ in loaded.actor_dynamics.named_parameters())
    inference_task = task()
    def forbidden():
        raise AssertionError('frozen inference read privileged Teacher input')
    inference_task.physics_teacher_target = forbidden
    inference_task.privileged_physics_observation = forbidden
    report = evaluate_k1_policy(frozen_k1_policy(loaded), inference_task, steps=5)
    assert report['weights_unchanged']
    resumed = resume_k1_trainer(path, task())
    assert resumed.privileged_teacher_updates == 1
    assert state_fingerprint(resumed.privileged_teacher, resumed.privileged_teacher_ema) == state_fingerprint(
        trainer.privileged_teacher, trainer.privileged_teacher_ema)
    assert resumed.privileged_teacher_optimizer.state
    for _ in range(2):
        rollout = resumed.collect(3)
        resumed.update_dynamics(rollout)
    assert resumed.privileged_teacher_updates == 2
    payload = torch.load(path, weights_only=True)
    del payload['privileged_teacher']
    torch.save(payload, tmp_path/'incomplete.pt')
    with pytest.raises(ValueError, match='incomplete privileged'):
        load_k1_checkpoint(tmp_path/'incomplete.pt')


def test_teacher_only_updates_no_control_or_student_weights():
    torch.manual_seed(17)
    trainer = make_trainer()
    trainer.prepare_teacher_only()
    frozen_before = state_fingerprint(
        trainer.actor, trainer.critic, trainer.dynamics, trainer.actor_dynamics,
        trainer.confidence)
    teacher_before = state_fingerprint(trainer.privileged_teacher)
    metrics = None
    for _ in range(2):
        metrics = trainer.update_teacher_only(trainer.collect(3))
    assert metrics is not None
    assert metrics['teacher_updates'] == 1
    assert metrics['teacher_validation_samples'] == 2
    assert frozen_before == state_fingerprint(
        trainer.actor, trainer.critic, trainer.dynamics, trainer.actor_dynamics,
        trainer.confidence)
    assert teacher_before != state_fingerprint(trainer.privileged_teacher)
    assert all(not parameter.requires_grad for module in (
        trainer.actor, trainer.critic, trainer.dynamics, trainer.actor_dynamics,
        trainer.confidence) for parameter in module.parameters())


def test_teacher_control_uses_teacher_predictions_and_never_updates_student(tmp_path):
    torch.manual_seed(23)
    trainer = make_trainer()
    trainer.privileged_teacher_updates = 1
    trainer.prepare_teacher_control()
    assert trainer.actor.config.physics_feature_dim == 55
    frozen_before = state_fingerprint(
        trainer.actor.base_actor, trainer.dynamics, trainer.actor_dynamics,
        trainer.privileged_teacher, trainer.privileged_teacher_ema)
    residual_before = state_fingerprint(trainer.actor.physics_residual)
    rollout = trainer.collect(3)
    assert rollout.physics_features.shape == (3, 3, 55)
    assert torch.equal(rollout.physics_confidence, torch.ones(3, 3, 1))
    assert rollout.teacher_candidate_actions is not None
    # The freshly initialized residual is exactly zero: the corrected mean is
    # therefore the base mean supplied to Teacher.  This must not be the random
    # base sample used by the old, invalid action-dependent transform.
    torch.testing.assert_close(
        rollout.teacher_candidate_actions, rollout.old_mu, rtol=0, atol=0
    )
    with torch.no_grad():
        replay = trainer._actor_forward(
            flat_observations_to_tensordict(
                rollout.actor_observation, rollout.critic_observation),
            actions=rollout.actions,
            mode='sequence_logp',
            attn_mask=build_episode_causal_mask(rollout.dones.squeeze(-1)),
            update_obs_norm=False,
            physics_features=rollout.physics_features,
            physics_confidence=rollout.physics_confidence,
        )
    torch.testing.assert_close(
        replay.get('actions_log_prob'), rollout.old_log_prob, rtol=1e-5, atol=1e-6
    )
    # The Teacher may use the base action while choosing the current residual,
    # but its retained history must contain the command that drove the simulator.
    torch.testing.assert_close(
        trainer._teacher_control_history[2][-1], rollout.executed_actions[:, -1]
    )
    trainer.update(rollout)
    assert trainer.update_dynamics(rollout) is None
    assert residual_before != state_fingerprint(trainer.actor.physics_residual)
    assert frozen_before == state_fingerprint(
        trainer.actor.base_actor, trainer.dynamics, trainer.actor_dynamics,
        trainer.privileged_teacher, trainer.privileged_teacher_ema)
    path = trainer.save_checkpoint(tmp_path/'teacher_control.pt', iteration=2)
    loaded = load_k1_checkpoint(path)
    assert loaded.payload['physics_runtime_config']['teacher_action_proposal'] == 'base_mean_pre_sample'
    report = evaluate_k1_policy(frozen_k1_policy(loaded), task(), steps=3)
    assert report['weights_unchanged']
    no_physics = evaluate_k1_policy(
        frozen_k1_policy(loaded, ablation='no_physics'), task(), steps=3)
    assert no_physics['weights_unchanged']
    resumed = resume_k1_trainer(path, task())
    assert resumed._physics_control_mode == 'teacher'
    assert resumed.actor.config.physics_feature_dim == 55
    resumed_rollout = resumed.collect(2)
    assert resumed_rollout.physics_features.shape == (3, 2, 55)
    legacy_payload = torch.load(path, map_location='cpu', weights_only=True)
    del legacy_payload['physics_runtime_config']['teacher_action_proposal']
    legacy_path = tmp_path/'legacy_sample_conditioned_teacher_control.pt'
    torch.save(legacy_payload, legacy_path)
    with pytest.raises(ValueError, match='sampled-action residuals'):
        resume_k1_trainer(legacy_path, task())


def test_frozen_teacher_replaces_candidate_with_executed_action_history(tmp_path):
    torch.manual_seed(2300)
    trainer = make_trainer()
    trainer.privileged_teacher_updates = 1
    trainer.prepare_teacher_control()
    path = trainer.save_checkpoint(tmp_path/'teacher_control.pt', iteration=2)
    policy = frozen_k1_policy(load_k1_checkpoint(path))
    inference_task = task()
    observation = inference_task.reset()
    starts = torch.ones(inference_task.num_envs, dtype=torch.bool)
    candidate_result = policy.act_privileged(observation, inference_task, starts)
    executed = candidate_result.detach().clone().add_(0.03125)
    policy.record_action(executed)
    torch.testing.assert_close(policy.history[2][-1], executed)
    assert not policy.pending and policy.steps == 1
    with pytest.raises(RuntimeError, match='must precede'):
        policy.record_action(executed)


def test_frozen_teacher_can_train_a_fresh_base_actor_and_residual():
    torch.manual_seed(2301)
    trainer = make_trainer()
    trainer.privileged_teacher_updates = 1
    teacher_before = state_fingerprint(
        trainer.privileged_teacher, trainer.privileged_teacher_ema)
    dynamics_before = state_fingerprint(
        trainer.dynamics, trainer.actor_dynamics, trainer.confidence)
    base_before = state_fingerprint(trainer.actor.base_actor)
    trainer.prepare_teacher_control(freeze_base_actor=False, residual_limit=.1)
    residual_before = state_fingerprint(trainer.actor.physics_residual)
    assert not trainer.update_config.freeze_base_actor
    assert trainer.actor.config.residual_limit == .1
    assert all(parameter.requires_grad for parameter in trainer.actor.base_actor.parameters())
    assert all(not parameter.requires_grad for module in (
        trainer.privileged_teacher, trainer.privileged_teacher_ema,
        trainer.dynamics, trainer.actor_dynamics, trainer.confidence,
    ) for parameter in module.parameters())
    optimized = {
        id(parameter)
        for group in trainer.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimized for parameter in trainer.actor.parameters())

    rollout = trainer.collect(3)
    trainer.update(rollout)
    assert trainer.update_dynamics(rollout) is None
    assert base_before != state_fingerprint(trainer.actor.base_actor)
    assert residual_before != state_fingerprint(trainer.actor.physics_residual)
    assert teacher_before == state_fingerprint(
        trainer.privileged_teacher, trainer.privileged_teacher_ema)
    assert dynamics_before == state_fingerprint(
        trainer.dynamics, trainer.actor_dynamics, trainer.confidence)


def test_teacher_gradients_do_not_reach_student_or_input_truth():
    from unified_motion.privileged_teacher import teacher_supervision
    robot = get_robot_contract('k1_22dof')
    cfg = PrivilegedTeacherConfig(width=8, heads=2, layers=1)
    teacher = PrivilegedTransformerTeacher(robot, 2, 2, cfg)
    inputs = torch.randn(1, 3, cfg.input_dim(22, 2))
    starts = torch.tensor([[True, False, False]])
    output = teacher(inputs, starts, torch.zeros(1, 3, 22), torch.zeros(1, 3, 53))
    student = torch.randn(1, 3, 2, requires_grad=True)
    distillation = (student-output['latent'].detach()).square().mean()
    distillation.backward()
    assert student.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    truth = torch.randn(1, 3, 53, requires_grad=True)
    loss = teacher_supervision(output, truth, torch.ones_like(truth, dtype=torch.bool),
        torch.zeros(1, 3, 2), torch.ones(1, 3, 2, dtype=torch.bool), torch.zeros(1, 3, 2),
        torch.ones(1, 3, dtype=torch.bool))
    loss.backward()
    assert truth.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in teacher.parameters())


def test_collection_uses_current_truth_and_masks_missing_scene_slots():
    from unified_motion.privileged_teacher import teacher_inputs
    trainer = make_trainer()
    def truth():
        frame = trainer.task.deployable_physics()
        body = frame.body_obs.clone()
        body[:, 0] = trainer.task.steps
        return replace(frame, body_obs=body)
    trainer.task.privileged_physics_observation = truth
    rollout = trainer.collect(3)
    torch.testing.assert_close(rollout.privileged_teacher_core[0, :, 3], torch.tensor([0., .1, .2]))
    frame = truth()
    # Missing scene data must neither propagate NaNs nor become teacher evidence.
    hidden_objects = torch.full((3, 2, 16), float('nan'))
    hidden = replace(frame, object_features=hidden_objects, object_mask=torch.zeros(3, 2, dtype=torch.bool))
    cfg = trainer.privileged_teacher.config
    clean = teacher_inputs(torch.zeros(3, 843), frame, torch.zeros(3, 2), trainer.dynamics.robot, cfg)
    masked = teacher_inputs(torch.zeros(3, 843), hidden, torch.zeros(3, 2), trainer.dynamics.robot, cfg)
    torch.testing.assert_close(clean, masked)
    with pytest.raises(ValueError, match='exceeds configured slots'):
        teacher_inputs(torch.zeros(3, 843), hidden, torch.zeros(3, 2), trainer.dynamics.robot,
                       replace(cfg, object_slots=1))


def test_calibration_tail_is_excluded_from_teacher_training():
    import copy
    trainer = make_trainer()
    rollout = trainer.collect(6)
    untouched = copy.deepcopy(trainer.privileged_teacher.state_dict())
    optimizer = copy.deepcopy(trainer.privileged_teacher_optimizer.state_dict())
    args = (rollout.next_physics_observation, rollout.physics_episode_start,
            rollout.executed_actions, rollout.physics_teacher_target, 2, 5)
    trainer._update_privileged_teacher([rollout], *args)
    expected = state_fingerprint(trainer.privileged_teacher)
    trainer.privileged_teacher.load_state_dict(untouched)
    trainer.privileged_teacher_optimizer.load_state_dict(optimizer)
    rollout.privileged_teacher_input[:, 5:] += 1000.
    rollout.next_physics_observation.body_obs[:, 5:] += 1000.
    rollout.physics_teacher_target[:, 5:] += 1000.
    trainer._update_privileged_teacher([rollout], *args)
    assert state_fingerprint(trainer.privileged_teacher) == expected
