from dataclasses import replace

import pytest
import torch

from unified_motion.privileged_teacher import PrivilegedTeacherConfig, PrivilegedTransformerTeacher
from unified_motion.robot_contract import get_robot_contract
from unified_motion.k1_training import create_custom_k1_algorithm, create_k1_trainer
from unified_motion.k1_policy_backends import K1ActorConfig
from unified_motion.holomotion_mjlab_ppo import HoloMotionPPOConfig
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
