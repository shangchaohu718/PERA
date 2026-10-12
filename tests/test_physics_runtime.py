import copy

import pytest
import torch

from test_holomotion_rollout_boundaries import _real_trainer, _Task
from unified_motion.holomotion_physics_runtime import (
    FrozenPhysicsPolicy, evaluate_physics_policy, load_physics_checkpoint, state_fingerprint,
)
from unified_motion.holomotion_physics_training import HoloMotionPhysicsMJLabPPO


def _saved(tmp_path, monkeypatch):
    trainer = _real_trainer(True, monkeypatch)
    trainer.dynamics_burn_in_steps = 2
    trainer.dynamics_train_steps = 4
    trainer.dynamics_epochs = 2
    with torch.no_grad():
        trainer.actor.physics_residual.net[-1].bias.fill_(.3)
        # Deliberately different online/EMA weights catch loading the wrong branch.
        trainer.dynamics.gru.weight_hh.add_(.4)
        for parameter in trainer.actor_dynamics.system_id_logvar_head.parameters():
            parameter.zero_()
        trainer.actor_dynamics.system_id_logvar_head[-1].bias.fill_(-4.)
    rollout = trainer.collect(6)
    trainer.update(rollout)
    trainer.update_dynamics(rollout)
    # Exercise a genuinely active calibrated branch, not only a gated-off save.
    trainer.confidence.update(torch.zeros(32, 2), torch.full((32, 2), -4.),
                              torch.zeros(32, 2), torch.ones(32, dtype=torch.bool))
    path = trainer.save_checkpoint(tmp_path / "model_4.pt", iteration=4)
    return trainer, path


def test_standalone_checkpoint_preserves_nonzero_residual_ema_and_frozen_outputs(tmp_path, monkeypatch):
    trainer, path = _saved(tmp_path, monkeypatch)
    reference = FrozenPhysicsPolicy(copy.deepcopy(trainer.actor), copy.deepcopy(trainer.actor_dynamics),
        confidence_warmup_steps=trainer.confidence_warmup_steps, rollout_steps=6,
        calibration=copy.deepcopy(trainer.confidence))
    loaded = FrozenPhysicsPolicy.from_checkpoint(path)
    assert state_fingerprint(reference.actor, reference.dynamics) == state_fingerprint(loaded.actor, loaded.dynamics)
    assert not torch.equal(loaded.dynamics.gru.weight_hh, trainer.dynamics.gru.weight_hh)
    assert loaded.actor.physics_residual.net[-1].bias.abs().sum() > 0
    # No adjacent safetensors/config are needed by the schema-4 loader.
    moved = tmp_path / "standalone.pt"
    moved.write_bytes(path.read_bytes())
    alone = FrozenPhysicsPolicy.from_checkpoint(moved)
    a = evaluate_physics_policy(reference, _Task(boundaries=True), steps=15)
    b = evaluate_physics_policy(loaded, _Task(boundaries=True), steps=15)
    c = evaluate_physics_policy(alone, _Task(boundaries=True), steps=15)
    assert a == b == c
    assert b["weights_unchanged"] and b["terminations"] == b["timeouts"] == 1
    assert all(not p.requires_grad for p in loaded.actor.parameters())
    assert all(not p.requires_grad for p in loaded.dynamics.parameters())


def test_frozen_evaluation_never_uses_teacher_or_privileged_targets(tmp_path, monkeypatch):
    _, path = _saved(tmp_path, monkeypatch)
    policy = FrozenPhysicsPolicy.from_checkpoint(path)
    task = _Task()
    def forbidden(*args, **kwargs):
        raise AssertionError("privileged Teacher must not run in evaluation")
    task.physics_teacher_target = forbidden
    task.privileged_physics_observation = forbidden
    policy.dynamics.encode_privileged_teacher = forbidden
    policy.dynamics.teacher_encoder.forward = forbidden
    report = evaluate_physics_policy(policy, task, steps=10)
    assert report["weights_unchanged"]
    assert policy.age.tolist() == [10, 10, 10]


@pytest.mark.parametrize("missing", ["base_actor_state_dict", "physics_residual_state_dict",
                                    "actor_dynamics_state_dict", "dynamics_state_dict"])
def test_missing_branch_rejected_without_fallback(tmp_path, monkeypatch, missing):
    _, path = _saved(tmp_path, monkeypatch)
    payload = torch.load(path, weights_only=True)
    payload.pop(missing)
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="incomplete"):
        FrozenPhysicsPolicy.from_checkpoint(bad)


def test_legacy_checkpoint_requires_explicit_migration(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"physics_schema": 3}, path)
    with pytest.raises(ValueError, match="schema 4"):
        FrozenPhysicsPolicy.from_checkpoint(path)


def test_resume_restores_all_models_and_optimizer_then_continues(tmp_path, monkeypatch):
    trainer, path = _saved(tmp_path, monkeypatch)
    task = _real_trainer(True, monkeypatch).task
    resumed = HoloMotionPhysicsMJLabPPO.from_checkpoint(path, task)
    assert state_fingerprint(resumed.actor, resumed.critic, resumed.dynamics, resumed.actor_dynamics) == state_fingerprint(
        trainer.actor, trainer.critic, trainer.dynamics, trainer.actor_dynamics)
    assert resumed.actor_optimizer.state_dict()["state"].keys() == trainer.actor_optimizer.state_dict()["state"].keys()
    assert resumed.dynamics_optimizer.state_dict()["state"].keys() == trainer.dynamics_optimizer.state_dict()["state"].keys()
    rollout = resumed.collect(6)
    assert resumed.update(rollout).num_updates == 1
    assert resumed.update_dynamics(rollout).optimizer_steps == 2


def test_invalid_config_is_rejected_before_live_trainer_changes(tmp_path, monkeypatch):
    _, path = _saved(tmp_path, monkeypatch)
    target = _real_trainer(True, monkeypatch)
    before = state_fingerprint(target.actor, target.dynamics, target.actor_dynamics)
    with pytest.raises(ValueError, match="configuration differs"):
        target.load_optimizer_checkpoint(path)
    assert before == state_fingerprint(target.actor, target.dynamics, target.actor_dynamics)


def test_runtime_requires_action_feedback_and_real_initial_boundary(tmp_path, monkeypatch):
    _, path = _saved(tmp_path, monkeypatch)
    policy = FrozenPhysicsPolicy.from_checkpoint(path)
    task = _Task()
    obs = task.reset()
    with pytest.raises(ValueError, match="episode boundaries"):
        policy.act(obs.actor, task.deployable_physics(), torch.zeros(3, dtype=torch.bool))
    action = policy.act(obs.actor, task.deployable_physics(), torch.ones(3, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="executed action"):
        policy.act(obs.actor, task.deployable_physics(), torch.zeros(3, dtype=torch.bool))
    policy.record_action(action)
    with pytest.raises(RuntimeError, match=r"act\(\)"):
        policy.record_action(action)


def test_new_physics_save_requires_complete_architecture_metadata(tmp_path, monkeypatch):
    trainer = _real_trainer(True, monkeypatch)
    trainer._network_config = None
    with pytest.raises(ValueError, match="network_config"):
        trainer.save_checkpoint(tmp_path / "incomplete.pt", iteration=0)
    assert not (tmp_path / "incomplete.pt").exists()
    assert not (tmp_path / "incomplete").exists()


def test_legacy_training_resume_rejected_before_mutating_corrected_trainer(tmp_path, monkeypatch):
    original, path = _saved(tmp_path, monkeypatch)
    payload = torch.load(path, weights_only=True)
    payload["physics_schema"] = 3
    for key in ("physics_runtime_config", "base_actor_state_dict", "critic_state_dict",
                "physics_loss_weights", "ppo_config"):
        payload.pop(key)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    target = _real_trainer(True, monkeypatch)
    # Old runs reconstruct these from their explicit config.yaml + safetensors.
    target.actor.base_actor.load_state_dict(original.actor.base_actor.state_dict())
    target.critic.load_state_dict(original.critic.state_dict())
    target.dynamics_burn_in_steps = 2; target.dynamics_train_steps = 4; target.dynamics_epochs = 2
    target._policy_rollout_steps = 6
    before = state_fingerprint(target.actor, target.dynamics, target.confidence)
    with pytest.raises(ValueError, match="schema 5"):
        target.load_optimizer_checkpoint(legacy)
    assert before == state_fingerprint(target.actor, target.dynamics, target.confidence)


def test_task_contract_mismatch_rejected_before_evaluation_reset(tmp_path, monkeypatch):
    _, path = _saved(tmp_path, monkeypatch)
    policy = FrozenPhysicsPolicy.from_checkpoint(path)
    policy.task_contract = {"joint_names": ("wrong",)}
    task = _Task()
    with pytest.raises(ValueError, match="contract differs"):
        evaluate_physics_policy(policy, task, steps=3)
    assert task.full_resets == 0


def test_saved_calibration_really_opens_nonzero_residual_and_supports_ablation(tmp_path, monkeypatch):
    trainer, path = _saved(tmp_path, monkeypatch)
    full = FrozenPhysicsPolicy.from_checkpoint(path)
    disabled = FrozenPhysicsPolicy.from_checkpoint(path, ablation="no_physics")
    task = _Task()
    obs = task.reset()
    first = torch.ones(3, dtype=torch.bool)
    a = full.act(obs.actor, task.deployable_physics(), first)
    b = disabled.act(obs.actor, task.deployable_physics(), first)
    torch.testing.assert_close(a, b)  # No response evidence at the real first frame.
    full.record_action(a); disabled.record_action(b)
    a = full.act(obs.actor, task.deployable_physics(), ~first)
    b = disabled.act(obs.actor, task.deployable_physics(), ~first)
    assert (a-b).abs().max() > 1e-6
    torch.testing.assert_close(full.calibration.mse, trainer.confidence.mse)


@pytest.mark.parametrize("invalid", ["missing", "nonfinite"])
def test_invalid_calibration_checkpoint_is_rejected(tmp_path, monkeypatch, invalid):
    _, path = _saved(tmp_path, monkeypatch)
    payload = torch.load(path, weights_only=True)
    if invalid == "missing":
        payload.pop("confidence_state_dict")
    else:
        payload["confidence_state_dict"]["mse"].fill_(float("nan"))
    bad = tmp_path/"bad_calibration.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError):
        FrozenPhysicsPolicy.from_checkpoint(bad)


def test_legacy_schema4_keeps_original_architecture_and_confidence_formula(tmp_path, monkeypatch):
    _, path = _saved(tmp_path, monkeypatch)
    payload = torch.load(path, weights_only=True)
    payload["physics_schema"] = 4
    config = payload["physics_runtime_config"]
    config["schema"] = 1
    config["physics_actor_config"].pop("condition_on_base_action")
    payload["physics_residual_state_dict"].pop("action_condition.weight")
    config["dynamics_config"].pop("mask_unobserved_core")
    for key in ("dynamics_state_dict", "actor_dynamics_state_dict"):
        payload[key]["_extra_state"].pop("innovation_schema")
        payload[key].pop("encoder.availability.weight")
    config.pop("confidence_config")
    payload.pop("confidence_state_dict")
    legacy = tmp_path/"legacy4.pt"
    torch.save(payload, legacy)
    policy = FrozenPhysicsPolicy.from_checkpoint(legacy)
    assert policy.actor.physics_residual.action_condition is None
    assert policy.dynamics.encoder.availability is None
    assert not policy.dynamics.mask_unobserved_core
    logvar = torch.full((3, 2), torch.log(torch.tensor(1/3)))
    confidence = policy.calibration(logvar, torch.full((3,), 127), 128)
    torch.testing.assert_close(confidence, torch.full((3, 1), .84648172489))
