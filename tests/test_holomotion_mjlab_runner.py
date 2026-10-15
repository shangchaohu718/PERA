import copy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from unified_motion.holomotion_mjlab_ppo import HoloMotionMJLabPPO
from unified_motion.holomotion_physics_actor import (
    HoloMotionPhysicsActor, HoloMotionPhysicsActorConfig,
)
from unified_motion.models.physics import DynamicsModel
from unified_motion.robot_contract import get_robot_contract
from unified_motion.holomotion_mjlab_runner import HoloMotionRunnerConfig


def _minimal_trainer() -> HoloMotionMJLabPPO:
    trainer = HoloMotionMJLabPPO.__new__(HoloMotionMJLabPPO)
    trainer.actor = torch.nn.Linear(3, 2)
    trainer.critic = torch.nn.Linear(3, 1)
    trainer.actor_optimizer = torch.optim.AdamW(trainer.actor.parameters())
    trainer.critic_optimizer = torch.optim.AdamW(trainer.critic.parameters())
    trainer._observation_normalization = dict(schema=1, actor=False, critic=False)
    return trainer


def test_checkpoint_layout_is_compatible_with_original_loader(tmp_path: Path):
    trainer = _minimal_trainer()
    checkpoint = trainer.save_checkpoint(
        tmp_path / "model_7.pt", iteration=7, infos={"reward": 1.0}
    )

    actor = load_file(str(tmp_path / "model_7" / "actor" / "model.safetensors"))
    critic = load_file(str(tmp_path / "model_7" / "critic" / "model.safetensors"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(actor) == set(trainer.actor.state_dict())
    assert set(critic) == set(trainer.critic.state_dict())
    assert payload["iter"] == 7
    assert payload["infos"] == {"reward": 1.0}
    assert "actor_optimizer_state_dict" in payload
    assert "critic_optimizer_state_dict" in payload


def test_checkpoint_refuses_to_overwrite_existing_target(tmp_path: Path):
    trainer = _minimal_trainer()
    target = trainer.save_checkpoint(tmp_path / "model_1.pt", iteration=1)

    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(target, iteration=1)


def test_physics_checkpoint_keeps_original_actor_layout_and_extra_state(tmp_path: Path, monkeypatch):
    from test_holomotion_rollout_boundaries import _real_trainer
    trainer = _real_trainer(True, monkeypatch)
    base_actor = trainer.actor.base_actor
    checkpoint = trainer.save_checkpoint(tmp_path / "model_2.pt", iteration=2)
    actor = load_file(str(tmp_path / "model_2" / "actor" / "model.safetensors"))
    payload = torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert set(actor)==set(base_actor.state_dict())
    assert payload["physics_schema"]==5
    assert "physics_residual_state_dict" in payload
    assert "dynamics_state_dict" in payload
    assert "actor_dynamics_state_dict" in payload
    assert payload["physics_training_config"]["dynamics_train_steps"]==128
    assert payload["physics_training_config"]["dynamics_epochs"]==4


def test_runner_configuration_rejects_unbounded_zero_iterations():
    with pytest.raises(ValueError, match="num_iterations"):
        HoloMotionRunnerConfig(num_iterations=0)
