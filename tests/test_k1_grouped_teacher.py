from pathlib import Path

import torch
import yaml

from unified_motion.privileged_teacher import (
    GROUPED_PHYSICS_LATENT,
    PrivilegedTeacherConfig,
    PrivilegedTransformerTeacher,
)
from unified_motion.holomotion_mjlab_ppo import HoloMotionPPOConfig
from unified_motion.k1_policy_backends import K1ActorConfig
from unified_motion.k1_training import create_custom_k1_algorithm, create_k1_trainer
from unified_motion.robot_contract import get_robot_contract
from test_k1_actor_switching import task


ROOT = Path(__file__).resolve().parents[1]


def production_teacher():
    values = yaml.safe_load((
        ROOT / "configs/teachers/k1_transformer_shared_critic_grouped_z64.yaml"
    ).read_text(encoding="utf-8"))
    config = PrivilegedTeacherConfig(**values)
    model = PrivilegedTransformerTeacher(
        get_robot_contract("k1_22dof"), 64, 2, config
    )
    return config, model


def test_k1_grouped_teacher_contract_is_941_to_routed_z64():
    config, teacher = production_teacher()
    assert config.input_dim(22, 2) == 941
    assert config.critic_privileged_dim(2) == 98
    assert config.latent_groups == GROUPED_PHYSICS_LATENT
    assert config.parameter_reconstruction_weight == 0.1
    assert tuple(teacher.latent_slices) == (
        "contact", "force", "kinematics", "dynamics"
    )
    assert teacher.state_reference_projection[1].in_features == 843
    assert teacher.physics_projection[1].in_features == 98

    inputs = torch.randn(2, 3, 941)
    starts = torch.zeros(2, 3, dtype=torch.bool)
    starts[:, 0] = True
    actions = torch.randn(2, 3, 22)
    core = torch.randn(2, 3, 53)
    core[..., 6:9] = torch.nn.functional.normalize(core[..., 6:9], dim=-1)
    output = teacher(inputs, starts, actions, core)

    assert output["latent"].shape == (2, 3, 64)
    assert output["next_core"].shape == (2, 3, 53)
    assert output["contact_logits"].shape == (2, 3, 2)
    assert output["contact_physics"].shape == (2, 3, 2, 4)
    assert output["contact_mode_logits"].shape == (2, 3, 2, 3)
    assert output["system_id"].shape == (2, 3, 2)


def test_k1_grouped_teacher_routes_losses_to_declared_groups():
    _, teacher = production_teacher()
    teacher.train()
    inputs = torch.randn(1, 2, 941)
    starts = torch.tensor([[True, False]])
    actions = torch.randn(1, 2, 22)
    core = torch.randn(1, 2, 53)
    core[..., 6:9] = torch.nn.functional.normalize(core[..., 6:9], dim=-1)

    def active_groups(key, select=lambda value: value):
        teacher.zero_grad(set_to_none=True)
        output = teacher(inputs, starts, actions, core)
        select(output[key]).square().mean().backward()
        return {
            name for name, head in teacher.latent_head.items()
            if head[1].weight.grad is not None
            and torch.count_nonzero(head[1].weight.grad).item()
        }

    assert active_groups("contact_logits") == {"contact"}
    assert active_groups(
        "contact_physics", lambda value: value[..., (0, 1, 3)]
    ) == {"force"}
    assert active_groups(
        "contact_physics", lambda value: value[..., 2]
    ) == {"kinematics"}
    assert active_groups("next_core") == {"kinematics", "dynamics"}
    assert active_groups("system_id") == {"dynamics"}


def test_k1_trainer_builds_grouped_z64_from_saved_config_contract():
    values = yaml.safe_load((
        ROOT / "configs/teachers/k1_transformer_shared_critic_grouped_z64.yaml"
    ).read_text(encoding="utf-8"))
    values.update(
        width=16,
        heads=2,
        layers=1,
        context_steps=10,
        train_envs=2,
        warmup_updates=1,
        minimum_validation_samples=2,
    )
    algorithm = create_custom_k1_algorithm(K1ActorConfig(
        backbone="mlp", d_model=16, hidden_dim=24, n_layers=1
    ))
    trainer = create_k1_trainer(
        algorithm,
        task(),
        physics=True,
        teacher=True,
        privileged_teacher_config=values,
        dynamics_config=dict(
            physics_width=4, memory_width=4, latent_width=64, hidden_dim=8
        ),
        ppo_config=HoloMotionPPOConfig(
            num_learning_epochs=1,
            num_mini_batches=2,
            aux_state_pred_enabled=False,
            inactive_expert_margin_enabled=False,
        ),
        dynamics_burn_in_steps=2,
        dynamics_train_steps=4,
        dynamics_epochs=1,
    )
    assert trainer.privileged_teacher.config.latent_groups == GROUPED_PHYSICS_LATENT
    assert trainer.dynamics.latent_width == 64
    assert trainer.privileged_teacher.encode(
        torch.randn(3, 2, 941),
        torch.tensor([[True, False]] * 3),
    ).shape == (3, 2, 64)

    trainer.prepare_teacher_only()
    metrics = trainer.update_teacher_only(trainer.collect(10))
    assert metrics["teacher_updates"] == 1
    assert metrics["teacher_multistep_loss"] > 0
