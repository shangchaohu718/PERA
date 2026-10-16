import torch

from unified_motion.g1_teacher_actor import (
    TEACHER_GUIDED_ACTOR,
    TEACHER_TRAINING,
    G1TeacherActorConfig,
    G1TeacherActorModel,
)
from unified_motion.privileged_teacher import PrivilegedTeacherConfig


def fixture(phase=TEACHER_TRAINING):
    return G1TeacherActorModel(
        G1TeacherActorConfig(
            actor_width=16, actor_hidden_dim=24, latent_width=64,
            reference_steps=3),
        PrivilegedTeacherConfig(
            objective_version=5, critic_dim=1046,
            width=16, heads=2, layers=1,
            context_steps=4, train_envs=2, contact_slots=4),
        phase=phase,
    )


def teacher_batch(model, batch=2, steps=3):
    cfg = model.config
    teacher_dim = model.teacher_config.input_dim(
        cfg.robot.dof, cfg.system_id_dim)
    core_dim = 9 + 2 * cfg.robot.dof
    inputs = torch.randn(batch, steps, teacher_dim)
    starts = torch.zeros(batch, steps, dtype=torch.bool)
    starts[:, 0] = True
    actions = torch.randn(batch, steps, cfg.robot.dof)
    core = torch.randn(batch, steps, core_dim)
    core[..., 6:9] = torch.nn.functional.normalize(core[..., 6:9], dim=-1)
    return inputs, starts, actions, core


def test_default_g1_teacher_matches_official_holomotion_29dof():
    model = fixture()
    inputs, starts, actions, core = teacher_batch(model)
    output = model.teacher_forward(inputs, starts, actions, core)
    assert model.config.robot.dof == 29
    assert actions.shape[-1] == 29
    assert core.shape[-1] == 67
    assert "left_wrist_pitch_joint" in model.config.robot.joint_names
    assert "right_wrist_yaw_joint" in model.config.robot.joint_names
    assert output["latent"].shape == (2, 3, 64)
    assert output["next_core"].shape == (2, 3, 67)
    assert output["contact_logits"].shape == (2, 3, 4)
    assert output["contact_physics"].shape == (2, 3, 4, 4)
    # 1046-D official Critic observation + 126-D exact physical privilege.
    assert inputs.shape[-1] == 1172
    assert model.critic.input_dim == inputs.shape[-1]


def test_production_teacher_uses_equal_width_state_and_physics_branches():
    model = G1TeacherActorModel()
    teacher = model.teacher
    state_linear = teacher.state_reference_projection[1]
    physics_linear = teacher.physics_projection[1]
    fusion_linear = teacher.input_projection[1]
    assert (state_linear.in_features, state_linear.out_features) == (1046, 128)
    assert (physics_linear.in_features, physics_linear.out_features) == (126, 128)
    assert (fusion_linear.in_features, fusion_linear.out_features) == (256, 128)
    assert tuple((name, head[1].out_features)
                 for name, head in teacher.latent_head.items()) == (
        ("contact", 8), ("force", 16),
        ("kinematics", 16), ("dynamics", 24),
    )


def test_grouped_z64_routes_each_physics_task_to_declared_coordinates():
    model = fixture().train()
    inputs, starts, actions, core = teacher_batch(model)

    def active_groups(loss_key, selector=lambda value: value):
        model.zero_grad(set_to_none=True)
        output = model.teacher_forward(inputs, starts, actions, core)
        selector(output[loss_key]).square().mean().backward()
        active = set()
        for name, head in model.teacher.latent_head.items():
            grad = head[1].weight.grad
            if grad is not None and torch.count_nonzero(grad).item():
                active.add(name)
        return active

    assert active_groups("contact_logits") == {"contact"}
    assert active_groups(
        "contact_physics", lambda value: value[..., (0, 1, 3)]) == {"force"}
    assert active_groups(
        "contact_physics", lambda value: value[..., 2]) == {"kinematics"}
    assert active_groups("next_core") == {"kinematics", "dynamics"}
    assert active_groups("system_id") == {"dynamics"}


def test_both_teacher_input_branches_receive_training_gradients():
    model = fixture().train()
    inputs, starts, actions, core = teacher_batch(model)
    output = model.teacher_forward(inputs, starts, actions, core)
    loss = output["latent"].square().mean() + output["system_id"].square().mean()
    loss.backward()
    state_grad = model.teacher.state_reference_projection[1].weight.grad
    physics_grad = model.teacher.physics_projection[1].weight.grad
    fusion_grad = model.teacher.input_projection[1].weight.grad
    assert state_grad is not None and torch.count_nonzero(state_grad) > 0
    assert physics_grad is not None and torch.count_nonzero(physics_grad) > 0
    assert fusion_grad is not None and torch.count_nonzero(fusion_grad) > 0


def test_stage_one_updates_only_teacher():
    model = fixture().train()
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.01)
    inputs, starts, actions, core = teacher_batch(model)
    loss = model.teacher_forward(inputs, starts, actions, core)["next_core"].square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.teacher.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.actor.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.critic.parameters())
    optimizer.step()
    before = tuple(p.clone() for p in model.teacher_ema.parameters())
    model.update_teacher_ema()
    assert any(not torch.equal(old, new) for old, new in zip(
        before, model.teacher_ema.parameters(), strict=True))


def test_stage_two_freezes_teacher_and_guides_official_g1_actor():
    model = fixture(TEACHER_GUIDED_ACTOR).train()
    inputs, starts, _, _ = teacher_batch(model)
    cfg = model.config
    state = torch.randn(2, 3, cfg.state_dim)
    reference = torch.randn(2, 3, cfg.reference_steps, cfg.reference_dim)
    mask = torch.ones(2, 3, cfg.reference_steps, dtype=torch.bool)
    output, values = model.actor_critic_forward(
        state, reference, mask, inputs, starts
    )
    assert output.mean.shape == (2, 3, 29)
    assert output.std.shape == (2, 3, 29)
    assert output.joint_names == cfg.robot.joint_names
    assert values.shape == (2, 3, 1)
    loss = -output.distribution().log_prob(torch.zeros_like(output.mean)).mean()
    loss = loss + values.square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.actor.parameters())
    assert any(p.grad is not None for p in model.critic.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.teacher.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.teacher_ema.parameters())
    assert not model.teacher.training
    assert not model.teacher_ema.training


def test_teacher_and_critic_share_inputs_but_not_parameters_or_gradients():
    model = fixture(TEACHER_GUIDED_ACTOR).train()
    inputs, starts, _, _ = teacher_batch(model)
    with torch.no_grad():
        teacher_latent = model.frozen_teacher_latent(inputs, starts)
    values = model.critic_forward(inputs, starts)
    assert teacher_latent.shape[:2] == values.shape[:2]
    teacher_ids = {id(parameter) for parameter in model.teacher.parameters()}
    critic_ids = {id(parameter) for parameter in model.critic.parameters()}
    assert teacher_ids.isdisjoint(critic_ids)
    values.mean().backward()
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert any(parameter.grad is not None for parameter in model.critic.parameters())


def test_checkpoint_can_cross_only_the_declared_phase_boundary():
    teacher = fixture()
    guided = G1TeacherActorModel.from_checkpoint(
        teacher.checkpoint(), phase=TEACHER_GUIDED_ACTOR)
    assert guided.phase == TEACHER_GUIDED_ACTOR
    assert guided.config.robot_id == "g1_29dof"
    assert all(not p.requires_grad for p in guided.teacher.parameters())
    assert all(p.requires_grad for p in guided.actor.parameters())
    assert all(p.requires_grad for p in guided.critic.parameters())


def test_body_only_g1_variants_are_explicit_and_never_guessed_from_g1():
    config = G1TeacherActorConfig(robot_id="g1_23dof")
    assert config.robot.dof == 23
    try:
        G1TeacherActorConfig(robot_id="g1")
    except ValueError as error:
        assert all(name in str(error) for name in (
            "g1_23dof", "g1_29dof", "g1_43dof", "g1_37dof_isaac"))
    else:
        raise AssertionError("ambiguous g1 robot id was accepted")
