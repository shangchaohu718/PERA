import math
from dataclasses import replace

import torch

from unified_motion.data.motion_sampler import BodyReferenceWindow
from unified_motion.holomotion_mjlab_rewards import (
    K1_BODY_NAMES,
    HoloMotionK1CriticObservationBuilder,
    HoloMotionMJLabState,
    gravity_projection_distance,
    holomotion_k1_reward_terms,
)
from unified_motion.reference_observations import ReferenceWindow
from unified_motion.robot_contract import get_robot_contract


def _inputs():
    robot = get_robot_contract("k1_22dof")
    body_names = K1_BODY_NAMES
    batch, horizon, body_count = 2, 2, len(body_names)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    body_pos = torch.zeros(batch, body_count, 3)
    body_quat = identity.expand(batch, body_count, 4).clone()
    state = HoloMotionMJLabState(
        joint_names=robot.joint_names,
        body_names=body_names,
        joint_pos=torch.zeros(batch, 22),
        joint_vel=torch.zeros(batch, 22),
        root_pos_w=torch.zeros(batch, 3),
        root_quat_wxyz=identity.expand(batch, 4).clone(),
        root_lin_vel_w=torch.zeros(batch, 3),
        root_ang_vel_w=torch.zeros(batch, 3),
        body_pos_w=body_pos,
        body_quat_wxyz=body_quat,
        body_lin_vel_w=torch.zeros(batch, body_count, 3),
        body_ang_vel_w=torch.zeros(batch, body_count, 3),
        action=torch.zeros(batch, 22),
        previous_action=torch.zeros(batch, 22),
        soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).expand(batch, 22, 2).clone(),
    )
    reference = ReferenceWindow(
        root_name=robot.root_name,
        joint_names=robot.joint_names,
        joint_pos=torch.zeros(batch, horizon, 22),
        joint_vel=torch.zeros(batch, horizon, 22),
        root_pos_w=torch.zeros(batch, horizon, 3),
        root_quat_wxyz=identity.expand(batch, horizon, 4).clone(),
        root_lin_vel_w=torch.zeros(batch, horizon, 3),
        root_ang_vel_w=torch.zeros(batch, horizon, 3),
        time_offsets_s=torch.tensor([0.0, 0.02]).expand(batch, horizon).clone(),
        valid_mask=torch.ones(batch, horizon, dtype=torch.bool),
    )
    body_reference = BodyReferenceWindow(
        body_names=body_names,
        pos_w=body_pos[:, None].expand(batch, horizon, body_count, 3).clone(),
        quat_wxyz=body_quat[:, None].expand(batch, horizon, body_count, 4).clone(),
        lin_vel_w=torch.zeros(batch, horizon, body_count, 3),
        ang_vel_w=torch.zeros(batch, horizon, body_count, 3),
        valid_mask=torch.ones(batch, horizon, dtype=torch.bool),
    )
    return state, reference, body_reference


def test_identical_state_has_maximum_supported_tracking_reward():
    state, reference, body_reference = _inputs()
    terms = holomotion_k1_reward_terms(state, reference, body_reference)
    unit_terms = (
        "is_alive",
        "root_rel_keybodylink_pos_tracking_l2_exp",
        "root_rel_keybodylink_rot_tracking_l2_exp",
        "root_lin_vel_tracking_rel_ratio_exp",
        "root_ang_vel_tracking_rel_ratio_exp",
        "local_reward_point_body_pos_tracking_exp",
        "global_keybodylink_lin_vel_tracking_l2_exp",
        "global_keybodylink_ang_vel_tracking_l2_exp",
    )
    for name in unit_terms:
        assert torch.equal(terms[name], torch.ones(2))
    assert torch.equal(terms["action_rate_l2"], torch.zeros(2))
    assert torch.equal(terms["joint_pos_limits"], torch.zeros(2))
    assert torch.equal(terms["undesired_contacts"], torch.zeros(2))
    assert torch.allclose(terms["weighted_total"], torch.full((2,), 13.5))


def test_action_rate_and_joint_limit_match_official_penalties():
    state, reference, body_reference = _inputs()
    state.action[:, 0] = 2.0
    state.joint_pos[:, 1] = 1.25
    terms = holomotion_k1_reward_terms(state, reference, body_reference)
    assert torch.equal(terms["action_rate_l2"], torch.full((2,), 4.0))
    assert torch.equal(terms["joint_pos_limits"], torch.full((2,), 0.25))
    assert torch.allclose(terms["weighted_total"], torch.full((2,), 10.6))


def test_undesired_contact_counts_bodies_over_one_newton_across_substeps():
    state, reference, body_reference = _inputs()
    contact = torch.zeros(2, 19, 4, 3)
    contact[0, 3, 1, 0] = 1.01
    contact[0, 8, 2, 1] = 5.0
    contact[1, 5, 0, 2] = 1.0
    state = replace(state, undesired_contact_force_history=contact)

    terms = holomotion_k1_reward_terms(state, reference, body_reference)

    assert torch.equal(terms["undesired_contacts"], torch.tensor([2.0, 0.0]))
    assert torch.allclose(terms["weighted_total"], torch.tensor([13.3, 13.5]))


def test_gravity_projection_distance_detects_half_turn_roll():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    half = math.pi / 2
    roll = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    distance = gravity_projection_distance(roll, identity)
    assert torch.allclose(distance, torch.tensor([2.0]), atol=1e-6)


def test_critic_observation_matches_trained_843_value_contract():
    state, reference, body_reference = _inputs()
    reference.joint_pos = reference.joint_pos[:, :1].expand(-1, 12, -1).clone()
    reference.joint_vel = reference.joint_vel[:, :1].expand(-1, 12, -1).clone()
    reference.root_pos_w = reference.root_pos_w[:, :1].expand(-1, 12, -1).clone()
    reference.root_quat_wxyz = (
        reference.root_quat_wxyz[:, :1].expand(-1, 12, -1).clone()
    )
    reference.root_lin_vel_w = (
        reference.root_lin_vel_w[:, :1].expand(-1, 12, -1).clone()
    )
    reference.root_ang_vel_w = (
        reference.root_ang_vel_w[:, :1].expand(-1, 12, -1).clone()
    )
    reference.time_offsets_s = torch.arange(12).expand(2, -1) * 0.02
    reference.valid_mask = torch.ones(2, 12, dtype=torch.bool)
    body_reference = BodyReferenceWindow(
        body_names=body_reference.body_names,
        pos_w=body_reference.pos_w[:, :1].expand(-1, 12, -1, -1).clone(),
        quat_wxyz=body_reference.quat_wxyz[:, :1].expand(-1, 12, -1, -1).clone(),
        lin_vel_w=body_reference.lin_vel_w[:, :1].expand(-1, 12, -1, -1).clone(),
        ang_vel_w=body_reference.ang_vel_w[:, :1].expand(-1, 12, -1, -1).clone(),
        valid_mask=torch.ones(2, 12, dtype=torch.bool),
    )

    builder = HoloMotionK1CriticObservationBuilder(state.joint_names)
    observation = builder(state, reference, body_reference)
    assert observation.shape == (2, 843)
    assert torch.isfinite(observation).all()
    assert builder.term_slices["critic_global_anchor_diff"] == slice(22, 31)
    assert (
        builder.term_slices["critic_ref_motion_fut_heading_aligned_root_ang_vel"].stop
        == 843
    )
