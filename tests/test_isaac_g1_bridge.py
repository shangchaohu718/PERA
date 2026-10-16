from types import SimpleNamespace
import sys

import pytest
import torch

from unified_motion.envs.isaac_g1 import (
    ISAAC_G1_CONTACT_SLOTS,
    IsaacG1Bridge,
    isaac_g1_articulation_cfg,
)
from unified_motion.robot_contract import get_robot_contract


class FakeActionManager:
    def __init__(self, term, previous):
        self.term = term
        self.prev_action = previous

    def get_term(self, name):
        if name != "joint_pos":
            raise KeyError(name)
        return self.term


class FakeEnv:
    def __init__(self):
        contract = get_robot_contract("g1_29dof")
        self.policy_names = contract.joint_names
        self.robot_names = tuple(reversed(self.policy_names))
        batch, dof = 2, contract.dof
        joint_pos = torch.arange(batch * dof, dtype=torch.float32).reshape(batch, dof)
        default = torch.arange(dof, dtype=torch.float32)[None].expand(batch, -1) / 10
        bodies = ("pelvis", "left_foot", "right_foot", "left_wrist", "right_wrist")
        data = SimpleNamespace(
            joint_pos=joint_pos,
            default_joint_pos=default,
            joint_vel=joint_pos + 100,
            root_ang_vel_b=torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
            projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
            root_lin_vel_b=torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
            body_pos_w=torch.randn(batch, len(bodies), 3),
            body_quat_w=torch.randn(batch, len(bodies), 4),
            body_link_lin_vel_w=torch.randn(batch, len(bodies), 3),
            body_ang_vel_w=torch.randn(batch, len(bodies), 3),
        )
        robot = SimpleNamespace(data=data, joint_names=self.robot_names, body_names=bodies)
        self.scene = {"robot": robot}
        action_ids = torch.arange(dof - 1, -1, -1)
        previous = torch.arange(batch * dof, dtype=torch.float32).reshape(batch, dof) + 200
        self.action_manager = FakeActionManager(
            SimpleNamespace(_joint_ids=action_ids), previous)
        self.stepped = None

    def step(self, action):
        self.stepped = action
        return "transition"


def contact_names():
    return dict(zip(
        ISAAC_G1_CONTACT_SLOTS,
        ("left_foot", "right_foot", "left_wrist", "right_wrist"),
        strict=True,
    ))


def test_isaac_bridge_reorders_official_29dof_joints():
    env = FakeEnv()
    bridge = IsaacG1Bridge(env, contact_body_names=contact_names())
    state = bridge.actor_state()
    assert bridge.contract.dof == 29
    assert state.shape == (2, 93)
    expected_offset = bridge.contract.to_policy_order(
        env.scene["robot"].data.joint_pos - env.scene["robot"].data.default_joint_pos,
        env.robot_names,
    )
    torch.testing.assert_close(state[:, 6:35], expected_offset)
    assert "left_wrist_pitch_joint" in bridge.contract.joint_names
    assert "right_wrist_yaw_joint" in bridge.contract.joint_names


def test_actor_action_is_mapped_to_isaac_action_term_order():
    env = FakeEnv()
    bridge = IsaacG1Bridge(env)
    action = torch.arange(58, dtype=torch.float32).reshape(2, 29)
    assert bridge.step(action) == "transition"
    expected = bridge.contract.to_simulator_order(
        action, bridge.indices.action_joint_names)
    torch.testing.assert_close(env.stepped, expected)


def test_contact_slots_are_explicit_feet_then_hands():
    bridge = IsaacG1Bridge(FakeEnv(), contact_body_names=contact_names())
    state = bridge.contact_body_state()
    assert bridge.indices.contact_body_names == (
        "left_foot", "right_foot", "left_wrist", "right_wrist")
    assert all(value.shape[:2] == (2, 4) for value in state.values())


def test_bridge_rejects_wrong_asset_or_partial_action_term():
    env = FakeEnv()
    env.scene["robot"].joint_names = env.scene["robot"].joint_names[:-1]
    with pytest.raises(ValueError, match="关节不匹配"):
        IsaacG1Bridge(env)
    env = FakeEnv()
    env.action_manager.term._joint_ids = torch.arange(28)
    with pytest.raises(ValueError, match="关节不匹配"):
        IsaacG1Bridge(env)


def test_articulation_factory_uses_official_holomotion_builder(monkeypatch):
    built = SimpleNamespace(replace=lambda **kwargs: SimpleNamespace(**kwargs))
    scene_functions = SimpleNamespace(build_robot_config=lambda config, **kwargs: built)
    module = SimpleNamespace(SceneFunctions=scene_functions)
    monkeypatch.setitem(
        sys.modules,
        "holomotion.src.env.isaaclab_components.isaaclab_scene",
        module,
    )
    config = isaac_g1_articulation_cfg(object(), prim_path="/World/G1")
    assert config.prim_path == "/World/G1"
