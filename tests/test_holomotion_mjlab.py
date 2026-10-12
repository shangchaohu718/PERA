import math

import torch

from unified_motion.holomotion_mjlab import (
    HoloMotionActorObservationBuilder,
    HoloMotionOnnxContract,
    HoloMotionRobotFrame,
)
from unified_motion.reference_observations import ReferenceWindow
from unified_motion.robot_contract import get_robot_contract


def _contract():
    names = get_robot_contract("k1_22dof").joint_names
    # Deliberately permute the exported order to exercise name-based mapping.
    exported = names[::2] + names[1::2]
    return HoloMotionOnnxContract(
        exported,
        tuple(20.0 for _ in range(22)),
        tuple(2.0 for _ in range(22)),
        tuple(float(i) / 100 for i in range(22)),
        tuple(0.5 for _ in range(22)),
    )


def _inputs(batch=2):
    robot = get_robot_contract("k1_22dof")
    dtype = torch.float32
    joint = torch.arange(batch * 22, dtype=dtype).reshape(batch, 22) / 100
    frame = HoloMotionRobotFrame(
        joint_names=robot.joint_names,
        joint_pos=joint,
        joint_vel=joint + 1,
        root_quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(batch, -1).clone(),
        root_ang_vel_b=torch.zeros(batch, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).expand(batch, -1).clone(),
    )
    # Current + 10 actor futures + one extra future for centered velocity.
    horizon = 12
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    reference = ReferenceWindow(
        root_name=robot.root_name,
        joint_names=robot.joint_names,
        joint_pos=joint[:, None, :].expand(batch, horizon, 22).clone(),
        joint_vel=torch.zeros(batch, horizon, 22),
        root_pos_w=torch.zeros(batch, horizon, 3),
        root_quat_wxyz=identity.expand(batch, horizon, 4).clone(),
        root_lin_vel_w=torch.zeros(batch, horizon, 3),
        root_ang_vel_w=torch.zeros(batch, horizon, 3),
        time_offsets_s=torch.arange(horizon).expand(batch, -1) * 0.02,
        valid_mask=torch.ones(batch, horizon, dtype=torch.bool),
    )
    return frame, reference


def test_contract_metadata_and_dimension():
    contract = _contract()
    metadata = {
        "joint_names": ",".join(contract.joint_names),
        "joint_stiffness": ",".join(map(str, contract.joint_stiffness)),
        "joint_damping": ",".join(map(str, contract.joint_damping)),
        "default_joint_pos": ",".join(map(str, contract.default_joint_pos)),
        "action_scale": ",".join(map(str, contract.action_scale)),
    }
    assert HoloMotionOnnxContract.from_metadata(metadata) == contract


def test_actor_observation_shape_identity_rotations_and_joint_order():
    contract = _contract()
    builder = HoloMotionActorObservationBuilder(contract)
    frame, reference = _inputs()
    obs = builder(frame, reference, torch.zeros(2, 22))
    assert obs.shape == (2, 506)
    assert torch.isfinite(obs).all()
    assert torch.allclose(obs[:, :3], torch.tensor([[0.0, 0.0, -1.0]]).expand(2, -1))
    # Current reference joint positions start at offset 9 and use ONNX order.
    expected = frame.joint_pos[
        :, [frame.joint_names.index(n) for n in contract.joint_names]
    ]
    assert torch.allclose(obs[:, 9:31], expected)
    # Identity robot/reference orientation gives yaw error [sin(0), cos(0)].
    assert torch.allclose(obs[:, 32:34], torch.tensor([[0.0, 1.0]]).expand(2, -1))


def test_action_is_reordered_by_name_not_position():
    contract = _contract()
    builder = HoloMotionActorObservationBuilder(contract)
    action = torch.arange(22, dtype=torch.float32).unsqueeze(0)
    mapped = builder.onnx_action_to_policy(action)
    expected = torch.tensor(
        [
            [
                contract.joint_names.index(name)
                for name in get_robot_contract("k1_22dof").joint_names
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.equal(mapped, expected)


def test_reference_tail_repeats_last_valid_frame():
    contract = _contract()
    builder = HoloMotionActorObservationBuilder(contract)
    frame, reference = _inputs(batch=1)
    reference.valid_mask[:, 5:] = False
    reference.joint_pos[:, 5:] = 999.0
    obs = builder(frame, reference, torch.zeros(1, 22))
    # Future block starts after 106 current values; first 220 values are 10x22 joints.
    future_joint = obs[:, 106:326].reshape(1, 10, 22)
    assert torch.allclose(future_joint[:, 4:], future_joint[:, 3:4].expand(-1, 6, -1))


def test_quarter_turn_yaw_error():
    contract = _contract()
    builder = HoloMotionActorObservationBuilder(contract)
    frame, reference = _inputs(batch=1)
    half = math.pi / 4
    reference.root_quat_wxyz[:] = torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)]
    )
    obs = builder(frame, reference, torch.zeros(1, 22))
    assert torch.allclose(obs[0, 32:34], torch.tensor([1.0, 0.0]), atol=1.0e-6)


def test_reference_velocity_is_derived_from_pose_context():
    contract = _contract()
    builder = HoloMotionActorObservationBuilder(contract)
    frame, reference = _inputs(batch=1)
    reference.root_pos_w[0, :, 0] = torch.arange(12) * 0.02
    reference.root_lin_vel_w.fill_(999.0)
    previous = torch.tensor([[-0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    obs = builder(
        frame,
        reference,
        torch.zeros(1, 22),
        previous_reference_root_pose=previous,
    )
    assert torch.allclose(obs[0, 3:6], torch.tensor([1.0, 0.0, 0.0]))
    future_lin = obs[:, 366:396].reshape(1, 10, 3)
    assert torch.allclose(
        future_lin, torch.tensor([[[1.0, 0.0, 0.0]]]).expand(1, 10, 3)
    )
