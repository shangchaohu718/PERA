import torch
import numpy as np

from unified_motion.foot_ground_teacher.evaluation import aggregate_metric_dicts
from unified_motion.foot_ground_teacher.isaac_collection import IsaacFootContactAccumulator
from unified_motion.foot_ground_teacher.features import (
    FOOT_CURRENT_DIM,
    PRIVILEGED_CURRENT_DIM,
    REFERENCE_INTENT_DIM,
    STUDENT_OBSERVATION_DIM,
    pack_privileged_current,
    pack_reference_intent,
    pack_student_observation,
)
from unified_motion.foot_ground_teacher.counterfactual import evaluate_action_counterfactual
from unified_motion.foot_ground_teacher.counterfactual_objective import (
    action_contrastive_losses,
)
from unified_motion.foot_ground_teacher.calibration import contact_calibration
from unified_motion.foot_ground_teacher.model import (
    FootGroundTeacher, FootGroundTeacherConfig,
)
from unified_motion.foot_ground_teacher.objective import (
    FootGroundLossConfig, gradient_audit, teacher_losses,
)
from unified_motion.foot_ground_teacher.physics import (
    aggregate_contacts, point_velocity, tangent_basis,
)
from unified_motion.foot_ground_teacher.schema import (
    FootGroundBatch, RegressionStatistics, deterministic_split,
)
from unified_motion.foot_ground_teacher.dataset import build_manifest, write_trajectory
from unified_motion.foot_ground_teacher.training import _masked_feature_columns
from unified_motion.foot_ground_teacher.reference import reference_intent_from_frames


def test_contact_point_velocity_tangent_frame_and_force_units():
    linear = torch.tensor([[1.0, 0.0, 0.0]])
    angular = torch.tensor([[0.0, 0.0, 2.0]])
    point = torch.tensor([[0.0, 1.0, 0.0]])
    torch.testing.assert_close(point_velocity(linear, angular, point,
                                              torch.zeros_like(point)),
                               torch.tensor([[-1.0, 0.0, 0.0]]))
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    t1, t2 = tangent_basis(normal)
    torch.testing.assert_close(t1, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(t2, torch.tensor([[0.0, 1.0, 0.0]]))
    result = aggregate_contacts(
        found=torch.tensor([[True, True]]),
        points=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        normals=normal[:, None].expand(-1, 2, -1),
        forces=torch.tensor([[[3.0, 0.0, 10.0], [0.0, 4.0, 30.0]]]),
        body_origin=torch.zeros(1, 3), body_linear_velocity=torch.zeros(1, 3),
        body_angular_velocity=torch.zeros(1, 3),
        ground_velocity=torch.zeros(1, 2, 3), heading=torch.tensor([[1.0, 0.0, 0.0]]),
        friction=torch.tensor([0.5]), physics_dt_s=0.002,
    )
    torch.testing.assert_close(result.point, torch.tensor([[0.75, 0.0, 0.0]]))
    torch.testing.assert_close(result.normal_force, torch.tensor([40.0]))
    torch.testing.assert_close(result.tangential_force, torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(result.impulse, torch.tensor([0.08]))
    torch.testing.assert_close(result.friction_utilization, torch.tensor([0.25]))


def test_explicit_student_and_current_privileged_feature_contracts():
    z = torch.zeros
    student = pack_student_observation(
        joint_position=z(2, 22), joint_velocity=z(2, 22),
        imu_angular_velocity_body=z(2, 3), imu_projected_gravity_body=z(2, 3),
        estimated_joint_torque=z(2, 22), previous_executed_action=z(2, 22),
    )
    privileged = pack_privileged_current(
        root_linear_velocity_body=z(2, 3), root_height=z(2, 1),
        friction=z(2, 1), mass_scale=z(2, 1),
        applied_external_force_body=z(2, 3), applied_external_torque_body=z(2, 3),
    )
    assert student.shape == (2, STUDENT_OBSERVATION_DIM) == (2, 94)
    assert privileged.shape == (2, PRIVILEGED_CURRENT_DIM) == (2, 12)
    assert FOOT_CURRENT_DIM == 15
    reference = pack_reference_intent(
        phase_sin=z(2, 1), phase_cos=torch.ones(2, 1),
        planned_anchor=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        planned_slide=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert reference.shape == (2, REFERENCE_INTENT_DIM) == (2, 6)


def test_reference_intent_uses_current_frame_not_next_realized_label():
    roles = np.asarray([[1, 2], [2, 1], [0, 1]], dtype=np.int8)
    result = reference_intent_from_frames(
        np.asarray([[0], [1]], dtype=np.int64), roles,
        cycle_start=0, cycle_frames=2)
    np.testing.assert_allclose(result[:, :2], [[0.0, 1.0], [0.0, -1.0]], atol=1e-6)
    np.testing.assert_array_equal(result[0, 2:], [1, 0, 0, 1])
    np.testing.assert_array_equal(result[1, 2:], [0, 1, 1, 0])


def test_masked_statistics_keep_left_and_right_channels_separate():
    value = torch.tensor([[[1.0, 100.0], [3.0, 200.0]]])
    mask = torch.tensor([[[True, False], [False, True]]])
    left, right = _masked_feature_columns(value, mask)
    torch.testing.assert_close(left, torch.tensor([1.0]))
    torch.testing.assert_close(right, torch.tensor([200.0]))


def test_metric_aggregation_pools_counts_not_trajectory_means():
    pooled = aggregate_metric_dicts([
        {"count": 1, "mae": 2.0, "rmse": 2.0},
        {"count": 3, "mae": 4.0, "rmse": 4.0},
    ])
    assert pooled["count"] == 4
    assert pooled["mae"] == 3.5
    torch.testing.assert_close(torch.tensor(pooled["rmse"]), torch.tensor(3.6055512))
    classification = aggregate_metric_dicts([
        {"count": 2, "tp": 1, "fp": 1, "fn": 0, "bce": .2,
         "brier": .1, "precision": .5, "recall": 1.0, "f1": 2 / 3},
        {"count": 8, "tp": 3, "fp": 0, "fn": 1, "bce": .4,
         "brier": .3, "precision": 1.0, "recall": .75, "f1": 6 / 7},
    ])
    assert classification["tp"] == 4 and classification["fp"] == 1
    assert classification["fn"] == 1 and abs(classification["bce"] - 0.36) < 1e-12


def _batch(batch=2, steps=5):
    z = torch.zeros
    contact = torch.ones(batch, steps, 2, dtype=torch.bool)
    valid = torch.ones(batch, steps, dtype=torch.bool)
    valid[:, -1] = False
    normal = z(batch, steps, 2, 3)
    normal[..., 2] = 1
    return FootGroundBatch(
        student_observation=z(batch, steps, 7), privileged_current=z(batch, steps, 5),
        executed_action=z(batch, steps, 22), foot_current=z(batch, steps, 2, 6),
        episode_start=torch.tensor([[1, 0, 0, 1, 0]] * batch, dtype=torch.bool),
        valid_transition=valid, contact_valid=contact, contact=contact,
        contact_mode=torch.ones(batch, steps, 2, dtype=torch.long),
        contact_point_local=z(batch, steps, 2, 3), contact_normal=normal,
        relative_velocity_normal=z(batch, steps, 2),
        relative_velocity_tangent=z(batch, steps, 2, 2), slip_speed=z(batch, steps, 2),
        normal_force=torch.ones(batch, steps, 2), tangential_force=z(batch, steps, 2, 2),
        contact_impulse=z(batch, steps, 2),
        foot_load_ratio=torch.full((batch, steps, 2), .5),
        delta_joint_velocity=z(batch, steps, 22),
        delta_root_angular_velocity=z(batch, steps, 3),
        joint_velocity_residual=z(batch, steps, 22),
        active_sliding=torch.zeros(batch, steps, 2, dtype=torch.bool),
        unexpected_slip=torch.zeros(batch, steps, 2, dtype=torch.bool),
        friction=torch.full((batch, steps), .6),
    ).validate()


def _statistics():
    widths = {
        "contact_point_local": 3, "relative_velocity_normal": 2,
        "relative_velocity_tangent": 2, "slip_speed": 2, "normal_force": 2,
        "tangential_force": 2, "contact_impulse": 2, "foot_load_ratio": 2,
        "delta_joint_velocity": 22, "delta_root_angular_velocity": 3,
        "joint_velocity_residual": 22,
    }
    return RegressionStatistics(
        {k: (0.0,) * v for k, v in widths.items()},
        {k: (1.0,) * v for k, v in widths.items()}, {k: 10 for k in widths})


def test_model_is_episode_causal_and_losses_are_auditable():
    torch.manual_seed(3)
    batch = _batch()
    model = FootGroundTeacher(FootGroundTeacherConfig(7, 5, 6, width=16,
                                                       foot_width=8, heads=2, layers=1))
    args = (batch.student_observation, batch.privileged_current,
            batch.executed_action, batch.foot_current, batch.episode_start)
    first = model(*args)
    changed = [value.clone() for value in args]
    changed[1][:, :3] += 1000
    second = model(*changed)
    # A new episode starts at index 3: earlier history cannot affect it.
    for key in first:
        torch.testing.assert_close(first[key][:, 3:], second[key][:, 3:])
    losses = teacher_losses(first, batch, _statistics(), FootGroundLossConfig())
    assert losses["total"].isfinite() and losses["total"] > 0
    audit = gradient_audit(model, losses)
    assert "gradient_norm/contact_group" in audit
    losses["total"].backward()
    assert model.foot_encoder[1].weight.grad is not None


def test_v2_task_profiles_remove_only_declared_groups():
    batch = _batch()
    model = FootGroundTeacher(FootGroundTeacherConfig(
        7, 5, 6, width=16, foot_width=8, heads=2, layers=1))
    output = model(batch.student_observation, batch.privileged_current,
                   batch.executed_action, batch.foot_current, batch.episode_start)
    a1 = teacher_losses(output, batch, _statistics(),
                        FootGroundLossConfig(task_profile="v2_a1"))
    core = teacher_losses(output, batch, _statistics(),
                          FootGroundLossConfig(task_profile="v2_a2_core3"))
    assert a1["geometry_group"] == 0
    torch.testing.assert_close(a1["kinematics_group"],
                               a1["relative_velocity_tangent"])
    assert a1["dynamics_group"] > 0
    assert core["geometry_group"] == 0 and core["dynamics_group"] == 0
    torch.testing.assert_close(core["force_group"], core["normal_force"])
    audit = gradient_audit(model, core)
    assert audit["gradient_norm/geometry_group"] == 0
    assert audit["gradient_norm/dynamics_group"] == 0


def test_reference_intent_model_contract_is_explicit_and_causal():
    batch = _batch()
    model = FootGroundTeacher(FootGroundTeacherConfig(
        7, 5, 6, reference_intent_dim=6,
        width=16, foot_width=8, heads=2, layers=1))
    reference = torch.zeros(2, 5, 6)
    output = model(batch.student_observation, batch.privileged_current,
                   batch.executed_action, batch.foot_current,
                   batch.episode_start, reference)
    assert output["contact_logits"].shape == (2, 5, 2)
    try:
        model(batch.student_observation, batch.privileged_current,
              batch.executed_action, batch.foot_current, batch.episode_start)
    except ValueError as error:
        assert "reference_intent" in str(error)
    else:
        raise AssertionError("intent-enabled model accepted a missing intent tensor")


def test_counterfactual_metrics_detect_action_ignored_teacher():
    contact = torch.ones(2, 3, 2, dtype=torch.bool)
    truth_v = torch.zeros(2, 3, 2, 2)
    truth_v[:, 1, :, 0] = .02
    truth_v[:, 2, :, 0] = -.02
    truth_f = torch.full((2, 3, 2), 100.0)
    truth_f[:, 1] += 20
    truth_f[:, 2] -= 20
    prediction_v = torch.zeros_like(truth_v)
    prediction_f = torch.full_like(truth_f, 100.0)
    result = evaluate_action_counterfactual(
        predicted_contact_probability=torch.ones(2, 3, 2),
        predicted_tangent_velocity=prediction_v,
        predicted_normal_force=prediction_f,
        target_contact=contact, target_tangent_velocity=truth_v,
        target_normal_force=truth_f, delta_rad=torch.full((2,), .02),
        processed_target_delta_rad=torch.tensor([[0.0, .02, -.02]] * 2),
        null_target_contact=contact[:, 0],
        null_target_tangent_velocity=truth_v[:, 0].clone(),
        null_target_normal_force=truth_f[:, 0].clone(),
    )
    assert result["branch"]["plus"]["tangent_velocity_delta"]["action_ignored_rate"] == 1
    assert result["branch"]["minus"]["normal_force_delta"]["action_ignored_rate"] == 1
    assert result["data_quality"]["null_contact_match_rate"] == 1
    assert result["data_quality"]["tangent_response_above_noise_rate"] == 1


def test_action_contrastive_loss_rewards_correct_action_response():
    contact = torch.zeros(1, 3, 2, dtype=torch.bool)
    contact[:, :, 0] = True
    truth_v = torch.zeros(1, 3, 2, 2)
    truth_v[:, 1, 0, 0] = .02
    truth_v[:, 2, 0, 0] = -.02
    truth_f = torch.full((1, 3, 2), 100.0)
    truth_f[:, 1, 0] = 110.0
    truth_f[:, 2, 0] = 90.0

    exact = action_contrastive_losses(
        truth_v, truth_f, contact, truth_v, truth_f)
    ignored = action_contrastive_losses(
        torch.zeros_like(truth_v), torch.full_like(truth_f, 100.0),
        contact, truth_v, truth_f)
    torch.testing.assert_close(exact["total"], torch.tensor(0.0))
    assert ignored["total"] > 0
    assert exact["tangent_material_count"] == 1
    assert exact["force_material_count"] == 1


def test_contact_calibration_exposes_mode_specific_probability_quality():
    probability = torch.tensor([.1, .9, .6, .8])
    truth = torch.tensor([False, True, True, True])
    mode = torch.tensor([0, 1, 2, 3])
    result = contact_calibration(
        probability, truth, mode, mode.clone(), torch.full((4,), .5), bins=2)
    assert result["count"] == 4 and result["brier"] < .1
    assert result["by_true_contact_mode"]["separation"]["predicted_contact_rate"] == 0
    assert result["by_true_contact_mode"]["impact"]["mode_head_accuracy"] == 1
    assert len(result["bins"]) == 2


def test_terminal_transition_is_masked_and_split_groups_are_stable():
    batch = _batch(batch=1)
    model = FootGroundTeacher(FootGroundTeacherConfig(7, 5, 6, width=16,
                                                       foot_width=8, heads=2, layers=1))
    output = model(batch.student_observation, batch.privileged_current,
                   batch.executed_action, batch.foot_current, batch.episode_start)
    baseline = teacher_losses(output, batch, _statistics())["total"]
    batch.delta_joint_velocity[:, -1] = 1e9
    changed = teacher_losses(output, batch, _statistics())["total"]
    torch.testing.assert_close(baseline, changed)
    meta = {"checkpoint_id": "a", "trajectory_id": "e7", "reference_id": "moon"}
    assert deterministic_split(meta) == deterministic_split(meta)
    assert deterministic_split(meta, heldout_checkpoints=("a",)) == "test"


def test_manifest_can_isolate_entire_checkpoint_sources(tmp_path):
    paths = []
    for index, checkpoint_id in enumerate(("train_a", "validation_b", "test_c")):
        path = tmp_path / f"trajectory_{index}.npz"
        write_trajectory(
            path, {"valid_transition": torch.ones(1, dtype=torch.bool).numpy()},
            {
                "checkpoint_id": checkpoint_id, "checkpoint_sha256": "0" * 64,
                "seed": index, "reference_id": "motion", "trajectory_id": f"t{index}",
                "start_phase": 0.0, "friction": 0.5, "mass_scale": 1.0,
                "disturbance": {"type": "none"}, "termination_reason": "time_out",
                "control_dt_s": 0.02,
            },
        )
        paths.append(path)
    manifest = build_manifest(
        paths, heldout_checkpoints=("test_c",),
        validation_checkpoints=("validation_b",))
    split_by_checkpoint = {
        row["metadata"]["checkpoint_id"]: row["split"] for row in manifest["records"]}
    assert split_by_checkpoint == {
        "train_a": "train", "validation_b": "validation", "test_c": "test"}
    assert manifest["checkpoint_source_exclusive"]


class _FakeContactView:
    def __init__(self, normal_force, point, normal, friction, *, found=None):
        self.normal_force = normal_force
        self.point = point
        self.normal = normal
        self.friction = friction
        self.found = normal_force != 0 if found is None else found

    def get_contact_data(self, dt):
        del dt
        if not self.found:
            count = torch.tensor([[0]])
        else:
            count = torch.tensor([[1]])
        return (
            torch.tensor([[self.normal_force], [0.0]]),
            torch.tensor([self.point, [0.0, 0.0, 0.0]]),
            torch.tensor([self.normal, [0.0, 0.0, 1.0]]),
            torch.zeros(2, 1), count, torch.tensor([[0]]),
        )

    def get_friction_data(self, dt):
        del dt
        if not self.found:
            count = torch.tensor([[0]])
        else:
            count = torch.tensor([[1]])
        return (torch.tensor([self.friction, [0.0, 0.0, 0.0]]),
                torch.tensor([self.point, [0.0, 0.0, 0.0]]),
                count, torch.tensor([[0]]))


class _FakeSensor:
    def __init__(self, view):
        self.contact_physx_view = view


def test_isaac_substeps_use_real_point_velocity_force_and_integrated_impulse():
    left = _FakeContactView(0.0, [1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                            [10.0, 0.0, 0.0], found=False)
    right = _FakeContactView(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0])
    collector = IsaacFootContactAccumulator(
        (_FakeSensor(left), _FakeSensor(right)), batch=1, slots=2,
        impact_impulse_threshold_n_s=1.0,
    )
    collector.begin_control_step()
    collector.sample_substep(.01)
    left.normal_force = 50.0
    left.found = True
    collector.sample_substep(.01)
    left.normal_force = 100.0
    collector.sample_substep(.01)
    identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2])
    zero = torch.zeros(1, 2, 3)
    angular = zero.clone()
    angular[..., 2] = 1.0
    heading = zero.clone()
    heading[..., 0] = 1.0
    frame = collector.finish_control_step(
        body_position_w=zero, body_quaternion_wxyz=identity,
        body_linear_velocity_w=zero, body_angular_velocity_w=angular,
        heading_w=heading, friction=torch.tensor([.5]),
        active_sliding=torch.tensor([[True, False]]),
    )
    assert frame.contact.tolist() == [[True, False]]
    torch.testing.assert_close(frame.contact_point_local[0, 0],
                               torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(frame.contact_normal[0, 0],
                               torch.tensor([0.0, 0.0, 1.0]))
    # omega=(0,0,1), r=(1,0,0) gives point velocity (0,1,0).
    torch.testing.assert_close(frame.relative_velocity_tangent[0, 0],
                               torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(frame.normal_force[0, 0], torch.tensor(100.0))
    torch.testing.assert_close(frame.tangential_force[0, 0],
                               torch.tensor([10.0, 0.0]))
    # 50 N * 0.01 s + 100 N * 0.01 s.
    torch.testing.assert_close(frame.contact_impulse[0, 0], torch.tensor(1.5))
    assert frame.contact_mode[0, 0].item() == 3
    assert frame.raw_substep_found.shape == (1, 2, 3, 2)

    # The same high load on an already-contacting foot is sliding, not impact.
    collector.begin_control_step()
    collector.sample_substep(.01)
    collector.sample_substep(.01)
    sustained = collector.finish_control_step(
        body_position_w=zero, body_quaternion_wxyz=identity,
        body_linear_velocity_w=zero, body_angular_velocity_w=angular,
        heading_w=heading, friction=torch.tensor([.5]),
        active_sliding=torch.tensor([[True, False]]),
    )
    assert sustained.contact_mode[0, 0].item() == 2


def test_zero_impulse_geometric_patch_is_raw_only_not_stable_contact():
    geometric = _FakeContactView(
        0.0, [0.1, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], found=True)
    absent = _FakeContactView(
        0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    collector = IsaacFootContactAccumulator(
        (_FakeSensor(geometric), _FakeSensor(absent)), batch=1, slots=2)
    collector.begin_control_step()
    collector.sample_substep(.005)
    identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2])
    zero = torch.zeros(1, 2, 3)
    heading = zero.clone()
    heading[..., 0] = 1.0
    frame = collector.finish_control_step(
        body_position_w=zero, body_quaternion_wxyz=identity,
        body_linear_velocity_w=zero, body_angular_velocity_w=zero,
        heading_w=heading, friction=torch.tensor([.5]),
        active_sliding=torch.tensor([[False, False]]),
    )
    assert frame.raw_found[0, 0, 0]
    assert not frame.contact[0, 0]
    assert frame.contact_mode[0, 0].item() == 0
