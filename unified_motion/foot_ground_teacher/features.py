"""Explicit Teacher/Student feature contracts with no future-label inputs."""

from __future__ import annotations

import torch

from .mjlab_collection import FootGroundFrame


STUDENT_OBSERVATION_TERMS = (
    ("joint_position", 22),
    ("joint_velocity", 22),
    ("imu_angular_velocity_body", 3),
    ("imu_projected_gravity_body", 3),
    ("estimated_joint_torque", 22),
    ("previous_executed_action", 22),
)
STUDENT_OBSERVATION_DIM = sum(width for _, width in STUDENT_OBSERVATION_TERMS)

PRIVILEGED_CURRENT_TERMS = (
    ("root_linear_velocity_body", 3),
    ("root_height", 1),
    ("friction", 1),
    ("mass_scale", 1),
    ("applied_external_force_body", 3),
    ("applied_external_torque_body", 3),
)
PRIVILEGED_CURRENT_DIM = sum(width for _, width in PRIVILEGED_CURRENT_TERMS)

FOOT_CURRENT_TERMS = (
    ("contact", 1),
    ("contact_point_local", 3),
    ("contact_normal_local", 3),
    ("relative_velocity_normal", 1),
    ("relative_velocity_tangent", 2),
    ("normal_force", 1),
    ("tangential_force", 2),
    ("foot_load_ratio", 1),
    ("slip_speed", 1),
)
FOOT_CURRENT_DIM = sum(width for _, width in FOOT_CURRENT_TERMS)

# Causal control-plan context for V2.  These values describe the reference at
# time t, never the realized contact mode at t+1.
REFERENCE_INTENT_TERMS = (
    ("phase_sin", 1),
    ("phase_cos", 1),
    ("planned_anchor_left", 1),
    ("planned_anchor_right", 1),
    ("planned_slide_left", 1),
    ("planned_slide_right", 1),
)
REFERENCE_INTENT_DIM = sum(width for _, width in REFERENCE_INTENT_TERMS)


def _checked(name: str, value: torch.Tensor, lead: tuple[int, ...], width: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != lead + (width,):
        raise ValueError(f"{name} must have shape {lead + (width,)}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating point values")
    return value


def pack_student_observation(*, joint_position: torch.Tensor,
        joint_velocity: torch.Tensor, imu_angular_velocity_body: torch.Tensor,
        imu_projected_gravity_body: torch.Tensor,
        estimated_joint_torque: torch.Tensor,
        previous_executed_action: torch.Tensor) -> torch.Tensor:
    """Pack only quantities intended to be available to the future Student."""
    lead = joint_position.shape[:-1]
    values = (
        _checked("joint_position", joint_position, lead, 22),
        _checked("joint_velocity", joint_velocity, lead, 22),
        _checked("imu_angular_velocity_body", imu_angular_velocity_body, lead, 3),
        _checked("imu_projected_gravity_body", imu_projected_gravity_body, lead, 3),
        _checked("estimated_joint_torque", estimated_joint_torque, lead, 22),
        _checked("previous_executed_action", previous_executed_action, lead, 22),
    )
    return torch.cat(values, -1)


def pack_privileged_current(*, root_linear_velocity_body: torch.Tensor,
        root_height: torch.Tensor, friction: torch.Tensor, mass_scale: torch.Tensor,
        applied_external_force_body: torch.Tensor,
        applied_external_torque_body: torch.Tensor) -> torch.Tensor:
    """Pack current-time truth only; future contact/state is never accepted."""
    lead = root_linear_velocity_body.shape[:-1]
    values = (
        _checked("root_linear_velocity_body", root_linear_velocity_body, lead, 3),
        _checked("root_height", root_height, lead, 1),
        _checked("friction", friction, lead, 1),
        _checked("mass_scale", mass_scale, lead, 1),
        _checked("applied_external_force_body", applied_external_force_body, lead, 3),
        _checked("applied_external_torque_body", applied_external_torque_body, lead, 3),
    )
    return torch.cat(values, -1)


def pack_foot_current(frame: FootGroundFrame) -> torch.Tensor:
    """Pack a previously measured current frame, never the prediction target."""
    lead = frame.contact.shape
    if len(lead) < 2 or lead[-1] != 2:
        raise ValueError("foot contact frame must end in a two-foot axis")
    values = (
        frame.contact.to(frame.contact_point_local.dtype).unsqueeze(-1),
        frame.contact_point_local, frame.contact_normal,
        frame.relative_velocity_normal.unsqueeze(-1),
        frame.relative_velocity_tangent, frame.normal_force.unsqueeze(-1),
        frame.tangential_force, frame.foot_load_ratio.unsqueeze(-1),
        frame.slip_speed.unsqueeze(-1),
    )
    expected_lead = lead
    if any(value.shape[:-1] != expected_lead for value in values):
        raise ValueError("inconsistent current foot-frame tensors")
    packed = torch.cat(values, -1)
    if packed.shape != lead + (FOOT_CURRENT_DIM,) or not torch.isfinite(packed).all():
        raise ValueError("invalid packed current foot features")
    return packed


def pack_reference_intent(*, phase_sin: torch.Tensor, phase_cos: torch.Tensor,
        planned_anchor: torch.Tensor, planned_slide: torch.Tensor) -> torch.Tensor:
    """Pack the current reference plan without using future realized labels."""
    lead = phase_sin.shape[:-1]
    values = (
        _checked("phase_sin", phase_sin, lead, 1),
        _checked("phase_cos", phase_cos, lead, 1),
        _checked("planned_anchor", planned_anchor, lead, 2),
        _checked("planned_slide", planned_slide, lead, 2),
    )
    packed = torch.cat(values, -1)
    if packed.shape != lead + (REFERENCE_INTENT_DIM,):
        raise RuntimeError("reference intent layout mismatch")
    return packed


def simple_joint_velocity_delta(robot, *, control_dt_s: float,
                                joint_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Unconstrained inverse-dynamics baseline, excluding contact impulses.

    This baseline deliberately ignores floating-base coupling and contact.  The
    Teacher's residual therefore represents precisely the correction those
    omitted effects add over one control step.
    """
    if control_dt_s <= 0:
        raise ValueError("control_dt_s must be positive")
    view = robot.root_physx_view
    dof = int(robot.num_joints)
    mass = view.get_generalized_mass_matrices()
    coriolis_comp = view.get_coriolis_and_centrifugal_compensation_forces()
    gravity_comp = view.get_gravity_compensation_forces()
    if mass.shape[-1] == dof + 6:
        mass = mass[:, 6:, 6:]
        coriolis_comp = coriolis_comp[:, 6:]
        gravity_comp = gravity_comp[:, 6:]
    if mass.shape[-2:] != (dof, dof):
        raise RuntimeError("unexpected Isaac generalized mass-matrix shape")
    torque = robot.data.applied_torque
    if torque.shape[-1] != dof:
        raise RuntimeError("unexpected Isaac applied-torque shape")
    acceleration = torch.linalg.solve(
        mass, (torque + coriolis_comp + gravity_comp).unsqueeze(-1)
    ).squeeze(-1)
    if joint_ids is not None:
        if joint_ids.ndim != 1 or joint_ids.numel() != 22:
            raise ValueError("joint_ids must identify the 22 Actor joints")
        acceleration = acceleration[:, joint_ids]
    elif dof != 22:
        raise ValueError("joint_ids are required when the robot has more than 22 joints")
    return acceleration * control_dt_s
