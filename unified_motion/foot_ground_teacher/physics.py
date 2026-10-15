"""Contact-point kinematics, stable tangent frames, and force aggregation."""
from __future__ import annotations

from dataclasses import dataclass
import torch


def point_velocity(linear_velocity: torch.Tensor, angular_velocity: torch.Tensor,
                   point: torch.Tensor, body_origin: torch.Tensor) -> torch.Tensor:
    """World velocity of a rigid-body point: ``v + omega cross r``."""
    return linear_velocity + torch.cross(angular_velocity, point - body_origin, dim=-1)


def tangent_basis(normal: torch.Tensor, heading: torch.Tensor | None = None,
                  eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic right-handed tangent basis anchored to robot heading.

    ``t1`` is projected robot-forward.  Near degeneracy it falls back to the
    least-aligned world axis, preventing frame flips on horizontal/steep ground.
    """
    n = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(eps)
    if heading is None:
        heading = torch.zeros_like(n)
        heading[..., 0] = 1.0
    projected = heading - (heading * n).sum(-1, keepdim=True) * n
    bad = torch.linalg.vector_norm(projected, dim=-1, keepdim=True) < 1.0e-5
    axes = torch.eye(3, device=n.device, dtype=n.dtype)
    index = n.abs().argmin(-1)
    fallback = axes[index]
    fallback = fallback - (fallback * n).sum(-1, keepdim=True) * n
    t1 = torch.where(bad, fallback, projected)
    t1 = t1 / torch.linalg.vector_norm(t1, dim=-1, keepdim=True).clamp_min(eps)
    t2 = torch.cross(n, t1, dim=-1)
    return t1, t2


def project_contact_vector(vector: torch.Tensor, normal: torch.Tensor,
                           heading: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed normal scalar and two tangent coordinates."""
    t1, t2 = tangent_basis(normal, heading)
    return (vector * normal).sum(-1), torch.stack(((vector * t1).sum(-1),
                                                   (vector * t2).sum(-1)), -1)


@dataclass(frozen=True)
class AggregatedFootContact:
    found: torch.Tensor
    point: torch.Tensor
    normal: torch.Tensor
    relative_velocity_normal: torch.Tensor
    relative_velocity_tangent: torch.Tensor
    normal_force: torch.Tensor
    tangential_force: torch.Tensor
    impulse: torch.Tensor
    slip_speed: torch.Tensor
    friction_utilization: torch.Tensor


def aggregate_contacts(*, found: torch.Tensor, points: torch.Tensor,
                       normals: torch.Tensor, forces: torch.Tensor,
                       body_origin: torch.Tensor, body_linear_velocity: torch.Tensor,
                       body_angular_velocity: torch.Tensor, ground_velocity: torch.Tensor,
                       heading: torch.Tensor, friction: torch.Tensor,
                       physics_dt_s: float, eps: float = 1.0e-6) -> AggregatedFootContact:
    """Aggregate raw solver slots for one foot while callers retain raw arrays.

    Shapes end in ``[contacts]`` or ``[contacts,3]``. Forces are instantaneous
    Newtons, summed vectorially. The auxiliary impulse is ``sum(force)*dt`` in
    N*s, never a force mislabeled as an impulse. Contact point and normal use
    compressive-normal-force weights (uniform fallback for a zero-force hit).
    """
    if physics_dt_s <= 0:
        raise ValueError("physics_dt_s must be positive")
    mask = found.to(points.dtype)
    unit = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True).clamp_min(eps)
    compression = (forces * unit).sum(-1).abs() * mask
    count = mask.sum(-1, keepdim=True)
    weights = compression / compression.sum(-1, keepdim=True).clamp_min(eps)
    uniform = mask / count.clamp_min(1)
    weights = torch.where((compression.sum(-1, keepdim=True) > eps), weights, uniform)
    aggregate_point = (points * weights.unsqueeze(-1)).sum(-2)
    aggregate_normal = (unit * weights.unsqueeze(-1)).sum(-2)
    aggregate_normal = aggregate_normal / torch.linalg.vector_norm(
        aggregate_normal, dim=-1, keepdim=True).clamp_min(eps)
    any_found = found.any(-1)
    aggregate_normal = torch.where(any_found.unsqueeze(-1), aggregate_normal,
                                   torch.zeros_like(aggregate_normal))
    point_vel = point_velocity(body_linear_velocity.unsqueeze(-2),
                               body_angular_velocity.unsqueeze(-2), points,
                               body_origin.unsqueeze(-2))
    relative = (point_vel - ground_velocity) * mask.unsqueeze(-1)
    relative = (relative * weights.unsqueeze(-1)).sum(-2)
    net_force = (forces * mask.unsqueeze(-1)).sum(-2)
    normal_v, tangent_v = project_contact_vector(relative, aggregate_normal, heading)
    normal_f_signed, tangent_f = project_contact_vector(net_force, aggregate_normal, heading)
    normal_f = normal_f_signed.abs()
    slip = torch.linalg.vector_norm(tangent_v, dim=-1)
    utilization = torch.linalg.vector_norm(tangent_f, dim=-1) / (
        friction * normal_f + eps)
    zero = torch.zeros_like
    return AggregatedFootContact(
        any_found,
        torch.where(any_found.unsqueeze(-1), aggregate_point, zero(aggregate_point)),
        aggregate_normal,
        torch.where(any_found, normal_v, zero(normal_v)),
        torch.where(any_found.unsqueeze(-1), tangent_v, zero(tangent_v)),
        torch.where(any_found, normal_f, zero(normal_f)),
        torch.where(any_found.unsqueeze(-1), tangent_f, zero(tangent_f)),
        torch.where(any_found, normal_f * physics_dt_s, zero(normal_f)),
        torch.where(any_found, slip, zero(slip)),
        torch.where(any_found, utilization, zero(utilization)),
    )
