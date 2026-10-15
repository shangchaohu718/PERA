"""Masked action-contrastive losses for same-state Isaac branches.

The branch contract is ``[base, plus, minus]``.  These losses supervise the
*change caused by the action*, not another absolute next-state prediction.
They are intentionally independent from the ordinary Teacher objective so the
coverage gate can remain a hard prerequisite for enabling them in training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CounterfactualLossConfig:
    tangent_weight: float = 1.0
    force_weight: float = 1.0
    tangent_direction_weight: float = 0.1
    tangent_scale_m_s: float = 0.1
    force_scale_n: float = 100.0
    tangent_material_floor_m_s: float = 0.005
    force_material_floor_n: float = 5.0

    def __post_init__(self) -> None:
        weights = (self.tangent_weight, self.force_weight,
                   self.tangent_direction_weight)
        if any(value < 0 for value in weights):
            raise ValueError("counterfactual loss weights must be non-negative")
        if self.tangent_scale_m_s <= 0 or self.force_scale_n <= 0:
            raise ValueError("counterfactual regression scales must be positive")
        if self.tangent_material_floor_m_s < 0 or self.force_material_floor_n < 0:
            raise ValueError("counterfactual material floors must be non-negative")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError("loss value and mask shapes differ")
    if mask.any():
        return value[mask].mean()
    # Retain a differentiable zero so an empty physical regime is safe.
    return value.sum() * 0.0


def action_contrastive_losses(
        predicted_tangent_velocity: torch.Tensor,
        predicted_normal_force: torch.Tensor,
        target_contact: torch.Tensor,
        target_tangent_velocity: torch.Tensor,
        target_normal_force: torch.Tensor,
        config: CounterfactualLossConfig | None = None,
) -> dict[str, torch.Tensor | int]:
    """Compare plus-minus action effects using only physically valid feet.

    Tangent velocity requires contact in all three branches.  Force uses feet
    contacting in at least one branch so action-induced loading/unloading is
    retained.  Material floors exclude responses indistinguishable from the
    collection campaign's numerical-noise floor.
    """
    config = config or CounterfactualLossConfig()
    if target_contact.ndim != 3 or target_contact.shape[1:] != (3, 2):
        raise ValueError("target_contact must be [N,3,2]")
    expected = target_contact.shape
    if predicted_normal_force.shape != expected or target_normal_force.shape != expected:
        raise ValueError("normal force tensors must be [N,3,2]")
    if predicted_tangent_velocity.shape != expected + (2,) or \
            target_tangent_velocity.shape != expected + (2,):
        raise ValueError("tangent velocity tensors must be [N,3,2,2]")
    for value in (predicted_tangent_velocity, predicted_normal_force,
                  target_tangent_velocity, target_normal_force):
        if not torch.isfinite(value).all():
            raise ValueError("counterfactual tensors must be finite")

    # Central contrast cancels common absolute-prediction bias and isolates the
    # response to the commanded plus/minus perturbation.
    tangent_prediction = (predicted_tangent_velocity[:, 1]
                          - predicted_tangent_velocity[:, 2])
    tangent_target = target_tangent_velocity[:, 1] - target_tangent_velocity[:, 2]
    force_prediction = predicted_normal_force[:, 1] - predicted_normal_force[:, 2]
    force_target = target_normal_force[:, 1] - target_normal_force[:, 2]

    persistent_contact = target_contact.all(dim=1)
    any_contact = target_contact.any(dim=1)
    tangent_material = persistent_contact & (
        torch.linalg.vector_norm(tangent_target, dim=-1)
        >= config.tangent_material_floor_m_s)
    force_material = any_contact & (force_target.abs() >= config.force_material_floor_n)

    tangent_element = F.smooth_l1_loss(
        tangent_prediction / config.tangent_scale_m_s,
        tangent_target / config.tangent_scale_m_s, reduction="none").mean(dim=-1)
    force_element = F.smooth_l1_loss(
        force_prediction / config.force_scale_n,
        force_target / config.force_scale_n, reduction="none")
    tangent_delta = _masked_mean(tangent_element, tangent_material)
    normal_force_delta = _masked_mean(force_element, force_material)

    direction_element = 1.0 - F.cosine_similarity(
        tangent_prediction, tangent_target, dim=-1, eps=1e-8)
    tangent_direction = _masked_mean(direction_element, tangent_material)
    total = (config.tangent_weight * tangent_delta
             + config.force_weight * normal_force_delta
             + config.tangent_direction_weight * tangent_direction)
    return {
        "tangent_delta": tangent_delta,
        "normal_force_delta": normal_force_delta,
        "tangent_direction": tangent_direction,
        "total": total,
        "tangent_material_count": int(tangent_material.sum()),
        "force_material_count": int(force_material.sum()),
    }
