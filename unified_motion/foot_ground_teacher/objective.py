"""Masked, standardized multi-task losses and gradient-conflict audit."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F

from .schema import FootGroundBatch, RegressionStatistics


TASK_PROFILE_ACTIVE_LOSSES = {
    "v1": (
        "contact_bce", "contact_mode_ce", "contact_point", "contact_normal",
        "relative_velocity_normal", "relative_velocity_tangent", "slip_speed",
        "normal_force", "tangential_force", "contact_impulse", "foot_load_ratio",
        "delta_joint_velocity", "delta_root_angular_velocity",
        "joint_velocity_residual",
    ),
    "v2_a1": (
        "contact_bce", "contact_mode_ce", "relative_velocity_tangent",
        "normal_force", "tangential_force", "contact_impulse", "foot_load_ratio",
        "delta_joint_velocity", "delta_root_angular_velocity",
        "joint_velocity_residual",
    ),
    "v2_a2_core3": (
        "contact_bce", "contact_mode_ce", "relative_velocity_tangent", "normal_force",
    ),
}


@dataclass(frozen=True)
class FootGroundLossConfig:
    task_profile: str = "v1"
    contact_weight: float = 0.1
    geometry_weight: float = 0.1
    kinematics_weight: float = 1.0
    force_weight: float = 1.0
    dynamics_weight: float = 1.0
    impulse_weight: float = 0.1
    contact_mode_class_weights: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.task_profile not in TASK_PROFILE_ACTIVE_LOSSES:
            raise ValueError("task_profile must be v1, v2_a1, or v2_a2_core3")
        for name, value in self.__dict__.items():
            if name in {"task_profile", "contact_mode_class_weights"}:
                continue
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.contact_mode_class_weights is not None and (
                len(self.contact_mode_class_weights) != 4
                or any(value <= 0 for value in self.contact_mode_class_weights)):
            raise ValueError("contact_mode_class_weights must contain four positive values")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, 0).sum() / mask.sum().clamp_min(1)


def _regression(name, prediction, target, mask, statistics):
    pred = statistics.standardize(name, prediction)
    truth = statistics.standardize(name, target)
    return _masked_mean(F.smooth_l1_loss(pred, truth, reduction="none"), mask)


def teacher_losses(output: dict[str, torch.Tensor], batch: FootGroundBatch,
                   statistics: RegressionStatistics,
                   config: FootGroundLossConfig | None = None
                   ) -> dict[str, torch.Tensor]:
    """Return every atomic loss plus group totals; no hidden validity fallback."""
    cfg = config or FootGroundLossConfig()
    batch.validate()
    transition = batch.valid_transition
    foot_transition = transition.unsqueeze(-1).expand_as(batch.contact)
    contact_mask = foot_transition & batch.contact_valid
    impulse_mask = foot_transition & (batch.contact_valid | (batch.contact_mode == 3))
    losses = {
        "contact_bce": _masked_mean(F.binary_cross_entropy_with_logits(
            output["contact_logits"], batch.contact.to(output["contact_logits"].dtype),
            reduction="none"), foot_transition),
        "contact_mode_ce": _masked_mean(F.cross_entropy(
            output["contact_mode_logits"].flatten(0, -2), batch.contact_mode.flatten(),
            weight=(None if cfg.contact_mode_class_weights is None else
                    output["contact_mode_logits"].new_tensor(
                        cfg.contact_mode_class_weights)),
            reduction="none").reshape_as(batch.contact_mode), foot_transition),
        "contact_point": _regression("contact_point_local", output["contact_point_local"],
                                     batch.contact_point_local, contact_mask, statistics),
        "contact_normal": _masked_mean(1.0 - (output["contact_normal"]
                                               * batch.contact_normal).sum(-1), contact_mask),
        "relative_velocity_normal": _regression(
            "relative_velocity_normal", output["relative_velocity_normal"],
            batch.relative_velocity_normal, contact_mask, statistics),
        "relative_velocity_tangent": _regression(
            "relative_velocity_tangent", output["relative_velocity_tangent"],
            batch.relative_velocity_tangent, contact_mask, statistics),
        "slip_speed": _regression("slip_speed", output["slip_speed"],
                                  batch.slip_speed, contact_mask, statistics),
        "normal_force": _regression("normal_force", output["normal_force"],
                                    batch.normal_force, contact_mask, statistics),
        "tangential_force": _regression("tangential_force", output["tangential_force"],
                                        batch.tangential_force, contact_mask, statistics),
        "contact_impulse": _regression("contact_impulse", output["contact_impulse"],
                                       batch.contact_impulse, impulse_mask, statistics),
        "foot_load_ratio": _regression("foot_load_ratio", output["foot_load_ratio"],
                                       batch.foot_load_ratio, transition, statistics),
        "delta_joint_velocity": _regression(
            "delta_joint_velocity", output["delta_joint_velocity"],
            batch.delta_joint_velocity, transition, statistics),
        "delta_root_angular_velocity": _regression(
            "delta_root_angular_velocity", output["delta_root_angular_velocity"],
            batch.delta_root_angular_velocity, transition, statistics),
        "joint_velocity_residual": _regression(
            "joint_velocity_residual", output["joint_velocity_residual"],
            batch.joint_velocity_residual, batch.joint_velocity_residual_valid, statistics),
    }
    zero = losses["contact_bce"] * 0.0
    losses["contact_group"] = cfg.contact_weight * (
        losses["contact_bce"] + losses["contact_mode_ce"])
    if cfg.task_profile == "v1":
        losses["geometry_group"] = cfg.geometry_weight * (
            losses["contact_point"] + losses["contact_normal"])
        losses["kinematics_group"] = cfg.kinematics_weight * (
            losses["relative_velocity_normal"] + losses["relative_velocity_tangent"]
            + losses["slip_speed"])
        losses["force_group"] = cfg.force_weight * (
            losses["normal_force"] + losses["tangential_force"]
            + losses["foot_load_ratio"]) + cfg.impulse_weight * losses["contact_impulse"]
        losses["dynamics_group"] = cfg.dynamics_weight * (
            losses["delta_joint_velocity"] + losses["delta_root_angular_velocity"]
            + losses["joint_velocity_residual"])
    elif cfg.task_profile == "v2_a1":
        # Remove Geometry and redundant kinematics while retaining the explicit
        # 2-D tangent velocity, full force group, and dynamics supervision.
        losses["geometry_group"] = zero
        losses["kinematics_group"] = (
            cfg.kinematics_weight * losses["relative_velocity_tangent"])
        losses["force_group"] = cfg.force_weight * (
            losses["normal_force"] + losses["tangential_force"]
            + losses["foot_load_ratio"]) + cfg.impulse_weight * losses["contact_impulse"]
        losses["dynamics_group"] = cfg.dynamics_weight * (
            losses["delta_joint_velocity"] + losses["delta_root_angular_velocity"]
            + losses["joint_velocity_residual"])
    else:
        # Core-3 predicts contact/mode, 2-D tangent velocity, and normal force.
        # Inactive groups remain differentiable zeros for a stable audit schema.
        losses["geometry_group"] = zero
        losses["kinematics_group"] = (
            cfg.kinematics_weight * losses["relative_velocity_tangent"])
        losses["force_group"] = cfg.force_weight * losses["normal_force"]
        losses["dynamics_group"] = zero
    losses["total"] = sum(losses[name] for name in (
        "contact_group", "geometry_group", "kinematics_group", "force_group",
        "dynamics_group"))
    return losses


def gradient_audit(model, losses: dict[str, torch.Tensor]) -> dict[str, float]:
    """Measure shared-trunk norms and pairwise cosines before one optimizer step."""
    parameters = tuple(model.shared_parameters())
    groups = ("contact_group", "geometry_group", "kinematics_group",
              "force_group", "dynamics_group")
    vectors = {}
    result = {}
    for group in groups:
        grads = torch.autograd.grad(losses[group], parameters, retain_graph=True,
                                    allow_unused=True)
        vector = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                            for p, g in zip(parameters, grads)])
        vectors[group] = vector
        result[f"gradient_norm/{group}"] = float(torch.linalg.vector_norm(vector).detach())
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            denom = (torch.linalg.vector_norm(vectors[left])
                     * torch.linalg.vector_norm(vectors[right])).clamp_min(1e-12)
            result[f"gradient_cosine/{left}:{right}"] = float(
                ((vectors[left] * vectors[right]).sum() / denom).detach())
    return result
