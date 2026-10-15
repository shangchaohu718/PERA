"""Paired action-perturbation metrics for action-conditioned dynamics."""
from __future__ import annotations

import torch


BRANCHES = ("base", "plus", "minus")


def _regression(prediction: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> dict[str, float | int | None]:
    error = (prediction - target).float()
    if mask is not None:
        while mask.ndim < error.ndim:
            mask = mask.unsqueeze(-1)
        error = error[mask.expand_as(error)]
    else:
        error = error.flatten()
    if not error.numel():
        return {"count": 0, "mae": None, "rmse": None}
    return {"count": int(error.numel()), "mae": float(error.abs().mean()),
            "rmse": float(error.square().mean().sqrt())}


def _vector_delta_metrics(prediction: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor, *, truth_floor: float,
                          ignored_ratio: float) -> dict:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("paired vector predictions and targets must have equal shape")
    if mask.shape != prediction.shape[:-1]:
        raise ValueError("vector mask shape mismatch")
    p = prediction[mask].reshape(-1, prediction.shape[-1]).float()
    t = target[mask].reshape(-1, target.shape[-1]).float()
    if not t.numel():
        return {"count": 0, "mae": None, "rmse": None, "cosine_mean": None,
                "sign_accuracy": None, "action_ignored_rate": None,
                "sensitivity_gain_median": None}
    p_norm = torch.linalg.vector_norm(p, dim=-1)
    t_norm = torch.linalg.vector_norm(t, dim=-1)
    informative = t_norm >= truth_floor
    cosine = torch.nn.functional.cosine_similarity(p[informative], t[informative]) \
        if informative.any() else None
    sign_usable = informative.unsqueeze(-1) & (t.abs() >= truth_floor)
    sign_accuracy = ((p.sign() == t.sign()) & sign_usable).sum() / sign_usable.sum() \
        if sign_usable.any() else None
    gain = p_norm[informative] / t_norm[informative].clamp_min(1e-12)
    ignored = p_norm[informative] <= ignored_ratio * t_norm[informative]
    base = _regression(p, t)
    return {**base,
            "cosine_mean": None if cosine is None else float(cosine.mean()),
            "sign_accuracy": None if sign_accuracy is None else float(sign_accuracy),
            "action_ignored_rate": None if not informative.any() else float(ignored.float().mean()),
            "sensitivity_gain_median": None if not gain.numel() else float(gain.median())}


def evaluate_action_counterfactual(*, predicted_contact_probability: torch.Tensor,
        predicted_tangent_velocity: torch.Tensor, predicted_normal_force: torch.Tensor,
        target_contact: torch.Tensor, target_tangent_velocity: torch.Tensor,
        target_normal_force: torch.Tensor, delta_rad: torch.Tensor,
        processed_target_delta_rad: torch.Tensor | None = None,
        null_target_contact: torch.Tensor | None = None,
        null_target_tangent_velocity: torch.Tensor | None = None,
        null_target_normal_force: torch.Tensor | None = None,
        contact_threshold: float = .5, tangent_truth_floor_m_s: float = .005,
        force_truth_floor_n: float = 5.0, ignored_ratio: float = .1,
        noise_multiplier: float = 3.0) -> dict:
    """Evaluate branch deltas; branch axis order is base, plus, minus."""
    expected_contact = target_contact.shape
    if len(expected_contact) != 3 or expected_contact[1:] != (3, 2):
        raise ValueError("target_contact must be [N,3,2]")
    if predicted_contact_probability.shape != expected_contact:
        raise ValueError("contact probability shape mismatch")
    if target_tangent_velocity.shape != expected_contact + (2,) or \
            predicted_tangent_velocity.shape != target_tangent_velocity.shape:
        raise ValueError("tangent velocity must be [N,3,2,2]")
    if target_normal_force.shape != expected_contact or \
            predicted_normal_force.shape != expected_contact:
        raise ValueError("normal force must be [N,3,2]")
    if delta_rad.shape != (expected_contact[0],) or (delta_rad <= 0).any():
        raise ValueError("delta_rad must be positive [N]")
    if processed_target_delta_rad is None:
        processed_target_delta_rad = torch.stack(
            (torch.zeros_like(delta_rad), delta_rad, -delta_rad), -1)
    if processed_target_delta_rad.shape != (expected_contact[0], 3):
        raise ValueError("processed_target_delta_rad must be [N,3]")
    if (processed_target_delta_rad[:, 1] <= 0).any() or \
            (processed_target_delta_rad[:, 2] >= 0).any():
        raise ValueError("processed plus/minus target deltas have wrong signs")
    for value in (predicted_contact_probability, predicted_tangent_velocity,
                  predicted_normal_force, target_tangent_velocity,
                  target_normal_force, delta_rad, processed_target_delta_rad):
        if not torch.isfinite(value).all():
            raise ValueError("counterfactual tensors must be finite")

    pred_contact = predicted_contact_probability >= contact_threshold
    persistent = target_contact[:, 0] & target_contact[:, 1] & target_contact[:, 2]
    branch_results = {}
    for index, name in ((1, "plus"), (2, "minus")):
        tangent_pred_delta = (predicted_tangent_velocity[:, index]
                              - predicted_tangent_velocity[:, 0])
        tangent_true_delta = target_tangent_velocity[:, index] - target_tangent_velocity[:, 0]
        force_pred_delta = (predicted_normal_force[:, index]
                            - predicted_normal_force[:, 0]).unsqueeze(-1)
        force_true_delta = (target_normal_force[:, index]
                            - target_normal_force[:, 0]).unsqueeze(-1)
        branch_results[name] = {
            "tangent_velocity_delta": _vector_delta_metrics(
                tangent_pred_delta, tangent_true_delta, persistent,
                truth_floor=tangent_truth_floor_m_s, ignored_ratio=ignored_ratio),
            "normal_force_delta": _vector_delta_metrics(
                force_pred_delta, force_true_delta,
                torch.ones_like(persistent), truth_floor=force_truth_floor_n,
                ignored_ratio=ignored_ratio),
            "contact_branch_accuracy": float(
                (pred_contact[:, index] == target_contact[:, index]).float().mean()),
            "contact_change_accuracy": float(
                ((pred_contact[:, index] != pred_contact[:, 0]) ==
                 (target_contact[:, index] != target_contact[:, 0])).float().mean()),
        }

    actual_span = (processed_target_delta_rad[:, 1]
                   - processed_target_delta_rad[:, 2])
    scale_tangent = actual_span.view(-1, 1, 1)
    tangent_jacobian_pred = ((predicted_tangent_velocity[:, 1]
                              - predicted_tangent_velocity[:, 2]) / scale_tangent)
    tangent_jacobian_true = ((target_tangent_velocity[:, 1]
                              - target_tangent_velocity[:, 2]) / scale_tangent)
    scale_force = actual_span.view(-1, 1, 1)
    force_jacobian_pred = ((predicted_normal_force[:, 1]
                            - predicted_normal_force[:, 2]).unsqueeze(-1) / scale_force)
    force_jacobian_true = ((target_normal_force[:, 1]
                            - target_normal_force[:, 2]).unsqueeze(-1) / scale_force)
    quality = {
        "processed_plus_abs_error_rad": float(
            (processed_target_delta_rad[:, 1] - delta_rad).abs().max()),
        "processed_minus_abs_error_rad": float(
            (processed_target_delta_rad[:, 2] + delta_rad).abs().max()),
        "null_replicate_available": null_target_contact is not None,
    }
    if null_target_contact is not None:
        if null_target_contact.shape != target_contact[:, 0].shape or \
                null_target_tangent_velocity is None or \
                null_target_tangent_velocity.shape != target_tangent_velocity[:, 0].shape or \
                null_target_normal_force is None or \
                null_target_normal_force.shape != target_normal_force[:, 0].shape:
            raise ValueError("null replicate shapes do not match the base branch")
        tangent_noise = torch.linalg.vector_norm(
            null_target_tangent_velocity - target_tangent_velocity[:, 0], dim=-1)
        force_noise = (null_target_normal_force - target_normal_force[:, 0]).abs()
        tangent_effect = torch.stack([
            torch.linalg.vector_norm(target_tangent_velocity[:, i]
                                     - target_tangent_velocity[:, 0], dim=-1)
            for i in (1, 2)], 1)
        force_effect = torch.stack([
            (target_normal_force[:, i] - target_normal_force[:, 0]).abs()
            for i in (1, 2)], 1)
        tangent_informative_floor = torch.maximum(
            torch.full_like(tangent_noise, tangent_truth_floor_m_s),
            noise_multiplier * tangent_noise)
        force_informative_floor = torch.maximum(
            torch.full_like(force_noise, force_truth_floor_n),
            noise_multiplier * force_noise)
        tangent_eligible = persistent
        force_eligible = target_contact.any(dim=1)
        tangent_noise_floor = noise_multiplier * tangent_noise
        force_noise_floor = noise_multiplier * force_noise

        def eligible_rate(effect: torch.Tensor, floor: torch.Tensor,
                          eligible: torch.Tensor) -> float | None:
            mask = eligible[:, None].expand_as(effect)
            if not mask.any():
                return None
            return float((effect[mask] >= floor[:, None].expand_as(effect)[mask])
                         .float().mean())

        quality.update(
            null_contact_match_rate=float(
                (null_target_contact == target_contact[:, 0]).float().mean()),
            null_tangent_velocity_noise_rmse=float(
                (null_target_tangent_velocity - target_tangent_velocity[:, 0])
                .square().mean().sqrt()),
            null_normal_force_noise_rmse_n=float(
                (null_target_normal_force - target_normal_force[:, 0])
                .square().mean().sqrt()),
            tangent_response_above_noise_rate=eligible_rate(
                tangent_effect, tangent_noise_floor, tangent_eligible),
            force_response_above_noise_rate=eligible_rate(
                force_effect, force_noise_floor, force_eligible),
            tangent_informative_response_rate=eligible_rate(
                tangent_effect, tangent_informative_floor, tangent_eligible),
            force_informative_response_rate=eligible_rate(
                force_effect, force_informative_floor, force_eligible),
            tangent_response_eligible_foot_samples=int(tangent_eligible.sum()),
            force_response_eligible_foot_samples=int(force_eligible.sum()),
        )
    return {
        "schema": 1, "branches": list(BRANCHES),
        "samples": expected_contact[0],
        "persistent_contact_foot_samples": int(persistent.sum()),
        "branch": branch_results,
        "data_quality": quality,
        "symmetric_jacobian": {
            "tangent_velocity": _vector_delta_metrics(
                tangent_jacobian_pred, tangent_jacobian_true, persistent,
                truth_floor=tangent_truth_floor_m_s / float(delta_rad.max()),
                ignored_ratio=ignored_ratio),
            "normal_force": _vector_delta_metrics(
                force_jacobian_pred, force_jacobian_true,
                torch.ones_like(persistent),
                truth_floor=force_truth_floor_n / float(delta_rad.max()),
                ignored_ratio=ignored_ratio),
        },
    }
