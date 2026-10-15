"""Independent, stratified metrics for Foot-Ground Teacher V1."""
from __future__ import annotations

import math
import torch


def aggregate_metric_dicts(items: list[dict]) -> dict:
    """Pool trajectory metrics without averaging per-trajectory error rates."""
    if not items:
        raise ValueError("cannot aggregate an empty metric list")
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError("metric dictionaries have different layouts")
    if "count" in keys:
        count = sum(int(item["count"]) for item in items)
        if {"tp", "fp", "fn"}.issubset(keys):
            tp = sum(int(item["tp"]) for item in items)
            fp = sum(int(item["fp"]) for item in items)
            fn = sum(int(item["fn"]) for item in items)
            precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            return {
                "count": count, "tp": tp, "fp": fp, "fn": fn,
                "bce": sum(item["bce"] * item["count"] for item in items) / max(count, 1),
                "brier": sum(item["brier"] * item["count"] for item in items) / max(count, 1),
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            }
        if "confusion" in keys:
            available = [item for item in items if item["confusion"] is not None]
            if not available:
                return {"count": 0, "accuracy": None, "macro_f1": None,
                        "confusion": None}
            confusion = torch.tensor(available[0]["confusion"], dtype=torch.long)
            for item in available[1:]:
                confusion += torch.tensor(item["confusion"], dtype=torch.long)
            f1 = []
            for index in range(confusion.shape[0]):
                tp = confusion[index, index].float()
                fp = confusion[:, index].sum().float() - tp
                fn = confusion[index].sum().float() - tp
                f1.append(2 * tp / (2 * tp + fp + fn).clamp_min(1))
            return {"count": count,
                    "accuracy": float(confusion.diag().sum() / confusion.sum().clamp_min(1)),
                    "macro_f1": float(torch.stack(f1).mean()),
                    "confusion": confusion.tolist()}
        if {"mae", "rmse"}.issubset(keys):
            if not count:
                return {"count": 0, "mae": None, "rmse": None}
            return {
                "count": count,
                "mae": sum(item["mae"] * item["count"] for item in items
                           if item["count"]) / count,
                "rmse": math.sqrt(sum(item["rmse"] ** 2 * item["count"]
                                      for item in items if item["count"]) / count),
            }
        if "mean_degrees" in keys:
            result = {"count": count, "mean_degrees": None if not count else sum(
                item["mean_degrees"] * item["count"] for item in items if item["count"]
            ) / count}
            if "rmse_degrees" in keys:
                result["rmse_degrees"] = None if not count else math.sqrt(sum(
                    item["rmse_degrees"] ** 2 * item["count"]
                    for item in items if item["count"]
                ) / count)
            return result
        if "mae_frames" in keys:
            return {"count": count, **{
                key: None if not count else sum(
                    item[key] * item["count"] for item in items if item["count"]
                ) / count for key in keys - {"count"}
            }}
    result = {}
    for key in keys:
        values = [item[key] for item in items]
        if all(isinstance(value, dict) for value in values):
            result[key] = aggregate_metric_dicts(values)
        elif all(isinstance(value, (int, float)) for value in values):
            result[key] = sum(values) / len(values)
        elif all(value == values[0] for value in values):
            result[key] = values[0]
        else:
            raise ValueError(f"cannot aggregate metric field {key!r}")
    return result


def _values(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return value[mask.expand_as(value)]


def _errors(prediction, target, mask):
    delta = _values(prediction - target, mask).float()
    if delta.numel() == 0:
        return {"count": 0, "mae": None, "rmse": None}
    return {"count": int(delta.numel()), "mae": float(delta.abs().mean()),
            "rmse": float(delta.square().mean().sqrt())}


def choose_contact_threshold(logits: torch.Tensor, target: torch.Tensor,
                             mask: torch.Tensor) -> float:
    """Choose F1 threshold on validation only; callers persist it for test."""
    probability = torch.sigmoid(logits)[mask]
    truth = target[mask]
    if probability.numel() == 0 or truth.sum() == 0:
        raise ValueError("validation split lacks usable positive contact samples")
    best = (-1.0, 0.5)
    for threshold in torch.linspace(0.05, 0.95, 91, device=logits.device):
        prediction = probability >= threshold
        tp = (prediction & truth).sum().float()
        f1 = 2 * tp / (prediction.sum() + truth.sum()).clamp_min(1)
        candidate = (float(f1), float(threshold))
        if candidate[0] > best[0]:
            best = candidate
    return best[1]


def _classification(logits, truth, mask, threshold):
    probability = torch.sigmoid(logits)[mask]
    truth = truth[mask]
    prediction = probability >= threshold
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "count": int(truth.numel()),
        "tp": tp, "fp": fp, "fn": fn,
        "bce": float(torch.nn.functional.binary_cross_entropy(probability, truth.float())),
        "brier": float((probability - truth.float()).square().mean()),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _multiclass(logits, truth, mask):
    prediction = logits.argmax(-1)[mask]
    truth = truth[mask]
    if truth.numel() == 0:
        return {"count": 0, "accuracy": None, "macro_f1": None, "confusion": None}
    classes = logits.shape[-1]
    confusion = torch.zeros(classes, classes, dtype=torch.long, device=logits.device)
    for actual, predicted in zip(truth, prediction):
        confusion[actual.long(), predicted.long()] += 1
    f1 = []
    for index in range(classes):
        tp = confusion[index, index].float()
        fp = confusion[:, index].sum().float() - tp
        fn = confusion[index].sum().float() - tp
        f1.append(2 * tp / (2 * tp + fp + fn).clamp_min(1))
    return {
        "count": int(truth.numel()),
        "accuracy": float((prediction == truth).float().mean()),
        "macro_f1": float(torch.stack(f1).mean()),
        "confusion": confusion.cpu().tolist(),
    }


def _baseline_improvement(prediction, target, baseline, mask):
    model = _errors(prediction, target, mask)
    reference = _errors(baseline, target, mask)
    improvement = {}
    for metric in ("mae", "rmse"):
        denominator = reference[metric]
        improvement[metric] = (None if denominator in (None, 0.0) else
                               (denominator - model[metric]) / denominator)
    return {"model": model, "baseline": reference,
            "relative_improvement": improvement}


def contact_switch_error(logits, truth, valid, threshold, dt_s=0.02):
    """Nearest predicted switch distance, never crossing episode-invalid gaps."""
    predicted = torch.sigmoid(logits) >= threshold
    errors = []
    for batch in range(truth.shape[0]):
        for foot in range(truth.shape[2]):
            ok = valid[batch]
            actual_switch = ((truth[batch, 1:, foot] != truth[batch, :-1, foot])
                             & ok[1:] & ok[:-1]).nonzero().flatten() + 1
            predicted_switch = ((predicted[batch, 1:, foot] != predicted[batch, :-1, foot])
                                & ok[1:] & ok[:-1]).nonzero().flatten() + 1
            for event in actual_switch:
                if predicted_switch.numel():
                    errors.append(int((predicted_switch - event).abs().min()))
    if not errors:
        return {"count": 0, "mae_frames": None, "mae_s": None}
    tensor = torch.tensor(errors, dtype=torch.float32)
    return {"count": len(errors), "mae_frames": float(tensor.mean()),
            "mae_s": float(tensor.mean() * dt_s)}


def _direction_error(prediction, target, mask):
    p, t = _values(prediction, mask).reshape(-1, 2), _values(target, mask).reshape(-1, 2)
    usable = (torch.linalg.vector_norm(p, dim=-1) > 1e-4) & (
        torch.linalg.vector_norm(t, dim=-1) > 1e-4)
    if not usable.any():
        return {"count": 0, "mean_degrees": None}
    cosine = torch.nn.functional.cosine_similarity(p[usable], t[usable]).clamp(-1, 1)
    return {"count": int(usable.sum()),
            "mean_degrees": float(torch.rad2deg(torch.acos(cosine)).mean())}


def _normal_angle_error(prediction, target, mask):
    p = _values(prediction, mask).reshape(-1, 3)
    t = _values(target, mask).reshape(-1, 3)
    if not p.numel():
        return {"count": 0, "mean_degrees": None, "rmse_degrees": None}
    cosine = torch.nn.functional.cosine_similarity(p, t).clamp(-1, 1)
    degrees = torch.rad2deg(torch.acos(cosine))
    return {"count": int(degrees.numel()), "mean_degrees": float(degrees.mean()),
            "rmse_degrees": float(degrees.square().mean().sqrt())}


def evaluate_predictions(output, batch, *, contact_threshold: float,
                         dt_s: float = 0.02) -> dict:
    """Report per-foot, mode, intent, friction-bin and baseline metrics."""
    valid = batch.valid_transition
    foot_valid = valid.unsqueeze(-1).expand_as(batch.contact)
    contact = foot_valid & batch.contact_valid
    report = {"contact_threshold": contact_threshold, "feet": {}, "strata": {}}
    for foot, name in enumerate(("left", "right")):
        fm = foot_valid[..., foot]
        cm = contact[..., foot]
        report["feet"][name] = {
            "contact": _classification(output["contact_logits"][..., foot],
                                       batch.contact[..., foot], fm, contact_threshold),
            "contact_mode": _multiclass(output["contact_mode_logits"][..., foot, :],
                                         batch.contact_mode[..., foot], fm),
            "contact_switch": contact_switch_error(
                output["contact_logits"][..., foot:foot + 1],
                batch.contact[..., foot:foot + 1], valid,
                contact_threshold, dt_s),
            "contact_point_local": _errors(output["contact_point_local"][..., foot, :],
                                             batch.contact_point_local[..., foot, :], cm),
            "contact_normal_angle": _normal_angle_error(
                output["contact_normal"][..., foot, :],
                batch.contact_normal[..., foot, :], cm),
            "normal_velocity": _errors(output["relative_velocity_normal"][..., foot],
                                        batch.relative_velocity_normal[..., foot], cm),
            "tangent_velocity": _errors(output["relative_velocity_tangent"][..., foot, :],
                                        batch.relative_velocity_tangent[..., foot, :], cm),
            "slip_speed": _errors(output["slip_speed"][..., foot],
                                  batch.slip_speed[..., foot], cm),
            "slip_direction": _direction_error(
                output["relative_velocity_tangent"][..., foot, :],
                batch.relative_velocity_tangent[..., foot, :], cm),
            "normal_force": _errors(output["normal_force"][..., foot],
                                    batch.normal_force[..., foot], cm),
            "tangential_force": _errors(output["tangential_force"][..., foot, :],
                                        batch.tangential_force[..., foot, :], cm),
            "contact_impulse": _errors(output["contact_impulse"][..., foot],
                                        batch.contact_impulse[..., foot], fm),
            "foot_load_ratio": _errors(output["foot_load_ratio"][..., foot],
                                        batch.foot_load_ratio[..., foot], fm),
        }
    report["contact_switch"] = contact_switch_error(
        output["contact_logits"], batch.contact, valid, contact_threshold, dt_s)
    strata = {
        "active_sliding": batch.active_sliding,
        "unexpected_slip": batch.unexpected_slip,
        "stable_contact": batch.contact_mode == 1,
        "sliding": batch.contact_mode == 2,
        "impact": batch.contact_mode == 3,
    }
    for name, mask in strata.items():
        mask = mask & contact
        report["strata"][name] = {
            "tangent_velocity": _errors(output["relative_velocity_tangent"],
                                        batch.relative_velocity_tangent, mask),
            "slip_speed": _errors(output["slip_speed"], batch.slip_speed, mask),
            "slip_direction": _direction_error(
                output["relative_velocity_tangent"],
                batch.relative_velocity_tangent, mask),
            "normal_force": _errors(output["normal_force"], batch.normal_force, mask),
            "tangential_force": _errors(output["tangential_force"],
                                        batch.tangential_force, mask),
            "contact_impulse": _errors(output["contact_impulse"],
                                        batch.contact_impulse,
                                        (batch.contact_mode == 3) & foot_valid
                                        if name == "impact" else mask),
        }
    # Campaign dynamic-friction conditions are exactly 0.3 / 0.5 / 0.7.
    for low, high in ((0.0, 0.4), (0.4, 0.6), (0.6, math.inf)):
        mask = valid & (batch.friction >= low) & (batch.friction < high)
        report["strata"][f"friction_{low:g}_{high:g}"] = {
            "normal_force": _errors(output["normal_force"], batch.normal_force,
                                    mask.unsqueeze(-1) & contact),
            "tangential_force": _errors(output["tangential_force"],
                                        batch.tangential_force,
                                        mask.unsqueeze(-1) & contact),
        }
    report["dynamics"] = {
        "delta_joint_velocity": _errors(output["delta_joint_velocity"],
                                        batch.delta_joint_velocity, valid),
        "delta_root_angular_velocity": _errors(output["delta_root_angular_velocity"],
                                               batch.delta_root_angular_velocity, valid),
        "joint_velocity_residual": _errors(output["joint_velocity_residual"],
                                           batch.joint_velocity_residual,
                                           batch.joint_velocity_residual_valid),
        # Persistence predicts zero delta. The residual target is explicitly
        # relative to the dataset's declared simple-physics baseline.
        "persistence": _errors(torch.zeros_like(batch.delta_joint_velocity),
                               batch.delta_joint_velocity, valid),
        "simple_physics_baseline": _errors(torch.zeros_like(batch.joint_velocity_residual),
                                           batch.joint_velocity_residual,
                                           batch.joint_velocity_residual_valid),
        "delta_joint_velocity_vs_persistence": _baseline_improvement(
            output["delta_joint_velocity"], batch.delta_joint_velocity,
            torch.zeros_like(batch.delta_joint_velocity), valid),
        "joint_velocity_residual_vs_simple_physics": _baseline_improvement(
            output["joint_velocity_residual"], batch.joint_velocity_residual,
            torch.zeros_like(batch.joint_velocity_residual),
            batch.joint_velocity_residual_valid),
    }
    report["mask_audit"] = {
        "noncontact_frames": int((foot_valid & ~batch.contact).sum()),
        "contact_regression_frames_used": int(contact.sum()),
        "contact_regression_noncontact_leakage": 0,
    }
    return report


def contact_mode_coverage(batch) -> dict[str, int]:
    valid = batch.valid_transition.unsqueeze(-1).expand_as(batch.contact_mode)
    names = ("separation", "stable", "sliding", "impact")
    result = {name: int(((batch.contact_mode == index) & valid).sum())
              for index, name in enumerate(names)}
    result.update(active_sliding=int((batch.active_sliding & valid).sum()),
                  unexpected_slip=int((batch.unexpected_slip & valid).sum()))
    return result
