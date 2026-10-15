"""Contact-probability calibration and true-mode stratification."""
from __future__ import annotations

import torch


def contact_calibration(probability: torch.Tensor, truth: torch.Tensor,
                        true_mode: torch.Tensor, predicted_mode: torch.Tensor,
                        threshold: torch.Tensor, *, bins: int = 10) -> dict:
    """Pool probabilities while retaining each model seed's frozen threshold."""
    probability = probability.float().flatten()
    truth = truth.bool().flatten()
    true_mode = true_mode.long().flatten()
    predicted_mode = predicted_mode.long().flatten()
    threshold = threshold.float().flatten()
    if not (probability.shape == truth.shape == true_mode.shape ==
            predicted_mode.shape == threshold.shape):
        raise ValueError("calibration inputs must have equal flattened shape")
    if not probability.numel() or bins < 2 or ((probability < 0) | (probability > 1)).any():
        raise ValueError("invalid contact calibration inputs")
    predicted = probability >= threshold
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probability.device)
    rows, weighted_gap = [], 0.0
    for index in range(bins):
        mask = ((probability >= edges[index]) &
                (probability <= edges[index + 1] if index == bins - 1
                 else probability < edges[index + 1]))
        count = int(mask.sum())
        confidence = float(probability[mask].mean()) if count else None
        frequency = float(truth[mask].float().mean()) if count else None
        gap = None if count == 0 else abs(confidence - frequency)
        if gap is not None:
            weighted_gap += count * gap
        rows.append({"low": float(edges[index]), "high": float(edges[index + 1]),
                     "count": count, "mean_probability": confidence,
                     "empirical_contact_frequency": frequency, "absolute_gap": gap})
    eps = torch.finfo(probability.dtype).eps
    bce = torch.nn.functional.binary_cross_entropy(
        probability.clamp(eps, 1 - eps), truth.float())
    modes = {}
    for index, name in enumerate(("separation", "stable", "sliding", "impact")):
        mask = true_mode == index
        count = int(mask.sum())
        if not count:
            modes[name] = {"count": 0}
            continue
        p, y = probability[mask], truth[mask].float()
        modes[name] = {
            "count": count, "mean_probability": float(p.mean()),
            "probability_q10": float(torch.quantile(p, .1)),
            "probability_median": float(torch.quantile(p, .5)),
            "probability_q90": float(torch.quantile(p, .9)),
            "bce": float(torch.nn.functional.binary_cross_entropy(
                p.clamp(eps, 1 - eps), y)),
            "brier": float((p - y).square().mean()),
            "predicted_contact_rate": float(predicted[mask].float().mean()),
            "mode_head_accuracy": float((predicted_mode[mask] == index).float().mean()),
        }
    positive, negative = probability[truth], probability[~truth]
    quantiles = lambda value: ({
        "count": int(value.numel()), "q01": float(torch.quantile(value, .01)),
        "q10": float(torch.quantile(value, .1)), "median": float(torch.quantile(value, .5)),
        "q90": float(torch.quantile(value, .9)), "q99": float(torch.quantile(value, .99)),
    } if value.numel() else {"count": 0})
    return {
        "count": int(probability.numel()), "bins": rows,
        "ece": weighted_gap / probability.numel(),
        "mce": max((row["absolute_gap"] or 0.0) for row in rows),
        "bce": float(bce), "brier": float((probability - truth.float()).square().mean()),
        "positive_probability": quantiles(positive),
        "negative_probability": quantiles(negative),
        "by_true_contact_mode": modes,
    }
