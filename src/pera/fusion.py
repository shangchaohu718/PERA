"""Continuous anchor fusion as an explicitly separate A/B condition."""

import torch
from torch.nn import functional as F
from blinkmimic import networks
from pera.routing import CompetitiveMoE, initialize as competitive_initialize


class FusionMoE(CompetitiveMoE):
    def compete(self, f):
        if self.routing_mode != "fusion":
            return super().compete(f)
        score = self.plugin.score(f["hidden"])
        margin = score - f["anchor"]
        weight = margin.sigmoid()
        with torch.autocast(
            device_type=score.device.type,
            dtype=networks.AMP_DTYPE or torch.bfloat16,
            enabled=networks.AMP_DTYPE is not None,
        ):
            domain = f["shared"] + self.plugin(f["xn"])
            branch = (1 - weight.to(domain.dtype)) * f["native"] + weight.to(domain.dtype) * domain
        return branch, weight > 0, weight, margin


@torch.no_grad()
def initialize(pi, bank, old_ids, target=0.95):
    if pi.moe.routing_mode != "fusion":
        return competitive_initialize(pi, bank, old_ids, target)
    mode = pi.moe.routing_mode
    pi.moe.routing_mode = "competitive"
    result = competitive_initialize(pi, bank, old_ids, target)
    pi.moe.routing_mode = mode
    target_weight = result["calibration"]["contribution_all"]
    h = bank["hidden"].float()
    anchor = bank["anchor"].float()
    n = len(h)
    w = torch.linalg.solve(
        h.T @ h / n + 0.001 * torch.eye(h.shape[1], device=h.device), h.T @ anchor / n
    ).T
    lo, hi = -30.0, 0.0
    for _ in range(60):
        bias = (lo + hi) / 2
        if float((F.linear(h, w) + bias - anchor).sigmoid().mean()) < target_weight:
            lo = bias
        else:
            hi = bias
    p = pi.moe.plugin
    p.router.weight.copy_(w)
    p.router.bias.fill_((lo + hi) / 2)
    p.selection_bias.zero_()
    weights = (p.score(h) - anchor).sigmoid()
    if abs(float(weights.mean()) - target_weight) >= 1e-5:
        raise RuntimeError("Fusion initialization mismatch")
    return dict(
        clone_expert_zero_based=result["clone_expert_zero_based"],
        clone_expert_human=result["clone_expert_human"],
        old_second_slot_counts=result["old_second_slot_counts"],
        anchor="logsumexp of all8 raw public logits minus log8",
        contribution_all=float(weights.mean()),
        matched_competitive_initial_contribution=target_weight,
        bias=float(p.router.bias),
        fixed_merge_bias=0.0,
        no_topk_competition=True,
        competitive_initialization_reference=result,
    )
