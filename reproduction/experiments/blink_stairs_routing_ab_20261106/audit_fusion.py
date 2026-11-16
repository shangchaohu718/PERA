"""CPU algebra audit for public-slot preservation and single shared contribution."""

import json
from pathlib import Path
import torch
from blinkmimic import networks
from experiments.blink_stairs_routing_ab_20261106.routing import load_base, public_digest

from pera.paths import ROOT

R = ROOT
torch.set_num_threads(4)
pi, _, _ = load_base(R / "checkpoints/base/best.pt", device="cpu", mode="fusion")
networks.AMP_DTYPE = None
bank = torch.load(
    R / "reports/blink_competitive_stairs_v3_20261106/training/old_reference_feature_bank.pt",
    map_location="cpu",
    weights_only=True,
)
f = {k: v[:64].float() for k, v in bank.items() if k not in ["oldids", "target"]}
with torch.no_grad():
    logits = pi.moe.router(f["hidden"])
    f["anchor"] = logits.logsumexp(-1, keepdim=True) - __import__("math").log(8)
    original_ids = pi.moe.select(logits).clone()
    pi.moe.plugin.gate_up.copy_(pi.moe.gate_up[4])
    pi.moe.plugin.down.copy_(pi.moe.down[4])
    pi.moe.plugin.router.weight.copy_(pi.moe.router.weight[0])
    pi.moe.plugin.router.bias.fill_(-1.0)
before = public_digest(pi)
branch, _, weight, _ = pi.moe.compete(f)
domain = pi.moe.plugin(f["xn"])
expected = (1 - weight) * f["native"] + weight * (f["shared"] + domain)
torch.testing.assert_close(branch, expected, atol=0, rtol=0)
torch.testing.assert_close(
    branch - f["shared"],
    (1 - weight) * (f["native"] - f["shared"]) + weight * domain,
    atol=3e-5,
    rtol=3e-5,
)
grads = torch.autograd.grad(branch.square().mean(), list(pi.moe.plugin.parameters()))
assert all(torch.isfinite(v).all() for v in grads)
with torch.no_grad():
    pi.moe.plugin.selection_bias.fill_(1e6)
    other = pi.moe.compete(f)[0]
    torch.testing.assert_close(other, branch, atol=0, rtol=0)
    pi.moe.plugin.router.weight.mul_(-3)
    pi.moe.plugin.router.bias.fill_(-10.0)
    assert torch.equal(pi.moe.select(pi.moe.router(f["hidden"])), original_ids)
assert before == public_digest(pi)
result = dict(
    fusion_matches_anchor_domain_formula=True,
    shared_counted_once=True,
    new_selection_offset_unused_in_fusion=True,
    new_router_does_not_modify_public_top2=True,
    public_parameters_and_buffers_unchanged=True,
    new_gradient_finite=True,
)
(R / "reports/blink_stairs_routing_ab_20261106/fusion_algebra_audit.json").write_text(
    json.dumps(result, indent=2)
)
print(result)
