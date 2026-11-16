"""Protect the optional A/B branch when sharing its public feature cache."""

import copy
import math

import torch
from torch import nn

from blinkmimic import networks
from blinkmimic.moe_policy import ReferenceMoE
from pera.fusion import FusionMoE, initialize
from pera.routing import Plugin


def test_fusion_formula_cache_and_initialization():
    torch.set_num_threads(2)
    torch.manual_seed(72)
    networks.AMP_DTYPE = None
    native = ReferenceMoE().eval().requires_grad_(False)
    pi = nn.Module()
    pi.moe = copy.deepcopy(native)
    pi.moe.__class__ = FusionMoE
    pi.moe.plugin = Plugin()
    pi.moe.domain_enabled = True
    pi.moe.routing_mode = "fusion"
    x, r = torch.randn(128, 256), torch.randn(128, 509)
    pi.moe(x, r)
    bank = {k: v.detach() for k, v in pi.moe.last_features.items()}
    info = initialize(pi, bank, pi.moe.last_old_indices, target=0.6)
    assert abs(info["contribution_all"] - info["matched_competitive_initial_contribution"]) < 1e-5
    actual, _ = pi.moe(x, r)
    cache = pi.moe.last_features
    public, _ = native(x, r)
    anchor = native.logits(r).logsumexp(-1, keepdim=True) - math.log(8)
    weight = (pi.moe.plugin.score(native.router_embed(r)) - anchor).sigmoid()
    g, u = native.shared_up(native.norm(x)).chunk(2, -1)
    shared = native.shared_down(torch.nn.functional.silu(g) * u)
    expected = (1 - weight) * public + weight * (shared + pi.moe.plugin(native.norm(x)))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    head = nn.Linear(256, 29).requires_grad_(False)
    cache = {**cache, "skip": torch.zeros(128, 29)}
    cached, *_ = pi.moe.predict(cache, head)
    torch.testing.assert_close(cached, head(x + actual), atol=0, rtol=0)
    pi.moe.domain_enabled = False
    disabled, _ = pi.moe(x, r)
    torch.testing.assert_close(disabled, public, atol=0, rtol=0)
    assert not torch.cuda.is_initialized()
