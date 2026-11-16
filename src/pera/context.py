"""Context-aware expert and exact single-expert competitive routing."""

import torch
from torch import nn
from torch.nn import functional as F
from blinkmimic import networks
from pera import routing as base


class ContextPlugin(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.gate_up = nn.Parameter(torch.zeros(256 + context_dim, 256))
        self.down = nn.Parameter(torch.zeros(128, 256))
        self.router = nn.Linear(256 + context_dim, 1)
        self.register_buffer("selection_bias", torch.zeros(()))

    def score(self, h):
        with torch.autocast(device_type=h.device.type, enabled=False):
            return F.linear(
                h.float(),
                self.router.weight.float(),
                self.router.bias.clamp_max(0).float(),
            )

    def forward(self, x):
        g, u = (x @ self.gate_up).chunk(2, -1)
        return (F.silu(g) * u) @ self.down


class ContextMoE(base.CompetitiveMoE):
    def compete(self, f):
        ctx = f.get("context", self.runtime_context)
        if ctx is None:
            ctx = f["hidden"].new_zeros(len(f["hidden"]), self.plugin.context_dim)
        if ctx.shape != (len(f["hidden"]), self.plugin.context_dim):
            raise ValueError("Task context shape mismatch")
        f["context"] = ctx.detach()
        score = self.plugin.score(torch.cat((f["hidden"], ctx), -1))
        choice = score + self.plugin.selection_bias
        selected = choice > f["choice2"]
        weight = torch.cat((f["logit1"], score), -1).softmax(-1)[:, 1:2]
        with torch.autocast(
            device_type=score.device.type,
            dtype=networks.AMP_DTYPE or torch.bfloat16,
            enabled=networks.AMP_DTYPE is not None,
        ):
            domain = self.plugin(torch.cat((f["xn"], ctx.to(f["xn"].dtype)), -1))
            candidate = (
                f["shared"]
                + (1 - weight.to(domain.dtype)) * f["top1"]
                + weight.to(domain.dtype) * domain
            )
        return (
            torch.where(selected, candidate, f["native"]),
            selected,
            torch.where(selected, weight, torch.zeros_like(weight)),
            choice - f["choice2"],
        )
