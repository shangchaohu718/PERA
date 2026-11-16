"""Entire Actor differentiable; immutable teachers and separate full checkpoints."""

import hashlib
from pathlib import Path
import torch
from blinkmimic import networks
from blinkmimic.moe_policy import ReferenceMoE
from experiments.blink_stairs_routing_ab_20261106.routing import (
    CompetitiveMoE,
    load_base,
    public_digest,
)

from pera.paths import ROOT

R = ROOT
BASE = R / "checkpoints/base/best.pt"
FORMAT = "blink_g1_full_actor_mixed_finetune_v1"
KEYS = ["body_hist", "router_obs", "goal", "target"]


def pack(history, goal, target):
    # Earlier router observations are unused by the Actor. This packs raw inputs,
    # not learned features; all trainable modules run again on each update.
    return dict(
        body_hist=history[..., :99].detach(),
        router_obs=history[:, -1, 99:].detach(),
        goal=goal.detach(),
        target=target.detach(),
    )


def unpack(batch):
    body = batch["body_hist"]
    h = body.new_zeros((len(body), 10, 608))
    h[..., :99] = body
    h[:, -1, 99:] = batch["router_obs"]
    return h, batch["goal"]


def digest(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class FullActorMoE(CompetitiveMoE):
    def forward(self, x, r, routed=True):
        # No no_grad block or detach here: old experts, reference encoder, shared
        # branch and both old/new router logits all belong to the same graph.
        native, stats = ReferenceMoE.forward(self, x, r, routed)
        xn = self.norm(x)
        with torch.autocast(device_type=r.device.type, enabled=False):
            hidden = self.router_embed(r.float())
            logits = self.router(hidden)
        old = self.select(logits)
        raw = logits.gather(-1, old)
        choice = raw + self.selection_bias[old]
        with torch.autocast(
            device_type=x.device.type,
            dtype=networks.AMP_DTYPE or torch.bfloat16,
            enabled=networks.AMP_DTYPE is not None,
        ):
            gate, up = torch.einsum("bd,edi->bei", xn, self.gate_up).chunk(2, -1)
            outs = torch.einsum("beh,ehd->bed", torch.nn.functional.silu(gate) * up, self.down)
            top1 = outs.gather(1, old[:, :1, None].expand(-1, 1, outs.shape[-1]))[:, 0]
            gate, up = self.shared_up(xn).chunk(2, -1)
            shared = self.shared_down(torch.nn.functional.silu(gate) * up)
        f = dict(
            x=x,
            xn=xn,
            hidden=hidden,
            native=native,
            top1=top1,
            shared=shared,
            logit1=raw[:, :1],
            choice2=choice[:, 1:2],
            anchor=logits.logsumexp(-1, keepdim=True) - __import__("math").log(8),
        )
        branch, selected, weight, margin = self.compete(f)
        if not self.domain_enabled:
            branch = native
            selected = torch.zeros_like(selected)
            weight = torch.zeros_like(weight)
        self.last_old_indices = old.detach()
        self.last_selected = selected.detach()
        self.last_weight = weight.detach()
        self.last_margin = margin.detach()
        return branch, stats


def load_actor(source=None, full=None, device="cuda:0", trainable=False):
    assert (source is None) != (full is None)
    if source is not None:
        pi, cfg, robot = load_base(BASE, domain=source, device=device)
    else:
        payload = torch.load(full, map_location=device, weights_only=True)
        assert payload["format"] == FORMAT and payload["robot"] == "g1"
        assert payload["source_base_sha256"] == hashlib.sha256(BASE.read_bytes()).hexdigest()
        pi, cfg, robot = load_base(BASE, mode=payload["routing_mode"], device=device)
        pi.load_state_dict(payload["pi"])
    pi.moe.__class__ = FullActorMoE
    pi.requires_grad_(trainable)
    pi.eval()
    return pi, cfg, robot


def teacher_base(device="cuda:0"):
    pi, _, _ = load_base(BASE, device=device)
    pi.moe.domain_enabled = False
    pi.requires_grad_(False)
    pi.eval()
    return pi
