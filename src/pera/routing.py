"""Frozen eight-expert G1 policy plus one trainable competing FFN/router."""

import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from blinkmimic import networks
from blinkmimic.networks import Policy, policy_kwargs
from pera.robot import get_robot
from blinkmimic.moe_policy import ReferenceMoE

FORMAT = "blink_g1_eight_plus_one_competitive_v1"


class Plugin(nn.Module):
    def __init__(self, d=256, h=128):
        super().__init__()
        self.gate_up = nn.Parameter(torch.zeros(d, h * 2))
        self.down = nn.Parameter(torch.zeros(h, d))
        self.router = nn.Linear(d, 1)
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


class CompetitiveMoE(ReferenceMoE):
    def forward(self, x, r, routed=True):
        native, stats = super().forward(x, r, routed)
        with torch.no_grad():
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
                outs = torch.einsum("beh,ehd->bed", F.silu(gate) * up, self.down)
                top1 = outs.gather(1, old[:, :1, None].expand(-1, 1, outs.shape[-1]))[:, 0]
                gate, up = self.shared_up(xn).chunk(2, -1)
                shared = self.shared_down(F.silu(gate) * up)
        features = dict(
            x=x.detach(),
            xn=xn.detach(),
            hidden=hidden.detach(),
            native=native.detach(),
            top1=top1.detach(),
            shared=shared.detach(),
            logit1=raw[:, :1].detach(),
            choice2=choice[:, 1:2].detach(),
            anchor=(logits.float().logsumexp(-1, keepdim=True) - math.log(8)).detach(),
        )
        self.last_features = features
        self.last_old_indices = old.detach()
        branch, selected, weight, margin = self.compete(features)
        if not self.domain_enabled:
            branch = native
            selected = torch.zeros_like(selected)
            weight = torch.zeros_like(weight)
        self.last_selected = selected.detach()
        self.last_weight = weight.detach()
        self.last_margin = margin.detach()
        return branch, stats

    def compete(self, f):
        score = self.plugin.score(f["hidden"])
        choice = score + self.plugin.selection_bias
        selected = choice > f["choice2"]
        # Keep the public distinction: corrected scores select; raw selected scores mix.
        weight = torch.cat((f["logit1"], score), -1).softmax(-1)[:, 1:2]
        with torch.autocast(
            device_type=score.device.type,
            dtype=networks.AMP_DTYPE or torch.bfloat16,
            enabled=networks.AMP_DTYPE is not None,
        ):
            domain = self.plugin(f["xn"])
            candidate = (
                f["shared"]
                + (1 - weight.to(domain.dtype)) * f["top1"]
                + weight.to(domain.dtype) * domain
            )
        branch = torch.where(selected, candidate, f["native"])
        return (
            branch,
            selected,
            torch.where(selected, weight, torch.zeros_like(weight)),
            choice - f["choice2"],
        )

    def predict(self, f, head):
        branch, selected, weight, margin = self.compete(f)
        # Teacher collection and frozen-base probes disable the new expert.
        # Cache replay must use the same switch as the complete forward path.
        if not self.domain_enabled:
            branch = f["native"]
            selected = torch.zeros_like(selected)
            weight = torch.zeros_like(weight)
        with torch.autocast(
            device_type=branch.device.type,
            dtype=networks.AMP_DTYPE or torch.bfloat16,
            enabled=networks.AMP_DTYPE is not None,
        ):
            a = head(f["x"] + branch)
        return a.float() + f["skip"], selected, weight, margin


def public_digest(pi):
    digest = hashlib.sha256()
    for name, value in sorted(pi.state_dict().items()):
        if name.startswith(("moe.plugin.", "moe.plugins.")):
            continue
        digest.update(name.encode())
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def policy_from_args(a, device="cpu"):
    robot = get_robot("g1")
    if a["robot"] != "g1":
        raise ValueError("Only G1 checkpoints are supported")
    widths = lambda n: tuple(map(int, a[n].split(",")))
    networks.AMP_DTYPE = torch.bfloat16 if a.get("amp") == "bf16" else None
    pi = Policy(
        608,
        730,
        29,
        hist=a["hist"],
        enc_hidden=widths("enc_hidden"),
        dec_hidden=widths("dec_hidden"),
        **policy_kwargs(a, robot),
    ).to(device)
    return pi, robot


def load_base(path, device="cuda:0", domain=None, *, mode="competitive", expected_format=FORMAT):
    ck = torch.load(path, map_location=device, weights_only=True)
    a = ck["args"]
    pi, robot = policy_from_args(a, device)
    pi.load_state_dict(ck["pi"])
    pi.eval().requires_grad_(False)
    pi.moe.__class__ = CompetitiveMoE
    pi.moe.plugin = Plugin().to(device)
    pi.moe.domain_enabled = True
    pi.moe.routing_mode = mode
    if domain:
        data = torch.load(domain, map_location=device, weights_only=True)
        assert data["format"] == expected_format and data["public_sha256"] == public_digest(pi)
        pi.moe.routing_mode = data.get("routing_mode", mode)
        pi.moe.plugin.load_state_dict(data["plugin"])
    return pi, a, robot


def features(pi, goal):
    f = {k: v.detach() for k, v in pi.moe.last_features.items()}
    f["skip"] = (
        ((goal[:, pi.ref_off : pi.ref_off + 29] - pi.ref_q0) / pi.ref_alpha).detach()
        if pi.ref_off is not None
        else goal.new_zeros((len(goal), 29))
    )
    return f


@torch.no_grad()
def initialize(pi, bank, old_ids, target=0.95):
    counts = old_ids[:, 1].bincount(minlength=8)
    clone = int(counts.argmax())
    p = pi.moe.plugin
    p.gate_up.copy_(pi.moe.gate_up[clone])
    p.down.copy_(pi.moe.down[clone])
    h = bank["hidden"].float()
    threshold = bank["choice2"]
    raw1 = bank["logit1"]
    n = len(h)
    # Public selection offsets and raw mixture logits have different scales.
    # Fit the raw score to the public winner so an admitted token has useful
    # action weight; then calibrate ONE frozen new selection offset separately.
    ridge = 1e-3
    w = torch.linalg.solve(
        h.T @ h / n + ridge * torch.eye(h.shape[1], device=h.device),
        h.T @ raw1 / n,
    ).T
    records = []
    best = None
    for probability in [0.2, 0.35, 0.5]:
        bias = h.new_tensor(float(__import__("math").log(probability / (1 - probability))))
        raw = F.linear(h, w, bias.reshape(1))
        correction = -(raw - threshold).quantile(1 - target)
        choice = raw + correction
        sel = choice > threshold
        weight = torch.sigmoid(raw - raw1)
        active = weight[sel]
        useful = float(((active > 0.001) & (active < 0.999)).float().mean())
        admit = float(sel.float().mean())
        record = dict(
            kind="ridge_fit_public_winner_raw_score",
            ridge=ridge,
            bias=float(bias),
            fixed_selection_bias=float(correction),
            selection_fraction=admit,
            contribution_selected=float(active.mean()),
            contribution_all=float(torch.where(sel, weight, torch.zeros_like(weight)).mean()),
            unsaturated_selected_fraction=useful,
            raw_score_rmse=float((raw - bias - raw1).square().mean().sqrt()),
        )
        records.append(record)
        rank = (-useful, abs(float(active.mean()) - 0.35), abs(admit - target))
        if best is None or rank < best[0]:
            best = (rank, w, bias, correction, record)
    _, w, bias, correction, record = best
    if record["unsaturated_selected_fraction"] < 0.5 or record["contribution_all"] < 0.01:
        raise RuntimeError(
            "Initialization cannot produce useful admitted contributions: " + str(records)
        )
    p.router.weight.copy_(w)
    p.router.bias.copy_(bias.reshape(1))
    p.selection_bias.copy_(correction)
    pi.moe.domain_enabled = True
    return dict(
        clone_expert_zero_based=clone,
        clone_expert_human=clone + 1,
        old_second_slot_counts=counts.tolist(),
        fixed_new_selection_bias=float(p.selection_bias),
        calibration=record,
        candidate_probes=records,
        target_selection_fraction=target,
        forced_admission=False,
        selection_calibration="new buffer calibrated once; all old8 selection buffers frozen; raw mixture softmax unchanged",
    )
