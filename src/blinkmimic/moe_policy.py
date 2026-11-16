"""Compact Blink policy with the frozen K1 reference-router and loss contract.

Robot data and action order are G1 SDK order. Only the reference-router feature
order follows the saved HoloMotion G1 configuration. No K1 robot state is used.
"""

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from . import networks

from pera.paths import ROOT

EXP = ROOT
CONTRACT = json.loads((ROOT / "configs/router.json").read_text())
_bank = Path(CONTRACT["selection_calibration"]["bank"])
if not _bank.is_absolute():
    CONTRACT["selection_calibration"]["bank"] = str(ROOT / _bank)


class RouterNorm(nn.Module):
    def __init__(self, width=509):
        super().__init__()
        cfg = CONTRACT["obs_norm"]
        self.momentum = cfg["ema_momentum"]
        self.eps, self.clip = cfg["epsilon"], cfg["clip_range"]
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("ex2", torch.ones(width))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, 509).float()
        self.mean.lerp_(x.mean(0), self.momentum)
        self.ex2.lerp_(x.square().mean(0), self.momentum)
        self.var.copy_((self.ex2 - self.mean.square()).clamp_min(0))
        self.count.add_(x.shape[0])

    def forward(self, x):
        return ((x - self.mean) / (self.var.clamp_min(0).sqrt() + self.eps)).clamp(
            -self.clip, self.clip
        )


class SplitNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = networks.RunningNorm(99)
        self.router = RouterNorm()

    def update(self, x):
        self.body.update(x[..., :99])
        self.router.update(x[..., 99:])

    def forward(self, x):
        return torch.cat((self.body(x[..., :99]), self.router(x[..., 99:])), -1)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


def router_losses(logits, topk, choice_scores=None):
    """Same switch-style balance and inactive margin equations as K1."""
    p = logits.float().softmax(-1)
    mask = F.one_hot(topk, num_classes=8).sum(-2).float()
    hard = mask.mean(0) / 2
    dense = p.mean(0)
    balance = 8 * (hard.detach() * dense).sum()
    inactive = (
        hard
        < hard.max().clamp_min(1e-12) * CONTRACT["inactive_expert_margin_to_topk"]["ratio_floor"]
    ).to(logits.dtype)
    choice_scores = logits if choice_scores is None else choice_scores
    kth = choice_scores.gather(-1, topk)[..., -1:]
    gap = (kth - choice_scores).relu()
    margin = (gap * inactive[None]).sum() / (inactive.sum().clamp_min(1) * logits.shape[0])
    return balance, margin, hard, dense


class ReferenceMoE(nn.Module):
    def __init__(self):
        super().__init__()
        d, h = CONTRACT["width"], CONTRACT["expert_hidden"]
        self.norm = RMSNorm(d)
        self.router_embed = nn.Sequential(
            nn.Linear(509, 512), nn.SiLU(), nn.Linear(512, d), RMSNorm(d)
        )
        self.router = nn.Linear(d, 8, bias=False)
        self.register_buffer("selection_bias", torch.zeros(8))
        self.gate_up = nn.Parameter(torch.empty(8, d, h * 2))
        self.down = nn.Parameter(torch.empty(8, h, d))
        self.shared_up = nn.Linear(d, h * 2, bias=False)
        self.shared_down = nn.Linear(h, d, bias=False)
        for expert in range(8):
            nn.init.xavier_uniform_(self.gate_up[expert])
            nn.init.xavier_uniform_(self.down[expert])

    def logits(self, router_features):
        # Casting only the result cannot recover rank differences lost in bf16.
        with torch.autocast(device_type=router_features.device.type, enabled=False):
            return self.router(self.router_embed(router_features.float()))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A legacy checkpoint has the same trainable parameters and zero correction.
        state_dict.setdefault(prefix + "selection_bias", torch.zeros_like(self.selection_bias))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def select(self, logits):
        return (logits + self.selection_bias).topk(2, dim=-1).indices

    def forward(self, x, router_features, routed=True):
        x = self.norm(x)
        # bf16 matrix arithmetic, fp32 router probabilities and regularizers.
        device = x.device.type
        with torch.autocast(
            device_type=device,
            dtype=networks.AMP_DTYPE or torch.bfloat16,
            enabled=networks.AMP_DTYPE is not None,
        ):
            logits = self.logits(router_features)
            topk = self.select(logits)
            weights = logits.gather(-1, topk).softmax(-1)
            mask = torch.zeros_like(logits).scatter(-1, topk, weights)
            # Fixed-shape masked Top2. All eight small FFNs are evaluated to
            # avoid dynamic expert dispatch/host sync in CUDA graph capture.
            gate, up = torch.einsum("bd,edi->bei", x, self.gate_up).chunk(2, -1)
            outputs = torch.einsum("beh,ehd->bed", F.silu(gate) * up, self.down)
            sparse = (outputs * mask.to(outputs.dtype).unsqueeze(-1)).sum(1)
            gate, up = self.shared_up(x).chunk(2, -1)
            shared = self.shared_down(F.silu(gate) * up)
        balance, margin, hard, dense = router_losses(logits, topk, logits + self.selection_bias)
        ratio = sparse.float().square().mean() / (shared.float().square().mean().clamp_min(1e-12))
        statistics = torch.cat((torch.stack((balance, margin)), hard, dense, ratio[None]))
        return shared + (sparse if routed else torch.zeros_like(sparse)), statistics


class Policy(networks.DensePolicy):
    def __init__(
        self,
        obs_dim,
        goal_dim,
        act_dim,
        hist=10,
        z_dim=32,
        enc_hidden=(2048, 1024, 512, 256),
        dec_hidden=(2048, 1024, 512, 256),
        ref_skip=None,
    ):
        if (obs_dim, act_dim) != (608, 29) or dec_hidden[-1] != 256:
            raise ValueError(
                "Expected G1 compact99+K1-router509,29 actions,256-wide final hidden state"
            )
        super().__init__(99, goal_dim, act_dim, hist, z_dim, enc_hidden, dec_hidden, ref_skip)
        self.obs_norm = SplitNorm()
        self.moe = ReferenceMoE()

    def action_and_aux(self, obs_hist, goal, z=None, disable_routed=False):
        o = self.obs_norm.body(obs_hist[..., :99]).flatten(-2)
        r = self.obs_norm.router(obs_hist[:, -1, 99:])
        if z is None:
            z = self.encode(goal)
        x = networks.run_mlp(self.dec[:-1], torch.cat((z, o), -1))
        branch, statistics = self.moe(x, r, routed=not disable_routed)
        with torch.autocast(
            device_type=x.device.type,
            dtype=networks.AMP_DTYPE or torch.bfloat16,
            enabled=networks.AMP_DTYPE is not None,
        ):
            a = self.dec[-1](x + branch)
        if self.ref_off is not None:
            q_ref = goal[..., self.ref_off : self.ref_off + a.shape[-1]]
            a = a + (q_ref - self.ref_q0) / self.ref_alpha
        return a.float(), statistics

    def forward(self, obs_hist, goal, z=None):
        return self.action_and_aux(obs_hist, goal, z)[0]

    def routing(self, observation):
        r = self.obs_norm.router(observation[..., 99:])
        logits = self.moe.logits(r)
        return logits.softmax(-1), self.moe.select(logits)


def reference_router_features(motion, state, frame, holo_from_sdk):
    """K1's12 reference terms,509 values, preserving robot-relative SO(3)."""
    shape = frame.shape
    f = frame.long().reshape(-1)
    robot_R = state[1].reshape(-1, 3, 3)
    future = motion.future_frames(f, 10)
    Rf = motion.root_R[f]
    future_R = motion.root_R[future]
    yaw = torch.atan2(Rf[..., 1, 0], Rf[..., 0, 0])
    yaw_f = torch.atan2(future_R[..., 1, 0], future_R[..., 0, 0])
    delta = yaw_f - yaw[:, None]
    relative = robot_R.transpose(-1, -2)[:, None] @ future_R
    values = [
        -Rf[..., 2, :],
        motion.root_v_local[f],
        motion.root_w_local[f],
        motion.q[f].index_select(-1, holo_from_sdk),
        motion.root_p[f, 2:3],
        motion.q[future].index_select(-1, holo_from_sdk).flatten(1),
        motion.root_p[future, 2].flatten(1),
        -future_R[..., 2, :].flatten(1),
        motion.root_v_local[future].flatten(1),
        motion.root_w_local[future].flatten(1),
        torch.stack((delta.sin(), delta.cos()), -1).flatten(1),
        relative[..., :, :2].flatten(1),
    ]
    return torch.cat(values, -1).reshape(*shape, 509)
