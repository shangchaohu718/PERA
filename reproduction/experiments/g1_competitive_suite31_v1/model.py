"""Current G1 MLP base and competitive task-expert observation/action adapter.

No legacy Transformer weights are loaded into this student. Legacy adapters
supply named-joint physical state/reference and frozen task context only.
"""

import hashlib
import json
import math
from types import SimpleNamespace
from .runtime_deps import ensure_fk_parser

ensure_fk_parser()

import torch
from torch import nn
from blinkmimic.rotations import quat_to_mat, mat_to_6d, yaw_mat, so3_log
from . import routing_base as base

from pera.paths import ROOT

FORMAT = "blink_g1_competitive_suite31_v1"
CAPTURE = {
    "inputs": None,
    "version": 0,
    "positions": None,
    "sampled_positions": None,
}


def install_capture():
    # Process-local: existing files and other running processes are untouched.
    import modumotion_g1.perceptive_observations as perceptive
    import holomotion.src.motion_tracking.actor_observation as official

    original_sample = perceptive.sample_holomotion_reference

    def sample(raw, frame, phase, start, end, **kw):
        result = original_sample(raw, frame, phase, start, end, **kw)
        dt = kw.get("actor_dt", 0.02)
        fps = kw["reference_fps"]
        offsets = (
            torch.arange(-1, kw.get("num_future_frames", 10) + 2, device=frame.device) * dt * fps
        )
        CAPTURE["sampled_positions"] = (
            (frame[:, None] + phase[:, None] + offsets)
            .maximum(start[:, None])
            .minimum(end[:, None])
        )
        return result

    perceptive.sample_holomotion_reference = sample
    for mod, name in [
        (perceptive, "build_perceptive_actor_terms"),
        (official, "derive_motion_actor_terms_torch"),
    ]:
        original = getattr(mod, name)

        def wrapped(inputs, *args, _original=original, **kw):
            CAPTURE.update(
                inputs=inputs,
                version=CAPTURE["version"] + 1,
                positions=CAPTURE["sampled_positions"],
                sampled_positions=None,
            )
            return _original(inputs, *args, **kw)

        setattr(mod, name, wrapped)
    import modumotion_g1.domain_moe as dm

    for name in [
        "set_domain_context",
        "set_domain_enabled",
        "set_active_domains",
        "set_trainable_domain",
    ]:
        original = getattr(dm, name)

        def dispatch(actor, *args, _name=name, _original=original, **kw):
            if not isinstance(actor, Actor):
                return _original(actor, *args, **kw)
            if getattr(actor, "_expert_bank", None) is not None:
                return actor._dispatch_bank_control(_name, *args, **kw)
            if _name == "set_domain_context":
                ctx = args[0]
                actor.context = next(iter(ctx.values())) if isinstance(ctx, dict) and ctx else ctx
            elif _name == "set_domain_enabled":
                actor.pi.moe.domain_enabled = bool(args[1])
            elif _name == "set_active_domains":
                actor.pi.moe.domain_enabled = bool(args[0])

        setattr(dm, name, dispatch)


from pera.context import ContextPlugin, ContextMoE


class Actor(nn.Module):
    def __init__(self, checkpoint, context_dim, domain, device="cuda:0", *, bank=None):
        super().__init__()
        # Register the shared policy exactly once in this adapter's state tree.
        object.__setattr__(self, "_expert_bank", bank)
        if bank is None:
            self.pi, self.cfg, self.robot = base.load_base(checkpoint, device)
            self.pi.moe.__class__ = ContextMoE
            self.pi.moe.plugin = ContextPlugin(context_dim).to(device)
            self.pi.moe.runtime_context = None
        else:
            self.pi, self.cfg, self.robot = bank.pi, bank.args, bank.robot
        self.context = None
        self.domain = domain
        self.history = None
        self.pending = None
        self.version = -1
        c = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())
        names = list(self.robot.joint_names)
        self.register_buffer(
            "sdk_from_holo",
            torch.tensor([c["joint_names"].index(n) for n in names], device=device),
        )
        self.register_buffer(
            "holo_from_sdk",
            torch.tensor([names.index(n) for n in c["joint_names"]], device=device),
        )
        self.register_buffer("old_offset", torch.tensor(c["action_offset"], device=device))
        self.register_buffer("old_scale", torch.tensor(c["action_scale"], device=device))
        router_contract = json.loads((ROOT / "configs/data/train128.json").read_text())
        self.register_buffer(
            "router_from_sdk",
            torch.tensor(router_contract["holomotion_from_sdk"], device=device),
        )
        self.context_dim = context_dim
        self.proxy = SimpleNamespace(
            domain_plugins=self.pi.moe.plugins
            if bank is not None
            else {domain: self.pi.moe.plugin},
            active_domains=() if bank is not None else (domain,),
            last_group_weights=None,
        )
        self.public_sha256 = base.public_digest(self.pi)

    @property
    def context(self):
        bank = self._expert_bank
        if bank is not None and self._context_generation != bank.switch_generation:
            return None
        return self._context

    @context.setter
    def context(self, value):
        bank = self._expert_bank
        if bank is not None and isinstance(value, dict):
            if bank.active_expert is None:
                value = None
            elif set(value) != {bank.active_expert}:
                raise ValueError("Provide context only for the active named expert")
            else:
                value = value[bank.active_expert]
        self._context = value
        self._context_generation = bank.switch_generation if bank is not None else -1

    def _dispatch_bank_control(self, name, *args, **kw):
        bank = self._expert_bank
        if name == "set_domain_context":
            self.context = args[0]
        elif name == "set_domain_enabled":
            bank.set_expert_enabled(args[0], args[1])
        elif name == "set_active_domains":
            names = tuple(args[0])
            if len(names) > 1:
                raise ValueError("At most one task expert can be enabled")
            bias = args[1] if len(args) > 1 else kw.get("merge_bias")
            if bias is not None and (set(bias) != set(names) or any(v != 0 for v in bias.values())):
                raise ValueError("The expert bank preserves trained routing scores and offsets")
            bank.select_expert(names[0] if names else None)
        elif name == "set_trainable_domain":
            raise RuntimeError("A composed expert bank is frozen and inference-only")
        else:
            raise ValueError("Unsupported expert-bank control")
        self.proxy.active_domains = (bank.active_expert,) if bank.active_expert else ()

    @property
    def actor_module(self):
        return SimpleNamespace(layers=[self.proxy])

    def reset_kv_cache(self, *args):
        self.history = None
        self.pending = None
        self.version = -1

    def clear_env_cache(self, ids):
        if ids is None:
            self.reset_kv_cache()
        elif self.pending is not None:
            self.pending[ids] = True

    def old_to_sdk(self, a):
        return (
            (self.old_offset + self.old_scale * a).index_select(-1, self.sdk_from_holo)
            - self.pi.ref_q0
        ) / self.pi.ref_alpha

    def sdk_to_old(self, a):
        return (
            (self.pi.ref_q0 + self.pi.ref_alpha * a).index_select(-1, self.holo_from_sdk)
            - self.old_offset
        ) / self.old_scale

    def inputs(self, raw):
        ref = raw.reference_qpos.float()
        idx = getattr(raw, "reference_dof_indices", None)
        jq = ref[..., 7:] if idx is None else ref[..., 7:].index_select(-1, idx.long())
        jq = jq.index_select(-1, self.sdk_from_holo)
        p = ref[..., :3]
        R = quat_to_mat(ref[..., 3:7])
        if ref.shape[1] != 13:
            raise ValueError("Expected13 reference frames at50Hz")
        if getattr(raw, "reference_yaw_alignment_wxyz", None) is not None:
            raise ValueError("Explicit yaw-alignment adapter required; refusing silent mismatch")
        # Same central SO(3)/position/joint differences as native Blink MotionLib.
        v = (p[:, 2:] - p[:, :-2]) / 0.04
        qd = (jq[:, 2:] - jq[:, :-2]) / 0.04
        w = so3_log(R[:, 2:] @ R[:, :-2].transpose(-1, -2)) / 0.04
        positions = getattr(raw, "blink_reference_positions", CAPTURE.get("positions"))
        if positions is None:
            before = (ref[:, 1:-1] - ref[:, :-2]).abs().amax(-1, keepdim=True) < 1e-8
            after = (ref[:, 2:] - ref[:, 1:-1]).abs().amax(-1, keepdim=True) < 1e-8
        else:
            before = (positions[:, 1:-1] == positions[:, :-2])[..., None]
            after = (positions[:, 2:] == positions[:, 1:-1])[..., None]

        def edge(x, arr):
            return torch.where(
                before,
                (arr[:, 2:] - arr[:, 1:-1]) / 0.02,
                torch.where(after, (arr[:, 1:-1] - arr[:, :-2]) / 0.02, x),
            )

        v = edge(v, p)
        qd = edge(qd, jq)
        w = torch.where(
            before,
            so3_log(R[:, 2:] @ R[:, 1:-1].transpose(-1, -2)) / 0.02,
            torch.where(
                after,
                so3_log(R[:, 1:-1] @ R[:, :-2].transpose(-1, -2)) / 0.02,
                w,
            ),
        )
        if positions is not None:
            # Native MotionLib repeats the LAST FRAME'S velocity in its clamped
            # future goals; it does not replace that velocity with zero.
            v = v.clone()
            qd = qd.clone()
            w = w.clone()
            for j in range(1, 11):
                same = (positions[:, j + 1] == positions[:, j])[:, None]
                v[:, j] = torch.where(same, v[:, j - 1], v[:, j])
                qd[:, j] = torch.where(same, qd[:, j - 1], qd[:, j])
                w[:, j] = torch.where(same, w[:, j - 1], w[:, j])
        rl = R[:, 1:12]
        vl = (rl.transpose(-1, -2) @ v[..., None])[..., 0]
        wl = (rl.transpose(-1, -2) @ w[..., None])[..., 0]
        rr = quat_to_mat(raw.robot_root_quat_wxyz.float())
        Rt = rr.transpose(-1, -2)
        q = raw.robot_dof_pos.index_select(-1, self.sdk_from_holo)
        dq = raw.robot_dof_vel.index_select(-1, self.sdk_from_holo)
        prev = self.old_to_sdk(raw.last_action)
        body = torch.cat(
            (
                q - self.pi.ref_q0,
                dq,
                raw.robot_root_angvel_local,
                -rr[:, 2, :],
                prev,
                mat_to_6d(Rt @ R[:, 1]),
            ),
            -1,
        )
        fut = R[:, 2:12]
        cur = R[:, 1]
        y = torch.atan2(cur[:, 1, 0], cur[:, 0, 0])
        yf = torch.atan2(fut[:, :, 1, 0], fut[:, :, 0, 0])
        delta = yf - y[:, None]
        router = torch.cat(
            (
                -cur[:, 2, :],
                vl[:, 0],
                wl[:, 0],
                jq[:, 1].index_select(-1, self.router_from_sdk),
                p[:, 1, 2:3],
                jq[:, 2:12].index_select(-1, self.router_from_sdk).flatten(1),
                p[:, 2:12, 2].flatten(1),
                -fut[:, :, 2, :].flatten(1),
                vl[:, 1:].flatten(1),
                wl[:, 1:].flatten(1),
                torch.stack((delta.sin(), delta.cos()), -1).flatten(1),
                (Rt[:, None] @ fut)[..., :, :2].flatten(1),
            ),
            -1,
        )
        Ht = yaw_mat(cur).transpose(-1, -2)[:, None]
        goal = torch.cat(
            (
                (Ht @ (p[:, 2:12] - p[:, 1:2])[..., None])[..., 0],
                mat_to_6d(Ht @ fut),
                vl[:, 1:],
                wl[:, 1:],
                jq[:, 2:12],
                qd[:, 1:] * 0.1,
            ),
            -1,
        ).flatten(1)
        obs = torch.cat((body, router), -1)
        assert obs.shape[-1] == 608 and goal.shape[-1] == 730
        return obs, goal

    def forward(self, td=None, mode="inference", **kw):
        if self._expert_bank is not None:
            self.domain = self._expert_bank.active_expert or ""
            self.context_dim = self._expert_bank.context_dim
        if CAPTURE["inputs"] is None:
            raise RuntimeError("Native reference capture missing")
        obs, goal = self.inputs(CAPTURE["inputs"])
        if self.history is None:
            self.history = obs[:, None].repeat(1, 10, 1)
            self.pending = torch.zeros(len(obs), dtype=torch.bool, device=obs.device)
        elif self.version != CAPTURE["version"]:
            self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
            self.history[self.pending] = obs[self.pending, None].expand(-1, 10, -1)
            self.pending.zero_()
        self.version = CAPTURE["version"]
        if self._expert_bank is not None:
            mu = self._expert_bank(self.history, goal, self.context)
        else:
            self.pi.moe.runtime_context = self.context
            mu = self.pi(self.history, goal)
        self.cached = base.features(self.pi, goal)
        wt = self.pi.moe.last_weight
        self.proxy.last_group_weights = torch.cat((1 - wt, wt), -1)[:, None]
        self.proxy.active_domains = (self.domain,) if self.pi.moe.domain_enabled else ()
        return {"actions": self.sdk_to_old(mu)}

    @torch.no_grad()
    def initialize(self, bank, target=0.95):
        if self._expert_bank is not None:
            raise RuntimeError("A composed expert bank is frozen and inference-only")
        p = self.pi.moe.plugin
        m = self.pi.moe
        counts = bank["oldids"][:, 1].bincount(minlength=8)
        clone = int(counts.argmax())
        p.gate_up.zero_()
        p.gate_up[:256].copy_(m.gate_up[clone])
        p.down.copy_(m.down[clone])
        h = bank["hidden"].float()
        n = len(h)
        ridge = 0.001
        w = torch.linalg.solve(
            h.T @ h / n + ridge * torch.eye(256, device=h.device),
            h.T @ bank["logit1"] / n,
        ).T
        p.router.weight.zero_()
        p.router.weight[:, :256].copy_(w)
        p.router.bias.fill_(math.log(0.35 / 0.65))
        raw = p.score(torch.cat((h, bank["context"]), -1))
        p.selection_bias.copy_(-(raw - bank["choice2"]).quantile(1 - target))
        _, sel, wt, _ = m.compete(bank)
        if float(wt.mean()) < 0.01:
            raise RuntimeError("New expert initial contribution is negligible")
        return dict(
            cloned_public_expert=clone,
            initial_admission=float(sel.float().mean()),
            initial_mean_contribution=float(wt.mean()),
            new_selection_bias=float(p.selection_bias),
            forced_selection=False,
        )

    def save(self, path, updates, optimizer, initialization):
        if self._expert_bank is not None:
            raise RuntimeError("Use the expert bank's save_bundle for inference export")
        torch.save(
            dict(
                format=FORMAT,
                base_sha256=hashlib.sha256(
                    (ROOT / "checkpoints/base/best.pt").read_bytes()
                ).hexdigest(),
                public_sha256=self.public_sha256,
                domain=self.domain,
                context_dim=self.context_dim,
                context_contract=getattr(self, "context_contract", None),
                plugin={k: v.detach().cpu() for k, v in self.pi.moe.plugin.state_dict().items()},
                actual_Adam_updates=updates,
                optimizer=optimizer.state_dict() if optimizer else None,
                initialization=initialization,
                robot="g1",
                routing="8 frozen public +1 new real Top2",
                world_model=False,
            ),
            path,
        )


class Features:
    def __init__(self, actor, domain=None):
        self.actor = actor
        self.features = {}

    def snapshot(self, target, actual_mu, context=None):
        f = dict(self.actor.cached)
        f["target"] = self.actor.old_to_sdk(target)
        f["oldids"] = self.actor.pi.moe.last_old_indices
        pred, *_ = self.actor.pi.moe.predict(f, self.actor.pi.dec[-1])
        torch.testing.assert_close(self.actor.sdk_to_old(pred), actual_mu, atol=3e-5, rtol=3e-5)
        return {k: v.detach().cpu().clone() for k, v in f.items()}

    def close(self):
        pass
