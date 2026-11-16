"""Paired real/predicted-state routing checks, using identical recorded actions."""

import json
from pathlib import Path
import torch
from .mdp import unflatten_state


@torch.no_grad()
def diagnose(deployed, live, wm, mdp, motion, buffer, args, normalize_xi, initial):
    report = {
        "robot": "G1_29dof",
        "from_scratch": args.resume is None,
        "paired_actions": "identical recorded physical actions; masks stop at resets/pushes",
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "finite_parameters": all(
            bool(p.isfinite().all()) for m in (live, wm) for p in m.parameters()
        ),
        "parameter_change_norm": {
            n: float((p.cpu() - initial[n]).norm()) for n, p in live.moe.named_parameters()
        },
        "paired_routing": {},
    }
    report["per_expert_parameter_change_norm"] = {
        n: (p.cpu() - initial[n]).flatten(1).norm(dim=1).tolist()
        for n, p in live.moe.named_parameters()
        if n in ("gate_up", "down")
    }
    for horizon in (1, 8, 32):
        if buffer.size < horizon:
            continue
        w = buffer.sample_windows(256, horizon)
        state = unflatten_state(w["s"][:, 0])
        for t in range(horizon):
            state = wm.step(state, w["a"][:, t], w["a_prev"][:, t], normalize_xi(w["xi"][:, t]))
        true = unflatten_state(w["s1"][:, -1])
        frame = w["f"][:, -1] + 1
        frame = torch.minimum(frame, motion.clip_end[motion.clip_id[w["f"][:, -1]]])
        mask = w["mask"][:, -1].bool()
        if not bool(mask.any()):
            report["paired_routing"][str(horizon)] = {"valid_windows": 0}
            continue
        obs_true = mdp.obs(true, w["a"][:, -1], w["xi"][:, -1], frame)
        obs_pred = mdp.obs(state, w["a"][:, -1], w["xi"][:, -1], frame)
        p, kt = deployed.routing(obs_true)
        q, kp = deployed.routing(obs_pred)
        p, q, kt, kp = p[mask], q[mask], kt[mask], kp[mask]
        common = (kt[:, :, None] == kp[:, None, :]).any(-1).float().sum(-1) / 2
        mixture = (p + q) / 2
        js = (
            (p * (p.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(-1)
            + (q * (q.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(-1)
        ) / 2
        usage = torch.nn.functional.one_hot(kt, 8).float().mean((0, 1))
        lp, lk = live.routing(obs_true)
        live_usage = torch.nn.functional.one_hot(lk[mask], 8).float().mean((0, 1))
        report["paired_routing"][str(horizon)] = {
            "valid_windows": int(mask.sum()),
            "top2_overlap": float(common.mean()),
            "changed_top2_fraction": float((common < 1).float().mean()),
            "mean_js_divergence": float(js.mean()),
            "mean_total_variation": float((p - q).abs().sum(-1).mean() / 2),
            "real_top2_slot_usage": usage.tolist(),
            "active_experts": int((usage > 0).sum()),
            "maximum_slot_fraction": float(usage.max()),
            "live_policy_real_top2_slot_usage": live_usage.tolist(),
            "collapse_can_inflate_agreement": bool((usage > 0).sum() < 8 or usage.max() > 0.35),
            "mean_root_position_error_m": float((state[0] - true[0])[mask].norm(dim=-1).mean()),
            "mean_root_orientation_frobenius_error": float(
                (state[1] - true[1])[mask].norm(dim=(-2, -1)).mean()
            ),
        }
    s, h = buffer.sample_starts(128, args.hist)
    history = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    goal = motion.goal(s["f"], args.goal_frames)
    full, _ = deployed.action_and_aux(history, goal)
    ablated, _ = deployed.action_and_aux(history, goal, disable_routed=True)
    report["routed_branch_action_rms"] = float((full - ablated).square().mean().sqrt())
    path = Path(args.log_dir) / "moe_diagnostics.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"G1 MoE diagnostics saved to {path}", flush=True)
