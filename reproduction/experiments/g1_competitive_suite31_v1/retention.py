"""Frozen-base self-distillation on named original references; no new dataset mixed into the base."""

import torch
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from .routing_base import features


@torch.no_grad()
def make_bank(pi, robot, bridge, per_clip=32):
    motion = MotionLib.from_dir(robot.motion_dir, robot, bridge.mdp.kin, bridge.device)
    mdp = MDP(bridge.mdp.kin, motion, bridge.q0, bridge.mdp.xi_dim, robot.sensing)
    frames = torch.cat(
        [
            torch.linspace(
                int(s) + 10,
                max(int(s) + 10, int(e) - 11),
                per_clip,
                device=bridge.device,
            ).long()
            for s, e in zip(motion.clip_start, motion.clip_end)
        ]
    )
    saved = []
    enabled = pi.moe.domain_enabled
    pi.moe.domain_enabled = False
    for start in range(0, len(frames), 256):
        f = frames[start : start + 256]
        c = motion.clip_id[f]
        history = (f[:, None] + torch.arange(-9, 1, device=f.device)).clamp_min(0)
        history = torch.maximum(history, motion.clip_start[c, None])
        fs = history.flatten()
        S = motion.ref_state(fs)
        prev = (motion.q[fs] - pi.ref_q0) / pi.ref_alpha
        xi = torch.zeros(len(fs), mdp.xi_dim, device=f.device)
        obs = mdp.obs(S, prev, xi, fs).reshape(len(f), 10, 608)
        goal = motion.goal(f, 10)
        target = pi(obs, goal)
        saved.append(
            {
                k: v.detach().cpu()
                for k, v in dict(
                    **features(pi, goal),
                    target=target,
                    oldids=pi.moe.last_old_indices,
                ).items()
            }
        )
    pi.moe.domain_enabled = enabled
    return {k: torch.cat([b[k] for b in saved]) for k in saved[0]}
