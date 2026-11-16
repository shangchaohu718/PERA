"""Raw old-motion reference histories with immutable-base action labels."""

import torch
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from experiments.blink_stairs_full_finetune_20261106.model import pack


@torch.no_grad()
def make_old_reference(teacher, robot, bridge, per_clip=32):
    motion = MotionLib.from_dir(robot.motion_dir, robot, bridge.mdp.kin, bridge.device)
    mdp = MDP(bridge.mdp.kin, motion, teacher.ref_q0, bridge.mdp.xi_dim, robot.sensing)
    frames = torch.cat(
        [
            torch.linspace(
                int(s) + 10, max(int(s) + 10, int(e) - 11), per_clip, device=bridge.device
            ).long()
            for s, e in zip(motion.clip_start, motion.clip_end)
        ]
    )
    saved = []
    for offset in range(0, len(frames), 256):
        f = frames[offset : offset + 256]
        clip = motion.clip_id[f]
        history = torch.maximum(
            f[:, None] + torch.arange(-9, 1, device=f.device), motion.clip_start[clip, None]
        )
        fs = history.flatten()
        state = motion.ref_state(fs)
        prev = (motion.q[fs] - teacher.ref_q0) / teacher.ref_alpha
        xi = torch.zeros(len(fs), mdp.xi_dim, device=f.device)
        h = mdp.obs(state, prev, xi, fs).reshape(len(f), 10, 608)
        g = motion.goal(f, 10)
        target = teacher(h, g)
        saved.append({k: v.cpu() for k, v in pack(h, g, target).items()})
    return {k: torch.cat([b[k] for b in saved]) for k in saved[0]}
