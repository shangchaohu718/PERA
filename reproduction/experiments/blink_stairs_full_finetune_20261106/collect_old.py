"""Collect short real old-task rollouts under frozen-base and combined control."""

import argparse, json, time, hashlib
from pathlib import Path
import torch
from blinkmimic import networks
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from blinkmimic.dr import XiLayout
from experiments.blink_stairs_routing_ab_20261106.routing import load_base, features, public_digest
from experiments.blink_stairs_full_finetune_20261106.model import pack

from pera.paths import ROOT

R = ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(79201)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    pi, _, robot = load_base(R / "checkpoints/base/best.pt", domain=a.domain)
    pub = public_digest(pi)
    kin = Kinematics(robot, "cuda:0")
    motion = MotionLib.from_dir(robot.motion_dir, robot, kin, "cuda:0")
    mdp = MDP(kin, motion, pi.ref_q0, XiLayout(robot, "cuda:0").dim, robot.sensing)
    C = len(motion.names)
    assert C == 128
    N = C * 4
    env = TrackingSim(
        make_sim(robot), N, robot, motion, mdp, "cuda:0", eval_mode=True, randomize=False
    )
    clips = torch.arange(C, device="cuda:0").repeat_interleave(4)
    phase = torch.tensor([0.0, 0.25, 0.5, 0.75], device="cuda:0").repeat(C)
    frames = motion.clip_start[clips] + (phase * (motion.clip_len[clips] - 2)).long()
    ends = motion.clip_end[clips]
    saved = []
    stats = []
    with torch.no_grad():
        for controller in ["public", "combined"]:
            env.done_eval.zero_()
            env.failed_eval.zero_()
            env.reset(torch.arange(N, device="cuda:0"), frames.clone())
            history = None
            transitions = 0
            rows = 0
            for step in range(256):
                obs = env.observe()
                history = (
                    obs[:, None].repeat(1, 10, 1)
                    if history is None
                    else torch.cat((history[:, 1:], obs[:, None]), 1)
                )
                goal = motion.goal(env.f, 10)
                pi.moe.domain_enabled = True
                combined = pi(history, goal)
                f = features(pi, goal)
                with torch.autocast(
                    device_type="cuda",
                    dtype=networks.AMP_DTYPE or torch.bfloat16,
                    enabled=networks.AMP_DTYPE is not None,
                ):
                    target = pi.dec[-1](f["x"] + f["native"])
                target = target.float() + f["skip"]
                if step == 12:
                    pi.moe.domain_enabled = False
                    full_public = pi(history, goal)
                    pi.moe.domain_enabled = True
                    torch.testing.assert_close(target, full_public, atol=2e-5, rtol=2e-5)
                alive = ~env.done_eval.clone()
                transitions += int(alive.sum())
                if step >= 12 and step % 4 == 0:
                    ids = alive.nonzero().flatten()
                    ids = ids[torch.randperm(len(ids), device=ids.device)[:128]]
                    if len(ids):
                        item = {
                            k: v[ids].detach().cpu() for k, v in pack(history, goal, target).items()
                        }
                        item.update(
                            clip=clips[ids].cpu(),
                            controller=torch.full(
                                (len(ids),), int(controller == "combined"), dtype=torch.long
                            ),
                        )
                        saved.append(item)
                        rows += len(ids)
                env.step(target if controller == "public" else combined)
                env.f = torch.minimum(env.f, ends - 1)
                if step % 64 == 0:
                    print(json.dumps(dict(controller=controller, step=step, rows=rows)), flush=True)
            stats.append(
                dict(
                    controller=controller,
                    steps=256,
                    physical_transitions=transitions,
                    rows=rows,
                    remaining_alive=int((~env.done_eval).sum()),
                )
            )
    bank = {k: torch.cat([s[k] for s in saved]) for k in saved[0]}
    assert public_digest(pi) == pub
    assert len(bank["clip"].unique()) == 128, "Not all old motions represented"
    torch.save(bank, a.output / "bank.pt")
    report = dict(
        robot="g1",
        source_domain=str(a.domain),
        source_domain_sha256=hashlib.sha256(a.domain.read_bytes()).hexdigest(),
        public_sha256=pub,
        routing_mode=pi.moe.routing_mode,
        clips=motion.names,
        rows=len(bank["target"]),
        collection=stats,
        phase_fractions=[0, 0.25, 0.5, 0.75],
        segments_per_clip=4,
        segment_steps=256,
        history_warmup_steps=12,
        teacher="Frozen original eight-expert base queried on actual state histories",
        teacher_target_matches_full=True,
        stores="raw10-frame body history99, current router observation509, goal730, frozen-base target29; no learned feature cache",
        public_unchanged=True,
        scope="Training snippets only, not full-motion retention evaluation",
        elapsed_seconds=time.time() - started,
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}), flush=True)


if __name__ == "__main__":
    main()
