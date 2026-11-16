"""Measure same-policy physical repeat variability, without changing the pilot."""

import json
from pathlib import Path
import sys
import torch
from experiments.martial_transfer_v1_20261106.prepare import ROOT, REPORT, save

from experiments.blink_competitive_martial_ppo_v1_20261106.routing import load_base
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from blinkmimic.dr import XiLayout
from blinkmimic.robot import get_robot


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    protocol = json.loads((REPORT / "protocol.json").read_text())
    data = json.loads((REPORT / "data_contract.json").read_text())
    pi, _, _ = load_base(protocol["checkpoints"]["base"]["file"])
    pi.moe.domain_enabled = False
    pi.requires_grad_(False)
    robot = get_robot("g1")
    kin = Kinematics(robot, "cuda:0")
    chosen = ["cmu__135_04", "pbhc__Bruce_Lee_pose"]
    rows = [next(r for r in data["rows"] if r["name"] == n) for n in chosen]
    motion = MotionLib([r["file"] for r in rows], kin, "cuda:0", robot.load_motion)
    mdp = MDP(kin, motion, pi.ref_q0, XiLayout(robot, "cuda:0").dim, robot.sensing)
    clips = torch.arange(2, device="cuda:0").repeat_interleave(4)
    env = TrackingSim(
        make_sim(robot), 8, robot, motion, mdp, "cuda:0", eval_mode=True, randomize=False
    )
    env.reset(torch.arange(8, device="cuda:0"), motion.clip_start[clips].clone())
    initial = env.state()
    initial_diffs = {
        n: max(float((x[i * 4 : (i + 1) * 4] - x[i * 4]).abs().max()) for x in initial)
        for i, n in enumerate(chosen)
    }
    ret = torch.zeros(8, device="cuda:0")
    h = None
    records = []
    with torch.inference_mode():
        for t in range(int(motion.clip_len.max()) - 1):
            obs = env.observe()
            h = (
                obs[:, None].repeat(1, 10, 1)
                if h is None
                else torch.cat((h[:, 1:], obs[:, None]), 1)
            )
            act = pi(h, motion.goal(env.f, 10))
            alive = ~env.done_eval.clone()
            out = env.step(torch.where(alive[:, None], act, torch.zeros_like(act)))
            ret += torch.where(alive, out["r"], 0)
            env.f = torch.minimum(env.f, motion.clip_end[clips] - 1)
            if t in (0, 4, 24, 99, 249, 499):
                state = env.state()
                records.append(
                    dict(
                        step=t + 1,
                        max_state_component_difference_between_identical_replicas={
                            n: max(
                                float((x[i * 4 : (i + 1) * 4] - x[i * 4]).abs().max())
                                for x in state
                            )
                            for i, n in enumerate(chosen)
                        },
                    )
                )
    report = dict(
        policy="same frozen public policy for all8 environments, no plugin",
        initial_state_replica_max_abs_error=initial_diffs,
        identical_replicas_per_clip=4,
        domain_randomization=False,
        reset_noise=False,
        action_noise=False,
        diagnostic_only=True,
        not_a_training_seed_experiment=True,
        divergence=records,
        normalized_returns={
            n: (ret[i * 4 : (i + 1) * 4] / ((motion.clip_len[i] - 1) * 0.1)).cpu().tolist()
            for i, n in enumerate(chosen)
        },
        conclusion="Empirical same-policy simulation repeat diagnostic; do not infer rollout equivalence solely from exact pointwise action equivalence.",
    )
    save(REPORT / "same_policy_repeat_diagnostic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
