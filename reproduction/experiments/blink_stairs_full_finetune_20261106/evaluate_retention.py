"""Evaluate a competitive plugin on the unchanged128-motion native G1 suite."""

import argparse, json, time
from pathlib import Path
import torch
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP, MAX_STEP_REWARD
from blinkmimic.dr import XiLayout
from experiments.blink_stairs_full_finetune_20261106.model import load_actor, public_digest
from pera.paths import ROOT

R = ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    t = time.time()
    pi, cfg, robot = load_actor(full=a.domain)
    pub = public_digest(pi)
    kin = Kinematics(robot, "cuda:0")
    m = MotionLib.from_dir(robot.motion_dir, robot, kin, "cuda:0")
    mdp = MDP(kin, m, pi.ref_q0, XiLayout(robot, "cuda:0").dim, robot.sensing)
    N = len(m.names)
    env = TrackingSim(make_sim(robot), N, robot, m, mdp, "cuda:0", eval_mode=True, randomize=False)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(N, device="cuda:0"), m.clip_start.clone())
    h = None
    ret = torch.zeros(N, device="cuda:0")
    steps = ret.clone()
    route = torch.zeros(N, 9, device="cuda:0")
    contrib = ret.clone()
    frozen = pi.moe.selection_bias.clone()
    newbias = pi.moe.plugin.selection_bias.clone()
    significant = ret.clone()
    with torch.no_grad():
        while not bool(env.done_eval.all()):
            o = env.observe()
            h = o[:, None].repeat(1, 10, 1) if h is None else torch.cat((h[:, 1:], o[:, None]), 1)
            act = pi(h, m.goal(env.f, 10))
            alive = ~env.done_eval.clone()
            ids = pi.moe.last_old_indices.clone()
            ids[:, 1] = torch.where(
                pi.moe.last_selected[:, 0] & (pi.moe.routing_mode == "competitive"), 8, ids[:, 1]
            )
            route += torch.nn.functional.one_hot(ids, 9).sum(1) * alive[:, None]
            contrib += pi.moe.last_weight[:, 0] * alive
            significant += (pi.moe.last_weight[:, 0] > 0.1) * alive
            out = env.step(act)
            ret += out["r"] * alive
            steps += alive
            env.f = torch.minimum(env.f, m.clip_end - 1)
        norm = ret / ((m.clip_len - 1) * MAX_STEP_REWARD)
        failed = env.failed_eval.cpu()
        result = dict(
            robot="g1",
            routing_mode=pi.moe.routing_mode,
            plugin=str(a.domain),
            clips=N,
            passed=N - int(failed.sum()),
            norm_return=float(norm.mean()),
            per_clip={n: round(float(v), 4) for n, v in zip(m.names, norm)},
            failed=[n for n, z in zip(m.names, failed) if z],
            route_slot_usage_frameweighted=(route.sum(0) / route.sum()).tolist(),
            new_expert_selection_fraction=float(route[:, 8].sum() / steps.sum())
            if pi.moe.routing_mode == "competitive"
            else None,
            new_expert_selection_semantics="Top2 admission; undefined for continuous fusion",
            new_expert_weight_greater_than_01_fraction=None,
            mean_new_contribution=float(contrib.sum() / steps.sum()),
            old_bias_frozen=bool(torch.equal(frozen, pi.moe.selection_bias)),
            new_bias_frozen=bool(torch.equal(newbias, pi.moe.plugin.selection_bias)),
            parameters_unchanged_during_evaluation=pub == public_digest(pi),
            elapsed_seconds=time.time() - t,
        )
    result["new_expert_weight_greater_than_01_fraction"] = float(significant.sum() / steps.sum())
    a.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["per_clip", "failed"]}, indent=2)
    )


if __name__ == "__main__":
    main()
