"""Re-evaluate completion only on the two existing held-out pools, without training."""

import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from pera.paths import ROOT

from experiments.blink_competitive_martial_ppo_v1_20261106.routing import load_base, public_digest
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from blinkmimic.dr import XiLayout
from blinkmimic.robot import get_robot


def sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol-input", type=Path)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.6)
    start = time.time()
    source = ROOT / "reports/martial_transfer_v1_20261106"
    old_contract = ROOT / "resources/base/data/heldout_contract.json"
    previous = json.loads((source / "protocol.json").read_text())
    checkpoints = {k: previous["checkpoints"][k] for k in ("base", "trained")}
    if args.protocol_input:
        selected = json.loads(args.protocol_input.read_text())
        assert selected["robot"] == "unitree_g1_29dof" and selected["training"] is False
        rows = selected["rows"]
        checkpoints = selected["checkpoints"]
        replicas = int(selected.get("replicas_per_clip", 4))
    else:
        rows = []
        for row in json.loads(old_contract.read_text())["clips"]:
            rows.append(dict(row, name=Path(row["file"]).stem, pool="original_heldout16"))
        old_new = json.loads((source / "data_contract.json").read_text())
        rows.extend(r for r in old_new["rows"] if r["pool"] in ("unseen_related", "unseen_other"))
        assert len(rows) == 48
        replicas = 4
    pool_names = list(dict.fromkeys(row["pool"] for row in rows))
    for row in rows:
        assert sha(row["file"]) == row["sha256"]
    for v in checkpoints.values():
        assert sha(v["file"]) == v["sha256"]
    protocol = dict(
        robot="unitree_g1_29dof",
        comparison=["base", "base_plus_trained_martial_expert"],
        checkpoints=checkpoints,
        existing_pool_counts={pool: sum(r["pool"] == pool for r in rows) for pool in pool_names},
        rows=rows,
        replicas_per_clip=replicas,
        initialization="phase0 exact reference pose and velocity; no perturbation",
        randomization=False,
        action_noise=False,
        training=False,
        new_dataset_selection=bool(args.protocol_input),
        prepared_input_protocol=str(args.protocol_input) if args.protocol_input else None,
        metrics=dict(
            mean_completion="min(actual control steps / full resampled reference transitions,1)",
            whole_clip_completion_rate="fraction reaching reference end without physical termination",
        ),
        note="Physical replicas of each clip are evaluation repeats, not independent training seeds; pools reported separately.",
    )
    save(out / "protocol.json", protocol)
    all_results = {}
    initial_bank = None
    for condition in ("base", "trained"):
        save(
            out / "status.json",
            dict(
                stage="evaluating",
                condition=condition,
                pid=os.getpid(),
                elapsed_seconds=time.time() - start,
            ),
        )
        pi, cfg, _ = load_base(
            checkpoints["base"]["file"],
            domain=checkpoints["trained"]["file"] if condition == "trained" else None,
        )
        pi.requires_grad_(False)
        pi.moe.domain_enabled = condition == "trained"
        public_before = public_digest(pi)
        robot = get_robot("g1")
        kin = Kinematics(robot, "cuda:0")
        m = MotionLib([r["file"] for r in rows], kin, "cuda:0", robot.load_motion)
        mdp = MDP(kin, m, pi.ref_q0, XiLayout(robot, "cuda:0").dim, robot.sensing)
        clip_ids = torch.arange(len(rows), device="cuda:0").repeat_interleave(replicas)
        N = len(clip_ids)
        env = TrackingSim(
            make_sim(robot), N, robot, m, mdp, "cuda:0", eval_mode=True, randomize=False
        )
        env.reset(torch.arange(N, device="cuda:0"), m.clip_start[clip_ids].clone())
        init = [x.clone().cpu() for x in env.state()]
        if initial_bank is None:
            initial_bank = init
        else:
            assert all(torch.equal(a, b) for a, b in zip(initial_bank, init)), (
                "Initial states differ"
            )
        np.savez_compressed(
            out / (condition + "_initial_states.npz"),
            **{str(i): x.numpy() for i, x in enumerate(init)},
        )
        counts = torch.zeros(N, dtype=torch.long, device="cuda:0")
        h = None
        with torch.inference_mode():
            for t in range(int(m.clip_len.max())):
                alive = ~env.done_eval.clone()
                if not bool(alive.any()):
                    break
                obs = env.observe()
                h = (
                    obs[:, None].repeat(1, 10, 1)
                    if h is None
                    else torch.cat((h[:, 1:], obs[:, None]), 1)
                )
                actions = pi(h, m.goal(env.f, 10))
                assert bool(torch.isfinite(actions[alive]).all())
                env.step(torch.where(alive[:, None], actions, torch.zeros_like(actions)))
                counts += alive
                env.f = torch.minimum(env.f, m.clip_end[clip_ids] - 1)
                if (t + 1) % 500 == 0:
                    save(
                        out / "status.json",
                        dict(
                            stage="evaluating",
                            condition=condition,
                            pid=os.getpid(),
                            control_step=t + 1,
                            maximum_control_steps=int(m.clip_len.max()),
                            active_environments=int((~env.done_eval).sum()),
                            elapsed_seconds=time.time() - start,
                        ),
                    )
        assert bool(env.done_eval.all()) and public_digest(pi) == public_before
        samples = []
        for i, row in enumerate(rows):
            length = int(m.clip_len[i]) - 1
            for replica in range(replicas):
                j = i * replicas + replica
                samples.append(
                    dict(
                        name=row["name"],
                        pool=row["pool"],
                        replica=replica,
                        reference_control_steps=length,
                        executed_control_steps=int(counts[j]),
                        completion_fraction=min(float(counts[j]) / length, 1.0),
                        full_clip_completed=bool(env.done_eval[j] and not env.failed_eval[j]),
                        terminated_early=bool(env.failed_eval[j]),
                        reference_duration_seconds=length * 0.02,
                        executed_seconds=int(counts[j]) * 0.02,
                    )
                )
        all_results[condition] = dict(
            samples=samples, public_sha256=public_before, public_unchanged=True
        )
        save(out / (condition + ".json"), all_results[condition])
        del pi, env, m, kin, mdp
        torch.cuda.empty_cache()
    paired = []
    for i, row in enumerate(rows):
        before = all_results["base"]["samples"][i * replicas : (i + 1) * replicas]
        after = all_results["trained"]["samples"][i * replicas : (i + 1) * replicas]
        b = float(np.mean([r["completion_fraction"] for r in before]))
        a = float(np.mean([r["completion_fraction"] for r in after]))
        pb = float(np.mean([r["full_clip_completed"] for r in before]))
        pa = float(np.mean([r["full_clip_completed"] for r in after]))
        paired.append(
            dict(
                name=row["name"],
                pool=row["pool"],
                base_mean_completion=b,
                trained_mean_completion=a,
                completion_change=a - b,
                base_full_completion_rate=pb,
                trained_full_completion_rate=pa,
                full_completion_rate_change=pa - pb,
            )
        )
    pools = {}
    for pool in pool_names:
        selected = [x for x in paired if x["pool"] == pool]
        pools[pool] = dict(
            clips=len(selected),
            rollouts_per_condition=len(selected) * replicas,
            base_mean_completion=float(np.mean([r["base_mean_completion"] for r in selected])),
            trained_mean_completion=float(
                np.mean([r["trained_mean_completion"] for r in selected])
            ),
            base_full_completion_rate=float(
                np.mean([r["base_full_completion_rate"] for r in selected])
            ),
            trained_full_completion_rate=float(
                np.mean([r["trained_full_completion_rate"] for r in selected])
            ),
            clips_completion_improved=sum(r["completion_change"] > 1e-6 for r in selected),
            clips_completion_worsened=sum(r["completion_change"] < -1e-6 for r in selected),
        )
    report = dict(
        protocol_file=str(out / "protocol.json"),
        metric="Completion, not reward",
        pools=pools,
        paired_clips=paired,
        public_parameters_unchanged=True,
        matched_initial_states_exact=True,
        existing_weights_only=True,
        training_updates=0,
        elapsed_seconds=time.time() - start,
        limitations=[
            "Single trained martial expert; evaluation replicas are not training seeds.",
            "Complete reference tracking survival is distinct from pose accuracy.",
            "Some existing prepared held-out references are segments of longer source motions; no new truncation.",
            "GPU physical rollouts have observed repeat variability; report as descriptive re-evaluation.",
        ],
    )
    save(out / "report.json", report)
    with (out / "per_clip.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader()
        w.writerows(paired)
    lines = [
        "G1 基座 vs 基座＋已训练武术专家：完成度复核",
        f"新增训练更新：0；每条{replicas}个同初态物理副本；仅使用本地已映射G1动作。",
        "",
    ]
    for pool, item in pools.items():
        lines.append(
            f"{pool}: 平均完成度 {item['base_mean_completion']:.2%} → {item['trained_mean_completion']:.2%}；整段完成率 {item['base_full_completion_rate']:.2%} → {item['trained_full_completion_rate']:.2%}"
        )
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    save(
        out / "status.json",
        dict(
            stage="complete",
            pid=os.getpid(),
            elapsed_seconds=time.time() - start,
            report=str(out / "report.json"),
        ),
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
