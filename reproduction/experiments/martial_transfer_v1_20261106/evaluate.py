"""Four matched physical rollouts; no training, calibration or test selection."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.martial_transfer_v1_20261106.prepare import ROOT, CODE, REPORT, sha, save

NATIVE = ROOT / "src"
PLUGIN_CODE = ROOT / "reproduction/experiments/blink_competitive_martial_ppo_v1_20261106"
from experiments.blink_competitive_martial_ppo_v1_20261106.routing import load_base, public_digest
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP, MAX_STEP_REWARD
from blinkmimic.dr import XiLayout
from blinkmimic.robot import get_robot
from blinkmimic.moe_policy import ReferenceMoE
from blinkmimic.rotations import mat_to_quat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", required=True, choices=["base", "initialized", "trained", "trained_disabled"]
    )
    parser.add_argument("--max-steps", type=int, help="Preflight only; never used in final metrics")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.65)
    protocol = json.loads((REPORT / "protocol.json").read_text())
    assert sha(REPORT / "data_contract.json") == protocol["data_contract_sha256"]
    assert protocol["robot"] == "unitree_g1_29dof"
    data = json.loads((REPORT / "data_contract.json").read_text())
    rows = data["rows"]
    for row in rows:
        assert sha(row["file"]) == row["sha256"], row["file"]
    for item in protocol["checkpoints"].values():
        assert sha(item["file"]) == item["sha256"]
    condition = args.condition
    outdir = REPORT / ("preflight_" + condition if args.max_steps else condition)
    outdir.mkdir(exist_ok=True)
    assert not (outdir / "result.json").exists(), "Refuse to replace a completed evaluation"
    start_time = time.time()

    def progress(stage, **kw):
        save(
            outdir / "status.json",
            dict(
                stage=stage,
                pid=os.getpid(),
                condition=condition,
                elapsed_seconds=time.time() - start_time,
                **kw,
            ),
        )

    progress("loading")
    domain_key = "initialized" if condition == "initialized" else "trained"
    domain = None if condition == "base" else protocol["checkpoints"][domain_key]["file"]
    pi, config, _ = load_base(protocol["checkpoints"]["base"]["file"], domain=domain)
    pi.moe.domain_enabled = condition in ("initialized", "trained")
    expected_hash = public_digest(pi)
    if domain:
        ck = torch.load(domain, map_location="cpu", weights_only=True)
        assert ck["public_sha256"] == expected_hash
        if condition != "initialized":
            assert ck["iteration"] == 2000, ck.keys()
    assert not any(
        p.requires_grad for n, p in pi.named_parameters() if not n.startswith("moe.plugin.")
    )
    pi.requires_grad_(False)
    robot = get_robot("g1")
    assert list(robot.joint_names) == data["joint_names"]
    kin = Kinematics(robot, "cuda:0")
    files = [r["file"] for r in rows]
    motion = MotionLib(files, kin, "cuda:0", robot.load_motion)
    assert motion.names == [r["name"] for r in rows]
    # Preserve the existing evaluator's resampler. Its floating-point floor
    # can drop one final frame of an already50Hz source. Audit that explicitly;
    # never mistake a policy failure or an intentional short crop for completion.
    frame_deltas = [int(n) - r["frames"] for n, r in zip(motion.clip_len.tolist(), rows)]
    assert all(d in (0, -1) for d in frame_deltas), ("Unexpected resampling length", frame_deltas)
    mdp = MDP(kin, motion, pi.ref_q0, XiLayout(robot, "cuda:0").dim, robot.sensing)
    cases = [(i, c) for i, row in enumerate(rows) for c in range(row["initial_cases"])]
    clip_id = torch.tensor([i for i, c in cases], device="cuda:0")
    N = len(cases)
    env = TrackingSim(
        make_sim(robot), N, robot, motion, mdp, "cuda:0", eval_mode=True, randomize=False
    )
    ids = torch.arange(N, device="cuda:0")
    env.reset(ids, motion.clip_start[clip_id].clone())
    # Fixed perturbation bank is independently regenerated identically in each condition.
    rngs = {c: np.random.default_rng(41000 + c) for c in (1, 2, 3)}
    S = [x.clone() for x in env.state()]
    for i, (j, c) in enumerate(cases):
        if c == 0:
            continue
        rng = rngs[c]
        S[0][i, :2] += torch.tensor(
            rng.uniform(-0.005, 0.005, 2), device="cuda:0", dtype=torch.float32
        )
        S[2][i] += torch.tensor(rng.uniform(-0.01, 0.01, 3), device="cuda:0", dtype=torch.float32)
        S[4][i] += torch.tensor(rng.uniform(-0.01, 0.01, 29), device="cuda:0", dtype=torch.float32)
        S[5][i] += torch.tensor(rng.uniform(-0.02, 0.02, 29), device="cuda:0", dtype=torch.float32)
    S[4].clamp_(kin.q_lo, kin.q_hi)
    env.robot.write_root_link_pose_to_sim(
        torch.cat((S[0] + env.scene.env_origins, mat_to_quat(S[1])), -1), env_ids=ids
    )
    env.robot.write_root_link_velocity_to_sim(torch.cat((S[2], S[3]), -1), env_ids=ids)
    env.robot.write_joint_position_to_sim(S[4], env_ids=ids)
    env.robot.write_joint_velocity_to_sim(S[5], env_ids=ids)
    env.robot.set_joint_position_target(S[4], env_ids=ids)
    np.savez_compressed(
        outdir / "initial_states.npz",
        **{str(i): x.cpu().numpy() for i, x in enumerate(env.state())},
    )
    old_selection_bias = pi.moe.selection_bias.clone()
    plugin_selection_bias = pi.moe.plugin.selection_bias.clone()
    # Capture the FK already computed by the environment reward, not a second
    # FK with a potentially different frame or metric definition.
    original_fk = kin.forward
    fk_cache = {}

    def cached_fk(*a, **kw):
        value = original_fk(*a, **kw)
        fk_cache["value"] = value
        return value

    kin.forward = cached_fk
    feet = [
        list(robot.tracked_bodies).index(n)
        for n in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    ref_foot_z = motion.body_p[:, feet, 2]
    peaks = torch.stack(
        [
            ref_foot_z[s : e + 1].max(0).values
            for s, e in zip(motion.clip_start.tolist(), motion.clip_end.tolist())
        ]
    )
    h = None
    ret = torch.zeros(N, device="cuda:0")
    steps = torch.zeros(N, device="cuda:0", dtype=torch.long)
    admission = torch.zeros_like(ret)
    contribution = torch.zeros_like(ret)
    routes = torch.zeros(N, 9, device="cuda:0")
    peak_reached = torch.zeros(N, 2, device="cuda:0", dtype=torch.long)
    peak_height_error = torch.zeros(N, 2, device="cuda:0")
    peak_height_hit = torch.zeros(N, 2, device="cuda:0", dtype=torch.bool)
    traces = {key: [] for key in ("body_rms", "root_xy", "joint_rms", "alive")}
    # Predetermined nominal unseen-motion trajectories, sufficient for honest
    # video replay. Record all32 rather than selecting only good cases.
    video_ids = torch.tensor(
        [i for i, (j, c) in enumerate(cases) if j < 32 and c == 0], device="cuda:0"
    )
    video_states, video_alive = [], []
    equality_max = 0.0
    max_steps = args.max_steps or int(motion.clip_len.max())
    progress("evaluating", environments=N, clips=len(rows), maximum_control_steps=max_steps)
    with torch.inference_mode():
        for t in range(max_steps):
            alive = ~env.done_eval.clone()
            if not bool(alive.any()):
                break
            o = env.observe()
            h = o[:, None].repeat(1, 10, 1) if h is None else torch.cat((h[:, 1:], o[:, None]), 1)
            goal = motion.goal(env.f, 10)
            if t in (0, 25, 100):
                saved_class, enabled = pi.moe.__class__, pi.moe.domain_enabled
                pi.moe.domain_enabled = False
                disabled_a = pi(h, goal)
                pi.moe.__class__ = ReferenceMoE
                public_a = pi(h, goal)
                pi.moe.__class__ = saved_class
                pi.moe.domain_enabled = enabled
                error = float((disabled_a - public_a).abs().max())
                equality_max = max(equality_max, error)
                assert error == 0.0, ("Disabled plugin does not recover public policy", error)
            act = pi(h, goal)
            assert bool(torch.isfinite(act[alive]).all()), "Nonfinite active policy action"
            selected = pi.moe.last_selected[:, 0]
            weight = pi.moe.last_weight[:, 0]
            selected_ids = pi.moe.last_old_indices.clone()
            selected_ids[:, 1] = torch.where(selected, 8, selected_ids[:, 1])
            routes += torch.nn.functional.one_hot(selected_ids, 9).sum(1) * alive[:, None]
            admission += selected & alive
            contribution += torch.where(alive, weight, 0)
            act = torch.where(alive[:, None], act, torch.zeros_like(act))
            out = env.step(act)
            next_f = out["f"] + 1
            state = env.state()
            body_error = out["terms"]["e_body_pos"].sqrt()
            root_error = (state[0][:, :2] - motion.root_p[next_f, :2]).norm(dim=-1)
            joint_error = (state[4] - motion.q[next_f]).square().mean(-1).sqrt()
            actual_foot_z = fk_cache["value"][0][:, feet, 2]
            reference_z = ref_foot_z[next_f]
            in_peak = (reference_z >= 0.9 * peaks[clip_id]) & (peaks[clip_id] >= 0.25)
            relevant = in_peak & alive[:, None]
            height_error = (actual_foot_z - reference_z).abs()
            peak_reached += relevant
            peak_height_error += torch.where(relevant, height_error, 0)
            peak_height_hit |= relevant & (height_error <= 0.15)
            for name, value in [
                ("body_rms", body_error),
                ("root_xy", root_error),
                ("joint_rms", joint_error),
            ]:
                traces[name].append(torch.where(alive, value, torch.zeros_like(value)))
            traces["alive"].append(alive)
            pose = torch.cat((state[0], mat_to_quat(state[1]), state[4]), -1)
            video_states.append(pose[video_ids].cpu().numpy())
            video_alive.append(alive[video_ids].cpu().numpy())
            ret += torch.where(alive, out["r"], 0)
            steps += alive
            env.f = torch.minimum(env.f, motion.clip_end[clip_id] - 1)
            if (t + 1) % 500 == 0:
                progress(
                    "evaluating",
                    environments=N,
                    clips=len(rows),
                    control_step=t + 1,
                    active_environments=int((~env.done_eval).sum()),
                    maximum_control_steps=max_steps,
                )
    arrays = {key: torch.stack(value).cpu().numpy() for key, value in traces.items()}
    assert all(np.isfinite(value).all() for key, value in arrays.items() if key != "alive")
    np.savez_compressed(outdir / "frame_metrics.npz", **arrays)
    np.savez_compressed(
        outdir / "trajectories.npz",
        qpos=np.stack(video_states),
        alive=np.stack(video_alive),
        names=np.array([rows[cases[int(i)][0]]["name"] for i in video_ids.cpu()]),
    )
    record = []
    for i, (j, c) in enumerate(cases):
        row = rows[j]
        count = int(steps[i])
        record.append(
            dict(
                clip=row["name"],
                pool=row["pool"],
                split_group=row["split_group"],
                case=c,
                source_clip_frames=row["frames"],
                full_clip_frames=int(motion.clip_len[j]),
                observed_steps=count,
                full_clip_survival=bool(env.done_eval[i] and not env.failed_eval[i]),
                failure=bool(env.failed_eval[i]),
                lifetime_seconds=count * 0.02,
                normalized_return=float(ret[i] / ((motion.clip_len[j] - 1) * MAX_STEP_REWARD)),
                body_anchor_aligned_rms_m=float(arrays["body_rms"][:, i].sum() / max(count, 1)),
                root_xy_error_m=float(arrays["root_xy"][:, i].sum() / max(count, 1)),
                joint_rms_rad=float(arrays["joint_rms"][:, i].sum() / max(count, 1)),
                new_admission=float(admission[i] / max(count, 1)),
                new_contribution=float(contribution[i] / max(count, 1)),
                high_reference_foot_height_error_m=[
                    float(peak_height_error[i, k] / peak_reached[i, k])
                    if peak_reached[i, k] > 0
                    else None
                    for k in range(2)
                ],
                high_reference_foot_height_hit=peak_height_hit[i].cpu().tolist(),
                peak_reference_height_m=peaks[j].cpu().tolist(),
            )
        )
    result = dict(
        condition=condition,
        protocol_sha256=sha(REPORT / "protocol.json"),
        public_sha256_before=expected_hash,
        public_sha256_after=public_digest(pi),
        public_unchanged=expected_hash == public_digest(pi),
        old_selection_bias_unchanged=bool(torch.equal(old_selection_bias, pi.moe.selection_bias)),
        new_selection_bias_unchanged=bool(
            torch.equal(plugin_selection_bias, pi.moe.plugin.selection_bias)
        ),
        disabled_vs_native_action_max_abs_error=equality_max,
        full_evaluation=not bool(args.max_steps),
        completed_bookkeeping=bool(env.done_eval.all()),
        environments=N,
        records=record,
        elapsed_seconds=time.time() - start_time,
        native_resampling_final_frame_deltas={
            r["name"]: d for r, d in zip(rows, frame_deltas) if d
        },
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    assert (
        result["public_unchanged"]
        and result["old_selection_bias_unchanged"]
        and result["new_selection_bias_unchanged"]
    )
    if condition in ("base", "trained_disabled"):
        assert all(r["new_admission"] == r["new_contribution"] == 0 for r in record)
    if not args.max_steps:
        assert result["completed_bookkeeping"]
    save(outdir / "result.json", result)
    progress(
        "complete",
        output=str(outdir / "result.json"),
        torch_peak_allocated_mib=result["torch_peak_allocated_mib"],
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
