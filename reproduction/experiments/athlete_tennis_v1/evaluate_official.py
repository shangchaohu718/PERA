"""Audit the unmodified ATHLETE deployment policy with explicit ball outcomes.

The official model is called a Student upstream; here it is a candidate teacher
for HoloMotion. This script performs no optimization and no robot-state resets
between consecutive incoming balls. Measurement hooks leave physics unchanged.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from experiments.athlete_tennis_v1.compat import setup

COMPATIBILITY = setup()
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from athlete.scripts.play_student_onnx import (
    StudentOnnxPlayConfig,
    _StandaloneStudentSimulation,
)
from athlete.scripts.tennis_physics import STANDARD_TENNIS_PHYSICS as PHYSICS

from pera.paths import ROOT

POLICY = ROOT / "third_party/ATHLETE/deploy/policies/gpu3_2_model45000/model_45000.onnx"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cfg(seed):
    return StudentOnnxPlayConfig(
        onnx_policy_file=POLICY,
        motion_config=None,
        court_preset="small",
        landing_target_x=5.0,
        landing_target_y=0.0,
        landing_target_observation_frame="current_root",
        sweet_spot_velocity_mode="relative_joint",
        seed=seed,
        headless=True,
        realtime=False,
    )


def evaluate(seed, seconds, output, render_seconds, profile="official"):
    sim = _StandaloneStudentSimulation(cfg(seed))
    records = []
    current = {}
    previous_ball = None
    minimum_height = float("inf")
    minimum_upright = float("inf")
    first_unstable_time = None
    original_contact = sim._ball_racket_contact
    original_rebound = sim._apply_explicit_ground_rebound

    def begin():
        nonlocal current, previous_ball
        current = {
            "cycle": len(records),
            "racket_contact": False,
            "net_crossing": None,
            "first_bounce_after_hit": None,
            "target_world": sim.landing_target_position_w.tolist(),
        }
        previous_ball = None

    def contact():
        nonlocal previous_ball
        hit = original_contact()
        p = sim.data.qpos[sim.ball_qpos_adr : sim.ball_qpos_adr + 3].copy()
        if hit and not current["racket_contact"]:
            current.update(
                racket_contact=True,
                hit_ball_position=p.tolist(),
                hit_simulation_time=float(sim.data.time),
            )
        if current["racket_contact"] and previous_ball is not None:
            x = sim.court.net_x_m
            if current["net_crossing"] is None and previous_ball[0] < x <= p[0]:
                a = (x - previous_ball[0]) / (p[0] - previous_ball[0])
                cross = previous_ball + a * (p - previous_ball)
                current["net_crossing"] = {
                    "position": cross.tolist(),
                    "clears_net": bool(
                        cross[2] > PHYSICS.court.net_height_m + PHYSICS.ball.radius_m
                        and abs(cross[1]) < sim.court.net_half_width_m
                    ),
                }
        previous_ball = p
        return hit

    def rebound():
        pending_before = sim._ground_impact_pending
        vertical_velocity_before = sim.data.qvel[sim.ball_dof_adr + 2]
        original_rebound()
        bounced = not sim._ground_impact_pending and (
            pending_before
            or (
                vertical_velocity_before < 0
                and sim.data.qvel[sim.ball_dof_adr + 2] > 0
                and sim.data.qpos[sim.ball_qpos_adr + 2] < PHYSICS.ball.radius_m + 0.003
            )
        )
        if bounced and current["racket_contact"]:
            if current["first_bounce_after_hit"] is None:
                p = sim.data.qpos[sim.ball_qpos_adr : sim.ball_qpos_adr + 3].copy()
                inside = (
                    sim.court.net_x_m
                    < p[0]
                    <= sim.court.net_x_m + sim.court.effective_half_length_m
                    and abs(p[1]) <= sim.court.effective_singles_half_width_m
                )
                current["first_bounce_after_hit"] = {
                    "position": p.tolist(),
                    "inside_opponent_singles_half": bool(inside),
                    "target_error_xy_m": float(
                        np.linalg.norm(p[:2] - sim.landing_target_position_w[:2])
                    ),
                }

    sim._ball_racket_contact = contact
    sim._apply_explicit_ground_rebound = rebound
    renderer = writer = None
    if render_seconds:
        renderer = mujoco.Renderer(sim.model, height=400, width=640)
        writer = imageio.get_writer(
            output / "official_teacher_two_views.mp4",
            fps=25,
            codec="libx264",
            macro_block_size=16,
            quality=7,
        )

    def render(t):
        pictures = []
        for follow in (False, True):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (
                (sim.data.xpos[sim.pelvis_body_id] + np.array([0.4, 0.0, 0.0]))
                if follow
                else (2.25, 0.0, 0.5)
            )
            camera.distance = 3.8 if follow else 11.0
            camera.azimuth = 125.0
            camera.elevation = -23.0 if follow else -32.0
            renderer.update_scene(sim.data, camera=camera)
            pictures.append(renderer.render().copy())
        frame = Image.new("RGB", (1280, 464), (15, 18, 23))
        frame.paste(Image.fromarray(np.concatenate(pictures, axis=1)), (0, 32))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (12, 9),
            "ATHLETE OFFICIAL ONNX (candidate teacher) | NO HoloMotion student | CPU inference",
            fill="white",
        )
        draw.text(
            (652, 9),
            f"Robot: {profile} | same simulation / continuous, no robot resets",
            fill="white",
        )
        draw.text(
            (12, 442),
            f"t={t:.2f}s  ball={len(records) + 1}  racket contacts={sum(r['racket_contact'] for r in records) + int(current['racket_contact'])}  | input: simulated ball state, NOT camera pixels",
            fill="white",
        )
        writer.append_data(np.asarray(frame))

    begin()
    started = time.monotonic()
    try:
        for step in range(round(seconds / sim.cfg.control_dt)):
            result = sim.step()
            if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
                raise RuntimeError("Nonfinite physical state")
            t = (step + 1) * sim.cfg.control_dt
            z = float(sim.data.xpos[sim.pelvis_body_id, 2])
            upright = float(sim.data.xmat[sim.pelvis_body_id].reshape(3, 3)[2, 2])
            minimum_height = min(z, minimum_height)
            minimum_upright = min(upright, minimum_upright)
            if first_unstable_time is None and (z < 0.35 or upright < 0.3):
                first_unstable_time = t
            if writer and step % 2 == 0 and t <= render_seconds:
                render(t)
            if result.episode_complete:
                current["complete_cycle"] = True
                records.append(current)
                sim.start_next_strike_cycle()
                begin()
        current["complete_cycle"] = False
        current["elapsed_s"] = sim.elapsed_s
        # Keep this censored incoming ball, but exclude it from completed-cycle rates.
        records.append(current)
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    return {
        "seed": seed,
        "simulation_seconds": seconds,
        "wall_seconds": time.monotonic() - started,
        "onnx_providers": list(sim.policy.providers),
        "robot_resets_between_balls": 0,
        "min_pelvis_height_m": minimum_height,
        "min_pelvis_up_z": minimum_upright,
        "first_unstable_time_s": first_unstable_time,
        "cycles": records,
    }


def summarize(runs):
    completed = [c for r in runs for c in r["cycles"] if c["complete_cycle"]]
    hit = [c for c in completed if c["racket_contact"]]
    cross = [c for c in hit if c["net_crossing"] and c["net_crossing"]["clears_net"]]
    landed = [c for c in hit if c["first_bounce_after_hit"]]
    success = [
        c
        for c in cross
        if c["first_bounce_after_hit"]
        and c["first_bounce_after_hit"]["inside_opponent_singles_half"]
    ]
    return {
        "completed_incoming_cycles": len(completed),
        "racket_contact_cycles": len(hit),
        "outgoing_net_clearance_cycles": len(cross),
        "observed_post_hit_first_bounces": len(landed),
        "hit_net_clearance_and_opponent_in_bounds_first_bounce": len(success),
        "racket_contact_rate": len(hit) / len(completed),
        "complete_return_rate": len(success) / len(completed),
        "post_hit_bounce_unobserved_cycles": len(hit) - len(landed),
        "mean_landing_target_error_for_observed_bounces_m": float(
            np.mean([c["first_bounce_after_hit"]["target_error_xy_m"] for c in landed])
        )
        if landed
        else None,
        "physically_stable_full_runs": sum(r["first_unstable_time_s"] is None for r in runs),
        "runs": len(runs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--video-seconds", type=float, default=20.0)
    p.add_argument(
        "--profile", choices=("official", "canonical_pd", "canonical_unitree"), default="official"
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from experiments.athlete_tennis_v1.target_g1 import action_contract, install

    contract = action_contract(
        __import__(
            "athlete.scripts.play_student_onnx", fromlist=["_StudentOnnxPolicy"]
        )._StudentOnnxPolicy(POLICY)
    )
    (args.output / "action_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    robot_info = {"profile": "official"}
    if args.profile != "official":
        global _StandaloneStudentSimulation
        _StandaloneStudentSimulation, robot_info = install(args.profile, args.output)
    before = digest(POLICY)
    state = {
        "status": "running",
        "policy": str(POLICY),
        "policy_sha256": before,
        "source_commit": "7330b472a67ec7698f31fe1e357af474311e5213",
        "policy_role": "official ATHLETE deployed Student used as candidate teacher for our project",
        "student_updates": 0,
        "compatibility": COMPATIBILITY,
        "physics_modified": args.profile != "official",
        "robot_profile": robot_info,
        "ball_state_source": "simulator; no camera perception",
        "runs": [],
        "measurement": {
            "ball_events_dt_s": 0.0025,
            "net_crossing": "linear interpolation of positive-X crossing after actual racket contact; ball radius clearance",
            "bounce": "official explicit ground rebound event; first bounce after actual racket contact",
            "stability_proxy": "pelvis z >= .35m and pelvis up-z >= .3 at all control steps",
            "censoring": "exclude final incomplete incoming cycle; missing post-hit bounce recorded as unobserved, not successful",
        },
    }
    report = args.output / "report.json"
    try:
        for i, seed in enumerate(args.seeds):
            state["runs"].append(
                evaluate(
                    seed,
                    args.seconds,
                    args.output,
                    args.video_seconds if i == 0 else 0,
                    args.profile,
                )
            )
            state["summary"] = summarize(state["runs"])
            report.write_text(json.dumps(state, indent=2) + "\n")
            print("[AUDIT]", json.dumps(state["summary"]), flush=True)
        state.update(status="completed", policy_unchanged=digest(POLICY) == before)
        assert state["policy_unchanged"]
    except BaseException as e:
        state.update(status="failed", error=repr(e))
        raise
    finally:
        report.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
