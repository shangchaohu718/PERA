"""Official ATHLETE environment -> frozen HoloMotion + one tennis FFN/router.

Only local adapter/training code is new. The upstream robot, physics, ball
planner, ONNX policy and observable 631-vector are unchanged. No camera model,
TCRS, PPO, teacher future reference, action postprocessor or joint clipping.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from experiments.athlete_tennis_v1.compat import setup

COMPAT = setup()
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from tensordict import TensorDict

from athlete.scripts.play_student_onnx import _StandaloneStudentSimulation
from athlete.scripts.tennis_physics import STANDARD_TENNIS_PHYSICS as PHYSICS
from experiments.athlete_tennis_v1.evaluate_official import cfg, POLICY, summarize
from modumotion_g1.domain_moe import (
    add_domain,
    domain_parameters,
    public_digest,
    save_domain,
    load_domain,
    set_active_domains,
    set_domain_context,
    set_domain_enabled,
    set_trainable_domain,
)
from modumotion_g1.feature_dagger import FrozenFeatureDagger
from modumotion_g1.official import load_official
from modumotion_g1.perceptive_observations import (
    PerceptiveActorObservationInput,
    build_perceptive_actor_terms,
)

from pera.paths import ROOT

DOMAIN = "athlete_tennis"
CONTRACT = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Bridge:
    def __init__(self, names, device):
        self.device = device
        self.gather = np.array([names.index(n) for n in CONTRACT["joint_names"]])
        self.inverse = np.argsort(self.gather)
        self.offset = np.array(CONTRACT["action_offset"])
        self.scale = np.array(CONTRACT["action_scale"])
        self.default = torch.tensor(
            CONTRACT["default_joint_pos"], device=device, dtype=torch.float32
        )
        self.reference = torch.zeros(1, 13, 36, device=device)
        self.reference[:, :, 2] = 0.76
        self.reference[:, :, 3] = 1.0
        self.reference[:, :, 7:] = self.default

    def label(self, sim, native):
        qtarget = sim.default_joint_pos + sim.action_scale * native
        return (qtarget[self.gather] - self.offset) / self.scale

    def native(self, sim, holo):
        target = (self.offset + self.scale * holo)[self.inverse]
        return (target - sim.default_joint_pos) / sim.action_scale

    def terms(self, sim, last_action):
        def t(x):
            return torch.as_tensor(np.array(x, copy=True), device=self.device, dtype=torch.float32)[
                None
            ]

        terms = build_perceptive_actor_terms(
            PerceptiveActorObservationInput(
                reference_qpos=self.reference,
                robot_root_quat_wxyz=t(sim.data.qpos[3:7]),
                robot_root_angvel_local=t(sim.data.sensor("imu_ang_vel").data),
                robot_dof_pos=t(sim.data.qpos[sim.joint_qpos_adr[self.gather]]),
                robot_dof_vel=t(sim.data.qvel[sim.joint_dof_adr[self.gather]]),
                last_action=t(last_action),
                default_dof_pos=self.default,
            )
        )
        return TensorDict({f"unified/{k}": v for k, v in terms.items()}, batch_size=[1])


class BallAudit:
    def __init__(self, sim):
        self.sim = sim
        self.records = []
        self.first_unstable = None
        self.minimum_height = float("inf")
        self.minimum_upright = float("inf")
        original_contact = sim._ball_racket_contact
        original_rebound = sim._apply_explicit_ground_rebound
        self.begin()

        def contact():
            hit = original_contact()
            p = sim.data.qpos[sim.ball_qpos_adr : sim.ball_qpos_adr + 3].copy()
            if hit:
                self.current["racket_contact"] = True
            if self.current["racket_contact"] and self.previous is not None:
                x = sim.court.net_x_m
                if self.current["net_crossing"] is None and self.previous[0] < x <= p[0]:
                    a = (x - self.previous[0]) / (p[0] - self.previous[0])
                    cross = self.previous + a * (p - self.previous)
                    self.current["net_crossing"] = {
                        "position": cross.tolist(),
                        "clears_net": bool(
                            cross[2] > PHYSICS.court.net_height_m + PHYSICS.ball.radius_m
                            and abs(cross[1]) < sim.court.net_half_width_m
                        ),
                    }
            self.previous = p
            return hit

        def rebound():
            pending = sim._ground_impact_pending
            vz = sim.data.qvel[sim.ball_dof_adr + 2]
            original_rebound()
            bounced = not sim._ground_impact_pending and (
                pending
                or (
                    vz < 0
                    and sim.data.qvel[sim.ball_dof_adr + 2] > 0
                    and sim.data.qpos[sim.ball_qpos_adr + 2] < PHYSICS.ball.radius_m + 0.003
                )
            )
            if (
                bounced
                and self.current["racket_contact"]
                and self.current["first_bounce_after_hit"] is None
            ):
                p = sim.data.qpos[sim.ball_qpos_adr : sim.ball_qpos_adr + 3].copy()
                self.current["first_bounce_after_hit"] = {
                    "position": p.tolist(),
                    "inside_opponent_singles_half": bool(
                        sim.court.net_x_m
                        < p[0]
                        <= sim.court.net_x_m + sim.court.effective_half_length_m
                        and abs(p[1]) <= sim.court.effective_singles_half_width_m
                    ),
                    "target_error_xy_m": float(
                        np.linalg.norm(p[:2] - sim.landing_target_position_w[:2])
                    ),
                }

        sim._ball_racket_contact = contact
        sim._apply_explicit_ground_rebound = rebound

    def begin(self):
        self.current = {
            "cycle": len(self.records),
            "racket_contact": False,
            "net_crossing": None,
            "first_bounce_after_hit": None,
        }
        self.previous = None

    def check(self, t):
        sim = self.sim
        z = float(sim.data.xpos[sim.pelvis_body_id, 2])
        up = float(sim.data.xmat[sim.pelvis_body_id].reshape(3, 3)[2, 2])
        self.minimum_height = min(self.minimum_height, z)
        self.minimum_upright = min(self.minimum_upright, up)
        if self.first_unstable is None and (z < 0.35 or up < 0.3):
            self.first_unstable = t
        if not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            raise RuntimeError("Nonfinite MuJoCo state")

    def end_cycle(self, completed):
        self.current["complete_cycle"] = completed
        self.records.append(self.current)


class Video:
    def __init__(self, sim, path):
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model, height=400, width=640)
        self.writer = imageio.get_writer(
            path, fps=25, codec="libx264", quality=7, macro_block_size=16
        )

    def frame(self, t, weight, controller):
        sim = self.sim
        pictures = []
        for follow in (False, True):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (
                (sim.data.xpos[sim.pelvis_body_id] + [0.4, 0, 0]) if follow else (2.25, 0, 0.5)
            )
            cam.distance = 3.8 if follow else 11.0
            cam.azimuth = 125.0
            cam.elevation = -23.0 if follow else -32.0
            self.renderer.update_scene(sim.data, camera=cam)
            pictures.append(self.renderer.render().copy())
        frame = Image.new("RGB", (1280, 464), (15, 18, 23))
        frame.paste(Image.fromarray(np.concatenate(pictures, axis=1)), (0, 32))
        draw = ImageDraw.Draw(frame)
        title = (
            "HOLOMOTION + TENNIS FFN/ROUTER STUDENT"
            if controller == "student"
            else "ATHLETE OFFICIAL TEACHER"
        )
        draw.text(
            (12, 9), title + " | original ATHLETE robot / physics / ball-state input", fill="white"
        )
        draw.text(
            (12, 442),
            f"t={t:.2f}s | expert weight={weight:.3f} | teacher actions executed={'0' if controller == 'student' else 'all'} | no camera pixels",
            fill="white",
        )
        self.writer.append_data(np.asarray(frame))

    def close(self):
        self.writer.close()
        self.renderer.close()


class Runner:
    def __init__(self, actor, device, normalization):
        self.actor, self.device = actor, device
        self.mean = torch.tensor(normalization["mean"], device=device)
        self.std = torch.tensor(normalization["std"], device=device)
        self.features = FrozenFeatureDagger(actor, DOMAIN)

    @torch.no_grad()
    def rollout(
        self,
        seed,
        seconds,
        teacher_probability=0.0,
        collect=False,
        video=None,
        stop_on_fall=False,
        public_only=False,
    ):
        sim = _StandaloneStudentSimulation(cfg(seed))
        bridge = Bridge(list(sim.joint_names), self.device)
        actor = self.actor
        set_domain_enabled(actor, DOMAIN, not public_only)
        actor.reset_kv_cache(1, self.device)
        actor.clear_env_cache(None)
        original = sim.policy
        rng = np.random.default_rng(seed + 90000)
        rows, weights, errors = [], [], []
        last_action = np.zeros(29)
        queries = teacher_steps = student_steps = 0
        audit = BallAudit(sim)
        recorder = Video(sim, video) if video else None

        class Policy:
            def __getattr__(self, key):
                return getattr(original, key)

            def __call__(self, observation):
                nonlocal last_action, queries, teacher_steps, student_steps
                context = (
                    torch.as_tensor(observation, dtype=torch.float32, device=self_device)[None]
                    - mean
                ) / std
                set_domain_context(actor, context)
                td = bridge.terms(sim, last_action)
                action = actor(td, mode="inference")["actions"]
                if collect or teacher_probability > 0:
                    native_teacher = original(observation)
                    queries += 1
                    label = torch.tensor(
                        bridge.label(sim, native_teacher), device=self_device, dtype=torch.float32
                    )[None]
                    errors.append(float((((action - label) * scale).square().mean()).cpu()))
                    if collect:
                        rows.append(features.snapshot(label, action, context=context))
                use_teacher = teacher_probability > 0 and rng.random() < teacher_probability
                if use_teacher:
                    executed = native_teacher
                    teacher_steps += 1
                else:
                    executed = bridge.native(sim, action[0].cpu().numpy())
                    student_steps += 1
                last_action = bridge.label(sim, executed)
                if not np.isfinite(executed).all():
                    raise RuntimeError("Nonfinite executed action")
                w = actor.actor_module.layers[0].last_group_weights
                weights.append(float(w[..., -1].mean()) if not public_only else 0.0)
                set_domain_context(actor, None)
                return executed

        self_device, mean, std, features = self.device, self.mean, self.std, self.features
        scale = torch.tensor(CONTRACT["action_scale"], device=self.device)
        sim.policy = Policy()
        started = time.monotonic()
        steps = 0
        sampling_failure = None
        try:
            for step in range(round(seconds / 0.02)):
                result = sim.step()
                steps += 1
                t = steps * 0.02
                audit.check(t)
                if recorder and step % 2 == 0:
                    recorder.frame(
                        t, weights[-1], "student" if teacher_probability == 0 else "teacher"
                    )
                if stop_on_fall and audit.first_unstable is not None:
                    break
                if result.episode_complete:
                    audit.end_cycle(True)
                    audit.begin()
                    try:
                        sim.start_next_strike_cycle()
                    except RuntimeError as error:
                        if str(error) != "No over-net, once-bounced incoming ball sampled near G1.":
                            raise
                        sampling_failure = {
                            "error": str(error),
                            "time_s": t,
                            "pelvis_position": sim.data.xpos[sim.pelvis_body_id].tolist(),
                            "pelvis_up_z": float(
                                sim.data.xmat[sim.pelvis_body_id].reshape(3, 3)[2, 2]
                            ),
                        }
                        break
            audit.end_cycle(False)
        finally:
            if recorder:
                recorder.close()
            set_domain_context(actor, None)
        packed = {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None
        result = {
            "seed": seed,
            "simulation_seconds": steps * 0.02,
            "requested_seconds": seconds,
            "wall_seconds": time.monotonic() - started,
            "teacher_queries": queries,
            "teacher_executed_steps": teacher_steps,
            "student_executed_steps": student_steps,
            "expert_weight_mean": float(np.mean(weights)),
            "joint_target_rmse_rad_on_collected_states": float(np.sqrt(np.mean(errors)))
            if errors
            else None,
            "first_unstable_time_s": audit.first_unstable,
            "min_pelvis_height_m": audit.minimum_height,
            "min_pelvis_up_z": audit.minimum_upright,
            "cycles": audit.records,
            "robot_resets_between_balls": 0,
        }
        result["incoming_sampling_failure"] = sampling_failure
        result["full_requested_duration_survived"] = (
            steps == round(seconds / 0.02)
            and audit.first_unstable is None
            and sampling_failure is None
        )
        return result, packed


def calibrate():
    observations = []
    for seed in (100, 101):
        sim = _StandaloneStudentSimulation(cfg(seed))
        original = sim.policy

        class Record:
            def __init__(self, policy):
                self.policy = policy

            def __getattr__(self, key):
                return getattr(self.policy, key)

            def __call__(self, obs):
                observations.append(obs.copy())
                return self.policy(obs)

        sim.policy = Record(original)
        for _ in range(1000):
            if sim.step().episode_complete:
                sim.start_next_strike_cycle()
    data = np.asarray(observations, dtype=np.float32)
    return {
        "mean": data.mean(0).tolist(),
        "std": data.std(0).clip(min=0.1).tolist(),
        "kind": "fixed_observable_631_standardization",
        "samples": len(data),
        "seeds": [100, 101],
        "uses_teacher_actions_as_input": False,
        "uses_teacher_future_trajectory": False,
    }


def append_bank(bank, new, capacity=40000):
    if bank is None:
        return new
    result = {k: torch.cat((bank[k], new[k])) for k in new}
    if len(result["target"]) > capacity:
        ids = torch.randperm(len(result["target"]))[:capacity]
        result = {k: v[ids] for k, v in result.items()}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--bootstrap-updates", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--updates-per-round", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--resume-run", type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    lock = (out.parent / "distillation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.manual_seed(183)
    if args.device.startswith("cuda"):
        free, _ = torch.cuda.mem_get_info()
        if free < 5 * 2**30:
            raise RuntimeError(
                "Need 5GiB free before allocating this isolated job; other jobs untouched"
            )
        torch.cuda.set_per_process_memory_fraction(0.28)
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    budget = args.bootstrap_updates + args.rounds * args.updates_per_round
    prior = json.loads((args.resume_run / "report.json").read_text()) if args.resume_run else None
    completed_round = 0
    if prior:
        assert prior["teacher_policy_sha256"] == digest(POLICY)
        assert prior["budget_actor_updates"] == budget
        assert (prior["actor_updates"] - args.bootstrap_updates) % args.updates_per_round == 0
        completed_round = (
            prior["actor_updates"] - args.bootstrap_updates
        ) // args.updates_per_round
        assert set(prior["actual_Adam_steps_per_tensor"]) == {prior["actor_updates"]}
    state = {
        "status": "initializing",
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "method": "BC_then_online_DAgger",
        "actor_updates": 0,
        "budget_actor_updates": budget,
        "original_ATHLETE_robot_physics_and_teacher": True,
        "source_commit": "7330b472a67ec7698f31fe1e357af474311e5213",
        "teacher_policy_sha256": digest(POLICY),
        "teacher_policy": str(POLICY),
        "uses_PPO": False,
        "uses_TCRS": False,
        "uses_camera_pixels": False,
        "public_frozen": True,
        "fixed_merge_bias": 0.0,
        "soft_residual_constraint": False,
        "reference": "fixed HoloMotion preparation pose; no teacher future trajectory",
        "torch_peak_allocated_mib": 0.0,
        "torch_reserved_mib": 0.0,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "history": [],
        "evaluations": {},
        "compatibility": COMPAT,
    }
    if prior:
        state.update(
            actor_updates=prior["actor_updates"],
            history=prior["history"],
            evaluations=prior["evaluations"],
            resumed_from=str(args.resume_run.resolve()),
            resumed_at_Adam_updates=prior["actor_updates"],
            prior_failure=prior.get("error"),
            prior_elapsed_s=prior["elapsed_s"],
            recovery="Preserve failed run. Reconstruct prior replay with each saved collection policy and the original seeds; no additional Adam updates.",
        )
    sources = [
        Path(__file__),
        Path(__file__).with_name("compat.py"),
        Path(__file__).with_name("evaluate_official.py"),
        Path(__file__).with_name("target_g1.py"),
        ROOT / "configs/g1_actor_io_contract.json",
        *[
            ROOT / "modumotion_g1" / p
            for p in (
                "official.py",
                "domain_moe.py",
                "feature_dagger.py",
                "perceptive_observations.py",
            )
        ],
    ]
    hashes = {str(path): digest(path) for path in sources}
    for path in sources:
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    state["source_sha256"] = hashes

    def progress(**changes):
        state.update(changes)
        state["elapsed_s"] = time.monotonic() - started
        if args.device.startswith("cuda"):
            state["torch_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
            state["torch_reserved_mib"] = torch.cuda.memory_reserved() / 2**20
            state["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        atomic(out / "report.json", state)
        print(
            "[PROGRESS]",
            json.dumps(
                {
                    k: state.get(k)
                    for k in (
                        "status",
                        "actor_updates",
                        "budget_actor_updates",
                        "round",
                        "loss",
                        "elapsed_s",
                        "torch_allocated_mib",
                        "eta_seconds",
                    )
                }
            ),
            flush=True,
        )

    def check_sources():
        assert all(digest(Path(path)) == h for path, h in hashes.items()), "Bound source changed"
        assert digest(POLICY) == state["teacher_policy_sha256"], "Teacher policy changed"

    actor = runner = None
    try:
        if prior:
            shutil.copy2(
                args.resume_run / "context_normalization.json", out / "context_normalization.json"
            )
            normalization = json.loads((out / "context_normalization.json").read_text())
        else:
            progress(status="calibrating_fixed_context")
            normalization = calibrate()
            atomic(out / "context_normalization.json", normalization)
        context_hash = digest(out / "context_normalization.json")
        progress(status="loading_public_actor")
        actor, critic, _ = load_official(device="cpu")
        del critic
        context = dict(
            context_dim=631,
            context_schema={
                "kind": "ATHLETE_observable_631",
                "normalization_sha256": context_hash,
                "frozen_preprocessing": True,
                "camera_pixels": False,
                "reference_free_teacher": True,
            },
            encoder_sha256=context_hash,
        )
        if prior:
            load_domain(
                actor, args.resume_run / f"round_{completed_round:03d}.domain.pt", **context
            )
        else:
            add_domain(actor, DOMAIN, initial_bias=0.0, **context)
        set_trainable_domain(actor, DOMAIN)
        set_active_domains(actor, [DOMAIN], {DOMAIN: 0.0})
        actor.to(args.device)
        params = domain_parameters(actor)
        initial = {
            n: p.detach().cpu().clone() for n, p in actor.named_parameters() if p.requires_grad
        }
        assert len(initial) == 4
        state.update(
            public_sha256=actor.public_sha256,
            trainable_tensors=list(initial),
            trainable_parameter_count=sum(p.numel() for p in params),
        )
        runner = Runner(actor, args.device, normalization)
        # A real first teacher trajectory also checks exact cached-feature replay.
        progress(status="preflight_real_trajectory")
        first, bank = runner.rollout(
            102, 2.0, teacher_probability=1.0, collect=True, stop_on_fall=True
        )
        state["preflight_rollout"] = first
        scale = torch.tensor(CONTRACT["action_scale"], device=args.device)
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        if prior:
            saved = torch.load(
                args.resume_run / "optimizer_last.pt", map_location=args.device, weights_only=True
            )
            assert saved["actor_updates"] == prior["actor_updates"]
            optimizer.load_state_dict(saved["optimizer"])
        batch = {k: v[:64].to(args.device) for k, v in bank.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = (((runner.features.predict(batch) - batch["target"]) * scale).square()).mean()
        loss.backward()
        gradients = {
            n: float(p.grad.norm()) if p.grad is not None else None
            for n, p in actor.named_parameters()
            if p.requires_grad
        }
        assert all(g is not None and np.isfinite(g) and g > 0 for g in gradients.values()), (
            gradients
        )
        assert all(p.grad is None for p in actor.parameters() if not p.requires_grad)
        state["preflight_gradients"] = gradients
        optimizer.zero_grad(set_to_none=True)
        # Preflight backprop only: Adam step budget remains zero.
        td_sim = _StandaloneStudentSimulation(cfg(103))
        td_bridge = Bridge(list(td_sim.joint_names), args.device)
        td = td_bridge.terms(td_sim, np.zeros(29))
        with torch.no_grad():
            set_domain_enabled(actor, DOMAIN, False)
            actor.reset_kv_cache(1, args.device)
            disabled = actor(td, mode="inference")["actions"]
            block = actor.actor_module.layers[0]
            specialized = block.compute_moe_ffn
            from holomotion.src.modules.network_modules import GroupedMoEBlock

            block.compute_moe_ffn = GroupedMoEBlock.compute_moe_ffn.__get__(block, type(block))
            try:
                actor.reset_kv_cache(1, args.device)
                native = actor(td, mode="inference")["actions"]
            finally:
                block.compute_moe_ffn = specialized
            torch.testing.assert_close(disabled, native, rtol=0, atol=0)
            set_domain_enabled(actor, DOMAIN, True)
        assert public_digest(actor) == actor.public_sha256
        state.update(
            preflight_public_unchanged=True,
            preflight_expert_off_exactly_native=True,
            preflight_cached_replay_passed=True,
        )
        progress(status="preflight_passed")
        if args.preflight_only:
            progress(status="preflight_completed")
            return
        if prior:
            result, _ = runner.rollout(7001, 20.0, collect=False)
            state["reproduced_prior_failure_evaluation"] = result
            progress(status="reconstructing_replay")
            # Restore the exact collection-policy sequence without touching Adam.
            resume_weights = {
                k: v.detach().cpu().clone()
                for k, v in actor.actor_module.layers[0].domain_plugins[DOMAIN].state_dict().items()
            }
            bootstrap = torch.load(
                args.resume_run / "round_000.domain.pt", map_location="cpu", weights_only=True
            )
            actor.actor_module.layers[0].domain_plugins[DOMAIN].load_state_dict(
                bootstrap["layers"][0]
            )
        # Bootstrap from unmodified official teacher rollouts, not canonical-motor data.
        for seed in (104, 105, 106, 107):
            progress(status="collecting_bootstrap", collection_seed=seed)
            result, new = runner.rollout(
                seed, 20.0, teacher_probability=1.0, collect=True, stop_on_fall=True
            )
            bank = append_bank(bank, new)
            if not prior:
                state["history"].append({"phase": "bootstrap_collection", "result": result})
        if prior:
            reconstructed = []
            for old_round in range(1, completed_round + 1):
                payload = torch.load(
                    args.resume_run / f"round_{old_round - 1:03d}.domain.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                actor.actor_module.layers[0].domain_plugins[DOMAIN].load_state_dict(
                    payload["layers"][0]
                )
                beta = (0.5, 0.25, 0.1, 0.0, 0.0, 0.0)[min(old_round - 1, 5)]
                for j in range(4):
                    progress(status="reconstructing_replay", reconstructing_round=old_round)
                    result, new = runner.rollout(
                        1000 + old_round * 10 + j,
                        20.0,
                        teacher_probability=beta,
                        collect=True,
                        stop_on_fall=True,
                    )
                    bank = append_bank(bank, new)
                    reconstructed.append(result)
            actor.actor_module.layers[0].domain_plugins[DOMAIN].load_state_dict(resume_weights)
            state["reconstructed_prior_rollouts"] = reconstructed
            # Confirm changing collection policies did not reset/increment Adam.
            assert set(int(s["step"]) for s in optimizer.state.values()) == {prior["actor_updates"]}
            progress(status="resume_ready", replay_samples=len(bank["target"]))

        total_update_seconds = 0.0

        def fit(count, round_id):
            nonlocal total_update_seconds
            start = time.monotonic()
            first_loss = None
            for i in range(count):
                ids = torch.randint(len(bank["target"]), (args.batch_size,))
                b = {k: v[ids].to(args.device) for k, v in bank.items()}
                prediction = runner.features.predict(b)
                loss = (((prediction - b["target"]) * scale).square()).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite action supervision loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                state["actor_updates"] += 1
                value = float(loss.detach())
                if first_loss is None:
                    first_loss = value
                if (i + 1) % 100 == 0 or i == count - 1:
                    elapsed = total_update_seconds + time.monotonic() - start
                    new_updates = state["actor_updates"] - (prior["actor_updates"] if prior else 0)
                    eta = (budget - state["actor_updates"]) * elapsed / new_updates
                    progress(
                        status="training",
                        round=round_id,
                        loss=value,
                        eta_update_compute_only_seconds=eta,
                        replay_samples=len(bank["target"]),
                    )
            total_update_seconds += time.monotonic() - start
            adam_steps = [int(s["step"]) for s in optimizer.state.values()]
            assert set(adam_steps) == {state["actor_updates"]}
            assert public_digest(actor) == actor.public_sha256
            assert all(p.grad is None for p in actor.parameters() if not p.requires_grad)
            changed = {
                n: bool(not torch.equal(initial[n], p.detach().cpu()))
                for n, p in actor.named_parameters()
                if p.requires_grad
            }
            assert all(changed.values())
            assert all(torch.isfinite(p).all() for p in params)
            state.update(
                actual_Adam_steps_per_tensor=adam_steps,
                new_tensors_changed=changed,
                public_unchanged=True,
            )
            state["history"].append(
                {
                    "phase": "fit",
                    "round": round_id,
                    "updates": count,
                    "first_loss": first_loss,
                    "last_loss": value,
                }
            )
            save_domain(
                actor,
                DOMAIN,
                out / f"round_{round_id:03d}.domain.pt",
                provenance={
                    "teacher_policy_sha256": state["teacher_policy_sha256"],
                    "actual_Adam_updates": state["actor_updates"],
                    "context_normalization": normalization,
                    "method": "BC_then_online_DAgger",
                    "physics": "original_ATHLETE",
                    "reference": "fixed preparation pose",
                },
            )
            torch.save(
                {"optimizer": optimizer.state_dict(), "actor_updates": state["actor_updates"]},
                out / "optimizer_last.pt",
            )
            check_sources()

        if not prior:
            fit(args.bootstrap_updates, 0)
        for round_id in range(completed_round + 1, args.rounds + 1):
            beta = (0.5, 0.25, 0.1, 0.0, 0.0, 0.0)[min(round_id - 1, 5)]
            collected = []
            # Four independent episodes per round; physics is never teleported to teacher states.
            for j in range(4):
                seed = 1000 + round_id * 10 + j
                progress(status="collecting_DAgger", round=round_id, teacher_probability=beta)
                result, new = runner.rollout(
                    seed, 20.0, teacher_probability=beta, collect=True, stop_on_fall=True
                )
                bank = append_bank(bank, new)
                collected.append(result)
            state["history"].append(
                {
                    "phase": "online_DAgger",
                    "round": round_id,
                    "teacher_probability": beta,
                    "rollouts": collected,
                }
            )
            fit(args.updates_per_round, round_id)
            result, _ = runner.rollout(7001, 20.0, collect=False, stop_on_fall=False)
            assert result["teacher_queries"] == result["teacher_executed_steps"] == 0
            state["evaluations"][f"student_round_{round_id}"] = result
            progress(status="round_completed", round=round_id)
        check_sources()
        progress(status="final_evaluation")
        for role in ("teacher", "student"):
            results = []
            for i, seed in enumerate(range(7101, 7109)):
                result, _ = runner.rollout(
                    seed,
                    40.0,
                    teacher_probability=1.0 if role == "teacher" else 0.0,
                    collect=False,
                    video=out / f"{role}_two_views.mp4" if i == 0 else None,
                )
                if role == "student":
                    assert result["teacher_queries"] == result["teacher_executed_steps"] == 0
                results.append(result)
                summary = summarize(results)
                summary["physically_stable_full_runs"] = sum(
                    x["full_requested_duration_survived"] for x in results
                )
                summary["incoming_sampling_failure_runs"] = sum(
                    x["incoming_sampling_failure"] is not None for x in results
                )
                state["evaluations"][role] = {"runs": results, "summary": summary}
                progress(status="final_evaluation", evaluating=role, seed=seed)
        assert public_digest(actor) == actor.public_sha256
        check_sources()
        progress(status="completed", public_unchanged=True, teacher_unchanged=True, eta_seconds=0.0)
    except BaseException as error:
        progress(status="failed", error=repr(error))
        raise
    finally:
        if runner:
            runner.features.close()


if __name__ == "__main__":
    main()
