"""Reuse existing ModuMotion-G1 native teachers, bridges and physical tests.

Each factory is imported only inside its own original interpreter. No teacher
weights are installed in the current MLP student. No dependency/source edits.
"""

import copy, json, sys
from types import SimpleNamespace
import numpy as np
import torch
from .model import ROOT, Features, CAPTURE


def packed(rows):
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None


def native(task, actor, cfg, out):
    family = task["family"]
    name = task["id"]
    N = task["num_envs"]
    seconds = cfg["collection_seconds"]
    feat = Features(actor)
    if family == "instinct":
        from scripts.instinct_dagger import prepare_dataset, TEACHER
        from modumotion_g1.instinct_dagger import NativeTeacherTask

        env = NativeTeacherTask(prepare_dataset(name), TEACHER, N, "cuda:0")

        def run(role, seed, collect=False, beta=0.0):
            result, data, _ = env.rollout(
                actor=actor,
                features=feat if collect else None,
                seed=seed,
                controller="teacher" if role == "teacher" else "mixed" if collect else "student",
                teacher_probability=beta,
            )
            result["success_fraction"] = result["success_rate"]
            return result, data

        return SimpleNamespace(
            run=run,
            close=env.close,
            modules=[env.teacher_actor, env.teacher_norm, env.depth],
            scope="Original InstinctMJ reference/mesh, native endings and depth32",
        )
    if family == "velocity":
        from modumotion_g1.velocity_walking import VelocityWalkTask
        from scripts.velocity_walk_dagger import collect as gather, evaluate

        env = VelocityWalkTask(N, "cuda:0")

        def run(role, seed, collect=False, beta=0.0):
            if collect:
                data, r = gather(
                    env,
                    actor,
                    feat,
                    seed=seed,
                    beta=1.0 if role == "teacher" else beta,
                    seconds=seconds,
                )
                r["success_fraction"] = 0.0
                return r, data
            r = evaluate(env, None if role == "teacher" else actor, seed=seed)
            r["success_fraction"] = float(r["acceptance_passed"])
            return r, None

        return SimpleNamespace(
            run=run,
            close=env.env.close,
            modules=[env.teacher],
            scope="Original velocity-command segments and stopping tests; navigation handover not revalidated",
        )
    if family == "husky":
        from modumotion_g1.husky_skateboarding import HuskySkateTask
        from scripts.husky_skateboard_dagger import collect as gather, evaluate

        env = HuskySkateTask(N, "cuda:0")

        def run(role, seed, collect=False, beta=0.0):
            if collect:
                data, r = gather(
                    env,
                    actor,
                    feat,
                    seed=seed,
                    beta=1.0 if role == "teacher" else beta,
                    seconds=seconds,
                )
                r["success_fraction"] = 0.0
                return r, data
            r = evaluate(
                env,
                None if role == "teacher" else actor,
                seconds=30,
                seed=seed,
                turn=bool(seed % 2),
            )
            r["success_fraction"] = r["successful_episodes"] / r["episodes"]
            return r, None

        return SimpleNamespace(
            run=run,
            close=lambda: None,
            modules=[env.teacher],
            scope="Original HUSKY board physics, on-board start, contact/displacement test",
        )
    if family == "parkour":
        return parkour(task, actor, cfg, out)
    if family == "mimicx":
        return mimicx(task, actor, cfg, out)
    if family == "ultra":
        return ultra(task, actor, cfg, out)
    if family == "tennis":
        return tennis(task, actor, cfg, out)
    if family == "pmt":
        return pmt(task, actor, cfg, out)
    if family == "kick":
        return kick(task, actor, cfg, out)
    raise ValueError(f"Unknown native adapter {family}")


def parkour(task, actor, cfg, out):
    from modumotion_g1.parkour_remaining import RemainingParkourTask
    from modumotion_g1.parkour_distillation import ParkourTask
    from modumotion_g1.parkour_extra import ExtraParkourTask

    key = task["id"].removeprefix("parkour_")
    separate_eval = key.endswith("_high") or key in [
        "stairs_up",
        "gaps",
        "obstacles",
    ]
    if key.endswith("_high"):
        from modumotion_g1.parkour_extra import rollout

        make = lambda n: ExtraParkourTask(key, n)
    elif key in ["stairs_up", "gaps", "obstacles"]:
        from modumotion_g1.parkour_remaining_rollout import rollout

        make = lambda n: RemainingParkourTask(key, n)
    else:
        from scripts.parkour_dagger import rollout

        make = lambda n: ParkourTask(key, n, "cuda:0")
    training = make(task["num_envs"])
    evaluation = make(1) if separate_eval else training
    feat = Features(actor)

    def run(role, seed, collect=False, beta=0.0):
        env = training if collect else evaluation
        result, data = rollout(
            env,
            actor if collect or role != "teacher" else None,
            seed=seed,
            seconds=cfg["collection_seconds"] if collect else 20,
            features=feat if collect else None,
            beta=1.0 if role == "teacher" else beta,
        )
        result["success_fraction"] = result.get("successes", 0) / max(
            result.get("num_episodes", 1), 1
        )
        return result, data

    def close():
        training.env.close()
        if evaluation is not training:
            evaluation.env.close()

    modules = [training.teacher, training.normalizer, training.depth]
    if evaluation is not training:
        modules.extend([evaluation.teacher, evaluation.normalizer, evaluation.depth])
    return SimpleNamespace(
        run=run,
        close=close,
        modules=modules,
        scope="Unchanged prior native parkour rollout/collect and task-specific physical success checks; separate1-world evaluation where originally required",
    )


def mimicx(task, actor, cfg, out):
    from experiments.mimicx_all_tasks_v1.runtime import setup
    from modumotion_g1.mimicx_bridge import MimicXBridge
    from omegaconf import OmegaConf

    env, teacher, row, paths = setup(
        task["id"].removeprefix("mimicx_"), cfg["seed"], task["num_envs"]
    )
    oldcfg = OmegaConf.load(ROOT / "configs/holomotion_v1.4.yaml")
    b = MimicXBridge(env.unwrapped, oldcfg, paths["motion"])
    N = env.num_envs
    norm = (
        copy.deepcopy(teacher.obs_normalizer).eval().requires_grad_(False)
        if task["context_dim"]
        else None
    )

    def run(role, seed, collect=False, beta=0.0):
        obs, _ = env.unwrapped.reset(seed=seed)
        actor.reset_kv_cache()
        b.last_action.zero_()
        alive = torch.ones(N, dtype=torch.bool, device="cuda:0")
        rows = []
        errors = []
        for step in range(int(row["horizon"])):
            ctx = (
                norm(torch.cat([obs[g] for g in teacher.obs_groups], -1))
                if norm is not None
                else None
            )
            actor.context = ctx
            td = b.observations()
            CAPTURE["positions"] = (
                b.command.time_steps[:, None] + torch.arange(-1, 12, device="cuda:0")
            ).clamp(0, len(b.qpos) - 1)
            mu = actor(td)["actions"]
            label = teacher(obs) if collect or role == "teacher" else None
            if label is not None and env.clip_actions is not None:
                label = label.clamp(-env.clip_actions, env.clip_actions)
            if collect:
                # Bound host replay while sampling the actual current states.
                f = {
                    **actor.cached,
                    "target": actor.old_to_sdk(b.teacher_to_student(label)),
                    "oldids": actor.pi.moe.last_old_indices,
                }
                ids = torch.randperm(N, device="cuda:0")[: min(N, 32)]
                rows.append({k: v[ids].detach().cpu().clone() for k, v in f.items()})
            native = label if role == "teacher" else b.student_to_teacher(mu)
            if collect and role != "teacher":
                native = torch.where(torch.rand(N, 1, device="cuda:0") < beta, label, native)
            if env.clip_actions is not None:
                native = native.clamp(-env.clip_actions, env.clip_actions)
            obs, _, done, _ = env.step(native)
            alive &= ~done.bool()
            b.last_action.copy_(b.teacher_to_student(native))
            ids = done.nonzero().flatten()
            if len(ids):
                actor.clear_env_cache(ids)
                b.last_action[ids] = 0
            cmd = b.command
            error = (cmd.robot_body_pos_w - cmd.body_pos_w).norm(dim=-1).mean(-1)
            if alive.any():
                errors.append(float(error[alive].mean()))
        enderr = (b.command.robot_anchor_pos_w - b.command.anchor_pos_w).norm(dim=-1)
        arrived = alive & (enderr <= 0.25)
        return dict(
            seed=seed,
            environments=N,
            reference_survivors=int(alive.sum()),
            reference_endpoint_near=int(arrived.sum()),
            success_fraction=float(arrived.float().mean()),
            mean_body_error_m=float(np.mean(errors)) if errors else None,
            full_terrain_completion_validated=False,
            teacher_queries=0 if not collect and role != "teacher" else N * int(row["horizon"]),
        ), packed(rows)

    return SimpleNamespace(
        run=run,
        close=env.close,
        modules=[teacher] + ([norm] if norm else []),
        scope="Original MimicX fixed reference/scene; survival and reference endpoint, not whole-terrain completion; football/tennis are motion tracking only",
    )


def ultra(task, actor, cfg, out):
    sys.path.insert(0, str(ROOT / "reproduction/experiments/ultra_box_v1"))
    from experiments.ultra_box_v1.vector import VectorULTRA, HoloBridge

    motion = ROOT / "data/ultra_official_v1/sub10_largebox_046_072_103_077_100_100_100.pt"
    v = VectorULTRA([motion], task["num_envs"], workers=min(12, task["num_envs"]))
    b = HoloBridge(v)
    feat = Features(actor)

    def run(role, seed, collect=False, beta=0.0):
        v.rng = np.random.default_rng(seed)
        v.reset_all(False)
        b.last.zero_()
        actor.reset_kv_cache()
        rows = []
        count = max(w.original_frames for w in v.worlds)
        for step in range(count):
            obs, _ = v.observations()
            obs = obs.to("cuda:0")
            actor.context = v.teacher.normalize(obs)
            mu = actor(b.terms())["actions"]
            teacher = v.teacher(obs) if collect or role == "teacher" else None
            if collect:
                rows.append(feat.snapshot(b.labels(teacher), mu, actor.context))
            a = teacher if role == "teacher" else b.native(mu)
            if collect and role != "teacher":
                a = torch.where(torch.rand(v.N, 1, device="cuda:0") < beta, teacher, a)
            reset = v.step(a.cpu().numpy(), auto_reset=collect)
            b.last.copy_(b.labels(a))
            if reset:
                actor.clear_env_cache(torch.tensor(reset, device="cuda:0"))
                b.last[reset] = 0
            if not collect and v.done.all():
                break
        records = [w.summary() for w in v.worlds]
        good = sum(x["physical_carry_demonstrated"] and x["reference_completed"] for x in records)
        return dict(
            seed=seed,
            environments=v.N,
            successes=good,
            success_fraction=good / v.N,
            records=records,
            initial_distribution="Original identical phase-zero native046 reset; not independent statistical trials",
        ), packed(rows)

    return SimpleNamespace(
        run=run,
        close=v.close,
        modules=[v.teacher],
        scope="Existing native dynamic-box physics,4052D frozen normalized input; contact/lift/carry required",
    )


def tennis(task, actor, cfg, out):
    sys.path.insert(0, str(ROOT / "reproduction/experiments/athlete_tennis_v1"))
    from experiments.athlete_tennis_v1 import distill as tennis_native

    # Existing Runner physics, labels, ball audit and normalization; swap only student cache.
    tennis_native.FrozenFeatureDagger = Features
    normalization = json.loads(
        (
            ROOT
            / "reports/athlete_tennis_v1/body_window_continue_10k_v1/context_normalization.json"
        ).read_text()
    )
    runner = tennis_native.Runner(actor, "cuda:0", normalization)
    policy_hash = __import__("hashlib").sha256(tennis_native.POLICY.read_bytes()).hexdigest()

    def run(role, seed, collect=False, beta=0.0):
        r, data = runner.rollout(
            seed,
            40 if not collect else cfg["collection_seconds"],
            teacher_probability=1.0 if role == "teacher" else beta if collect else 0.0,
            collect=collect,
            stop_on_fall=not collect,
        )
        if (
            __import__("hashlib").sha256(tennis_native.POLICY.read_bytes()).hexdigest()
            != policy_hash
        ):
            raise RuntimeError("ATHLETE teacher ONNX changed")
        n = sum(x["complete_cycle"] for x in r["cycles"])
        if n:
            score = tennis_native.summarize([r])
            r["ball_summary"] = score
            r["success_fraction"] = (
                score["complete_return_rate"] if r["full_requested_duration_survived"] else 0.0
            )
        else:
            r["success_fraction"] = 0.0
        return r, data

    return SimpleNamespace(
        run=run,
        close=lambda: None,
        modules=[],
        scope="Original ATHLETE631 observable, ONNX teacher/native ball physics; actual legal returns, no mounting/navigation claim",
    )


def pmt(task, actor, cfg, out):
    from pmt_tasks import builder
    from experiments.pmt_dagger_v5.target_robot import configure_target_robot
    from experiments.pmt_dagger_v5.task import OptimizedPlacementTask
    from experiments.pmt_dagger_v2.rollout import rollout

    original = builder.build_env_cfg

    def configure(*a, **kw):
        return configure_target_robot(original(*a, **kw))[0]

    builder.build_env_cfg = configure
    try:
        env = OptimizedPlacementTask(
            task["num_envs"],
            clip_id=0,
            actor_reference="optimized",
            full_reference=True,
        )
    finally:
        builder.build_env_cfg = original
    path = ROOT / "reports/pmt_teacher_adaptation_v1/training/best.pt"
    teacher = torch.load(path, map_location=env.device, weights_only=False)
    if (
        teacher["robot"] != "canonical_HoloMotion_G1_nominal_Unitree"
        or teacher["dataset_manifest_sha256"] != env.assets_manifest_sha256
    ):
        raise ValueError("PMT teacher/target robot mismatch")
    env.blind.load_state_dict(teacher["model_state_dict"], strict=True)
    env.blind.eval().requires_grad_(False)
    feat = Features(actor)

    def run(role, seed, collect=False, beta=0.0):
        # Native rollout queries labels only with features. Teacher execution is
        # used for its gate; evaluation student always has beta0 and no labels.
        if role == "teacher":
            r, data = rollout(env, actor, seed, beta=1.0, features=feat)
        else:
            r, data = rollout(
                env,
                actor,
                seed,
                beta=beta if collect else 0.0,
                features=feat if collect else None,
            )
        r["success_fraction"] = r["successes"] / r["num_episodes"]
        return r, data if collect else None

    return SimpleNamespace(
        run=run,
        close=env.env.close,
        modules=[env.teacher, env.blind],
        scope="Original PMT optimized scene0/targetG1,326D observable,30s native termination; no raw-terrain/vision transfer claim",
    )


def kick(task, actor, cfg, out):
    """Original Isaac Lab scene and legacy learned roundhouse as frozen teacher."""
    import os, h5py
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from modumotion_g1.official import load_official
    from modumotion_g1.domain_moe import load_domain, set_active_domains
    from modumotion_g1.perceptive_observations import (
        PerceptiveActorObservationInput,
        build_perceptive_actor_terms,
    )
    from tensordict import TensorDict

    first = ROOT / "reproduction"
    sys.path.insert(0, str(first))
    c = OmegaConf.load(first / "training/config.yaml")
    c.env.config.num_envs = task["num_envs"]
    c.env.config.seed = cfg["seed"]
    # Keep the old teacher's own merge bias; it is not a student merge bias.
    teacher, critic, _ = load_official("cuda:0")
    del critic
    path = ROOT / task["legacy_evidence_plugin"]
    name = load_domain(teacher, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    set_active_domains(teacher, [name], {name: payload["merge_bias"][0]})
    teacher.eval().requires_grad_(False)
    previous = os.getcwd()
    os.chdir(ROOT / "integrations/HoloMotion")
    try:
        env = instantiate(
            c.env,
            device="cuda:0",
            log_dir=str(out / "native_scene"),
            headless=True,
            accelerator=None,
        )
    finally:
        os.chdir(previous)
    native_env = env._env
    cmd = native_env.command_manager.get_term("ref_motion")
    robot = native_env.scene["robot"]
    N = task["num_envs"]
    feat = Features(actor)
    manifest = json.loads((first / "reference_h5/manifest.json").read_text())
    clip = manifest["clips"]["Roundhouse_kick"]
    meta = clip["metadata"]
    with h5py.File(first / "reference_h5/shards/Roundhouse_kick.h5", "r") as f:
        arrays = {}

        def visit(k, obj):
            if isinstance(obj, h5py.Dataset) and k.rsplit("/", 1)[-1] in [
                "ref_root_pos",
                "ref_root_rot",
                "ref_dof_pos",
            ]:
                arrays[k.rsplit("/", 1)[-1]] = obj[...]

        f.visititems(visit)
    q = arrays["ref_dof_pos"]
    p = arrays["ref_root_pos"]
    quat = arrays["ref_root_rot"]
    q = q.reshape(-1, 29)
    p = p.reshape(-1, 3)
    quat = quat.reshape(-1, 4)
    # HoloMotion HDF5 contract stores xyzw root quaternions, explicitly reorder.
    contract = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())
    holo = contract["joint_names"]
    gather = [meta["dof_names"].index(n) for n in holo]
    raw = torch.tensor(
        np.concatenate((p, quat[:, [3, 0, 1, 2]], q[:, gather]), -1),
        device="cuda:0",
        dtype=torch.float32,
    )
    joints = [robot.joint_names.index(n) for n in holo]
    default = torch.tensor(contract["default_joint_pos"], device="cuda:0")
    last = torch.zeros(N, 29, device="cuda:0")

    def run(role, seed, collect=False, beta=0.0):
        env.seed(seed)
        env.set_is_evaluating()
        obs, _ = env.reset_all()
        cmd.setup_offline_eval_deterministic()
        cmd._reference_observation_cache = {}
        cmd._update_ref_motion_state()
        cmd.force_realign_offline_eval_no_perturb(torch.arange(N, device="cuda:0"))
        native_env.scene.write_data_to_sim()
        native_env.sim.forward()
        native_env.scene.update(dt=0.0)
        actor.reset_kv_cache()
        teacher.reset_kv_cache(N, "cuda:0")
        teacher.clear_env_cache(None)
        last.zero_()
        alive = torch.ones(N, dtype=torch.bool, device="cuda:0")
        rows = []
        errors = []
        peak = torch.zeros(N, device="cuda:0")
        for step in range(len(raw) - 10):
            frame = (cmd.ref_motion_global_frame_ids - cmd.ref_motion_global_start_frame_ids).long()
            window = raw[
                (frame[:, None] + torch.arange(-1, 12, device="cuda:0")).clamp(0, len(raw) - 1)
            ].clone()
            window[..., :3] += native_env.scene.env_origins[:, None]
            CAPTURE["sampled_positions"] = (
                frame[:, None] + torch.arange(-1, 12, device="cuda:0")
            ).clamp(0, len(raw) - 1)
            terms = build_perceptive_actor_terms(
                PerceptiveActorObservationInput(
                    reference_qpos=window,
                    robot_root_quat_wxyz=robot.data.root_quat_w,
                    robot_root_angvel_local=robot.data.root_ang_vel_b,
                    robot_dof_pos=robot.data.joint_pos[:, joints],
                    robot_dof_vel=robot.data.joint_vel[:, joints],
                    last_action=last,
                    default_dof_pos=default,
                )
            )
            td = TensorDict({"unified/" + k: v for k, v in terms.items()}, batch_size=[N])
            actor.context = None
            mu = actor(td)["actions"]
            label = (
                teacher(td, mode="inference", update_obs_norm=False)["actions"]
                if collect or role == "teacher"
                else None
            )
            if collect:
                rows.append(feat.snapshot(label, mu))
            a = label if role == "teacher" else mu
            if collect and role != "teacher":
                a = torch.where(torch.rand(N, 1, device="cuda:0") < beta, label, mu)
            obs, reward, done, timeouts, info = env.step(a)
            last.copy_(a)
            alive &= ~(done & ~timeouts)
            ids = done.nonzero().flatten()
            if len(ids):
                actor.clear_env_cache(ids)
                teacher.clear_env_cache(ids)
                last[ids] = 0
            rp = cmd.get_ref_motion_bodylink_global_pos_immediate_next(prefix="ref_")
            e = (robot.data.body_pos_w - rp).norm(dim=-1).mean(-1)
            errors.append(e.detach())
            feet = [robot.body_names.index(s + "_ankle_roll_link") for s in ["left", "right"]]
            peak = torch.maximum(peak, robot.data.body_pos_w[:, feet, 2].amax(-1))
        err = torch.stack(errors).mean(0)
        good = alive & (err <= 0.15) & (peak >= 1.0)
        return dict(
            seed=seed,
            environments=N,
            success_fraction=float(good.float().mean()),
            mean_body_error_m=float(err.mean()),
            actual_peak_ankle_height_m=peak.tolist(),
            survived=int(alive.sum()),
            teacher="Frozen legacy roundhouse composite; not action reference labels",
        ), packed(rows)

    return SimpleNamespace(
        run=run,
        close=native_env.close,
        modules=[teacher],
        scope="Original roundhouse Isaac Lab scene and teacher; only kick clip, full valid reference and physical foot height/body error",
    )
