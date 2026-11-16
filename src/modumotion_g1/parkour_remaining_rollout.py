"""DAgger and strict first-episode evaluation for the remaining native tasks."""

from pathlib import Path

import numpy as np
import torch

from modumotion_g1.domain_moe import set_domain_context

MIN_PROGRESS = {"stairs_up": 2.0, "gaps": 2.0, "obstacles": 1.5}


def actor_action(task, actor):
    td, context = task.student_inputs()
    name = "parkour_" + task.task
    set_domain_context(actor, {name: context})
    try:
        action = actor(td, mode="inference")["actions"]
    finally:
        set_domain_context(actor, None)
    return action, context


def endings(task):
    p, tilt, distance = task.metrics()
    finite = torch.isfinite(p).all(-1) & torch.isfinite(tilt)
    fired = {
        n: task.env.termination_manager.get_term(n).bool().clone()
        for n in ("base_contact", "bad_orientation", "terrain_out_bound", "time_out")
    }
    failed = ~finite | fired["base_contact"] | fired["bad_orientation"] | fired["terrain_out_bound"]
    # Data.overflow accumulates all physics substeps until reset. This also
    # detects overflows that are absent in the last substep's contact buffers.
    overflow = task.overflow_flags()
    return p, tilt, distance, fired, failed, overflow


def collect(task, actor, features, seed, seconds, beta):
    task.reset(seed, actor)
    if task.overflow_flags().any():
        raise RuntimeError("Numerical overflow at the source reset")
    rows, student_steps, teacher_steps, failures, invalid = [], 0, 0, 0, 0
    for _ in range(round(seconds / task.env.step_dt)):
        teacher = task.teacher_actions()
        mu, context = actor_action(task, actor)
        # Every label is queried on the actual, pre-transition student state.
        # No sample is taken from a state after an invalid transition.
        rows.append(features.snapshot(task.native_to_holo(teacher), mu, context))
        choose = torch.rand(task.N, 1, device=task.device) < beta
        executed = torch.where(choose, teacher, task.holo_to_native(mu))
        teacher_steps += int(choose.sum())
        student_steps += int((~choose).sum())
        done = task.step(executed)
        _, _, _, _, failed, overflow = endings(task)
        failures += int(failed.sum())
        invalid += int((overflow != 0).sum())
        ids = (done | failed | (overflow != 0)).nonzero().flatten()
        if len(ids):
            task.reset(seed, actor, env_ids=ids)
            if task.overflow_flags()[ids].any():
                raise RuntimeError("Overflow did not clear at reset; collection aborted")
    return {
        "task": task.task,
        "controller": "mixed_DAgger_native_skill_profile",
        "teacher_probability": beta,
        "teacher_controlled_env_steps": teacher_steps,
        "student_controlled_env_steps": student_steps,
        "native_failure_resets": failures,
        "invalid_transition_resets": invalid,
        "post_overflow_states_used_for_labels": 0,
        "teacher_actions_are_labels_not_student_inputs": True,
        "seconds": seconds,
    }, {k: torch.cat([r[k] for r in rows]) for k in rows[0]}


def episode(task, actor, seed, seconds, recording=None):
    import mujoco

    task.reset(seed, actor)
    start, _, _ = task.metrics()
    start = start[0].clone()
    goal = task.route_goal[0].clone()
    goal_ground = float(task.ground(goal[None, :2].cpu().numpy())[0])
    near_steps, goal_dwell, first_pelvis_rule = 0, False, None
    minimum_distance, maximum_height = float("inf"), float(start[2])
    weights, poses, depths, commands, trace = [], [], [], [], []
    initial_flags = int(task.overflow_flags()[0])
    if initial_flags:
        raise RuntimeError(f"Source reset has numerical overflow: {initial_flags}")
    overflow_union = 0
    count = min(round(seconds / task.env.step_dt), task.env.max_episode_length)
    for step in range(count):
        if actor is None:
            action = task.teacher_actions()
        else:
            mu, _ = actor_action(task, actor)
            action = task.holo_to_native(mu)
            block = actor.actor_module.layers[0]
            weights.append(
                float(block.last_group_weights[..., -1].mean()) if block.active_domains else 0.0
            )
        if recording is not None and step % 2 == 0:
            poses.append(task.env.sim.data.qpos[0].cpu().numpy().copy())
            depths.append(
                task.obs[0, task.depth_slice].reshape(*task.depth_shape)[-1].cpu().numpy().copy()
            )
            commands.append(task.command.command[0].cpu().numpy().copy())
        done = task.step(action)
        p, tilt, distance, fired, failed, overflow = endings(task)
        overflow_union |= int(overflow[0])
        minimum_distance = min(minimum_distance, float(distance[0]))
        maximum_height = max(maximum_height, float(p[0, 2]))
        if float(tilt[0]) > 1 and first_pelvis_rule is None:
            first_pelvis_rule = float(task.elapsed[0])
        proper_height = goal_ground + 0.55 <= float(p[0, 2]) <= goal_ground + 1.1
        near_steps = (
            near_steps + 1
            if float(distance[0]) <= 0.4 and proper_height and not failed[0] and not overflow[0]
            else 0
        )
        goal_dwell |= near_steps >= round(1 / task.env.step_dt)
        trace.append(
            [
                float(task.elapsed[0]),
                *p[0].tolist(),
                float(tilt[0]),
                float(distance[0]),
                int(overflow[0]),
                int(failed[0]),
            ]
        )
        if done[0] or failed[0] or overflow[0]:
            break
    displacement = float((p[0, :2] - start[:2]).norm())
    climbed = float(p[0, 2] - start[2])
    survived = bool(fired["time_out"][0] and not failed[0] and not overflow_union)
    success = bool(
        survived
        and goal_dwell
        and displacement >= MIN_PROGRESS[task.task]
        and (task.task != "stairs_up" or climbed >= 0.25)
    )
    if recording is not None:
        recording = Path(recording)
        recording.mkdir(parents=True, exist_ok=False)
        mujoco.mj_saveModel(task.cpu_model, str(recording / "scene.mjb"))
        np.savez_compressed(
            recording / "physics.npz",
            qpos=np.asarray(poses),
            depth=np.asarray(depths),
            commands=np.asarray(commands),
            fps=25,
            trace=np.asarray(trace),
        )
    return (
        {
            "seed": seed,
            "success": success,
            "physical_survival_to_horizon": survived,
            "physical_failure": bool(failed[0]),
            "physics_valid": overflow_union == 0,
            "overflow_flags": overflow_union,
            "goal_dwell_1s_at_standing_height": goal_dwell,
            "start_pelvis_xyz": start.tolist(),
            "final_pelvis_xyz": p[0].tolist(),
            "goal_ground_xyz": [float(goal[0]), float(goal[1]), goal_ground],
            "horizontal_displacement_m": displacement,
            "height_gain_m": climbed,
            "minimum_goal_distance_m": minimum_distance,
            "final_goal_distance_m": float(distance[0]),
            "maximum_pelvis_height_m": maximum_height,
            "elapsed_s": float(task.elapsed[0]),
            "first_additional_pelvis_rule_crossing_s": first_pelvis_rule,
            "fired": {k: bool(v[0]) for k, v in fired.items()},
            "resets_before_episode_end": 0,
        },
        weights,
        step + 1,
    )


@torch.no_grad()
def rollout(task, actor=None, *, seed, seconds=20, features=None, beta=0.0, recording=None):
    if features is not None:
        return collect(task, actor, features, seed, seconds, beta)
    if task.N != 1:
        raise ValueError("Evaluation uses one environment so no failed world is stepped again")
    episodes, weights, env_steps = [], [], 0
    for index in range(4):
        result, w, steps = episode(
            task, actor, seed + index, seconds, recording if index == 0 else None
        )
        episodes.append(result)
        weights += w
        env_steps += steps
    successes = sum(e["success"] for e in episodes)
    return {
        "task": task.task,
        "controller": "official_parkour_teacher"
        if actor is None
        else "frozen_HoloMotion_public_plus_parkour_FFN_router",
        "profile": "source_reset_10s_episode_and_valid_flat_patch_goal_v2",
        "collision_workspace": task.collision_workspace,
        "terrain_difficulty_range": list(task.difficulty),
        "terrain": "pyramid_stairs_inv"
        if task.task == "stairs_up"
        else "square_gaps"
        if task.task == "gaps"
        else "boxes",
        "episodes": episodes,
        "successes": successes,
        "num_episodes": len(episodes),
        "acceptance_passed": successes / len(episodes) >= 0.9,
        "horizon_s": task.cfg.episode_length_s,
        "success_definition": "Survive the full native 10s without reset or overflow, reach the actual valid goal within 0.4m at standing height for 1s, satisfy terrain displacement and ascent checks",
        "minimum_horizontal_displacement_m": MIN_PROGRESS[task.task],
        "ascent_minimum_height_gain_m": 0.25 if task.task == "stairs_up" else None,
        "extra_pelvis_rule_used_as_termination": False,
        "native_terminations_preserved": True,
        "actual_physics": True,
        "mean_new_group_weight": float(np.mean(weights)) if weights else None,
        "teacher_controlled_env_steps": env_steps if actor is None else 0,
        "student_controlled_env_steps": env_steps if actor is not None else 0,
        "evaluates_first_episode_only": True,
    }, None
