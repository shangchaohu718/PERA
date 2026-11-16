"""Native bounded episodes; validation always executes the student alone."""

import torch

from modumotion_g1.domain_moe import set_domain_context


@torch.no_grad()
def rollout(task, actor, seed, beta=0.0, features=None):
    if beta and features is None:
        raise ValueError("Validation cannot use teacher assistance")
    obs = task.reset(seed, actor, from_start=True)
    active = torch.ones(task.N, device=task.device, dtype=torch.bool)
    rows, episodes = [], [None] * task.N
    queries, student_steps = 0, 0
    weights, errors = [], []
    clip_ids = task.raw.motion_ids.clone()
    for step in range(1551):
        valid = active.clone()
        teacher = task.teacher_action(obs) if features is not None else None
        if teacher is not None:
            queries += int(valid.sum())
        td, context = task.student_inputs(obs)
        set_domain_context(actor, context)
        mu = actor(td, mode="inference")["actions"]
        set_domain_context(actor, None)
        if features is not None:
            row = features.snapshot(task.action_to_holo(teacher), mu, context)
            rows.append({k: v[valid.cpu()].clone() for k, v in row.items()})
        weight = actor.actor_module.layers[0].last_group_weights[valid, ..., -1]
        weights.append(float(weight.mean()))
        errors.append(
            float(
                (task.motion.anchor_pos_w - task.motion.robot_anchor_pos_w)
                .norm(dim=-1)[valid]
                .mean()
            )
        )
        native = task.action_to_native(mu)
        if teacher is None:
            action = native
            student_steps += int(valid.sum())
        else:
            choose = torch.rand(task.N, 1, device=task.device) < beta
            action = torch.where(choose, teacher, native)
            student_steps += int((valid & ~choose[:, 0]).sum())
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite rollout action")
        obs, _, terminated, truncated, _ = task.step(action)
        ended = (terminated | truncated | task.completions) & valid
        for i in ended.nonzero().flatten().tolist():
            episodes[i] = {
                "env": i,
                "clip_id": int(clip_ids[i]),
                "seconds": (step + 1) * 0.02,
                "full_reference_completed": bool(task.completions[i]),
                "failed": bool(terminated[i]),
                "timeout": bool(truncated[i]),
                "termination_reasons": [
                    n
                    for n in task.env.termination_manager.active_terms
                    if bool(task.env.termination_manager.get_term(n)[i])
                ],
            }
        active &= ~ended
        reset = task.resampled.nonzero().flatten()
        if len(reset):
            actor.clear_env_cache(reset)
        if not active.any():
            break
    result = {
        "seed": seed,
        "teacher_probability": beta,
        "teacher_action_queries": queries,
        "student_executed_steps": student_steps,
        "episodes": episodes,
        "successes": sum(
            bool(r and r["full_reference_completed"] and not r["failed"]) for r in episodes
        ),
        "num_episodes": task.N,
        "mean_seconds": sum(r["seconds"] for r in episodes if r) / task.N,
        "mean_anchor_error_m": sum(errors) / len(errors),
        "mean_expert_weight": sum(weights) / len(weights),
        "unfinished_envs": active.nonzero().flatten().tolist(),
    }
    data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None
    return result, data
