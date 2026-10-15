"""B0: frozen per-task gradient audit for the privileged Teacher.

The command collects a fresh rollout, forms the exact currently weighted v3
Teacher objectives, and differentiates each task separately with respect to the
shared input projection and Transformer encoder.  It never calls optimizer.step.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from tools.audit_teacher_multihorizon import GROUP_NAMES, _motion_paths, _valid_horizon
from unified_motion.algorithms.sequence_training import parameter_stamp
from unified_motion.data.motion_sampler import MotionLibrary, MotionSampler
from unified_motion.envs.mjlab_k1 import K1MJLabBridge
from unified_motion.holomotion_mjlab import (
    HoloMotionK1MJLabAdapter,
    HoloMotionOnnxContract,
    make_holomotion_k1_mjlab_cfg,
)
from unified_motion.holomotion_mjlab_task import (
    HoloMotionK1MJLabTask,
    HoloMotionPhysicsRandomizationConfig,
)
from unified_motion.k1_policy_runtime import resume_k1_trainer, runtime_configuration
from unified_motion.privileged_teacher import core_groups, normalized_core

DEFAULT_HOLOMOTION_ROOT = PROJECT_ROOT / "third_party" / "HoloMotion" / "source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--exclude-motion-dir", type=Path)
    parser.add_argument(
        "--include-motion-prefix",
        action="append",
        default=[],
        help="restrict evaluation to complete original motions by filename prefix; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--holomotion-root", type=Path, default=DEFAULT_HOLOMOTION_ROOT)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 16:
        parser.error("--num-envs must be positive and --steps must be at least 16")
    return args


def _masked_mse(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value.square(), 0.0).sum() / mask.sum().clamp_min(1)


def _shared_parameters(teacher):
    named = []
    for prefix, module in (("input_projection", teacher.input_projection),
                           ("encoder", teacher.encoder)):
        named.extend((f"{prefix}.{name}", parameter)
                     for name, parameter in module.named_parameters())
    return named


def _task_gradient(loss, parameters, *, retain_graph: bool):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]).detach()


def gradient_report(weighted_losses: dict[str, torch.Tensor], shared_parameters) -> dict:
    """Differentiate independent weighted tasks and report pre-clip geometry."""
    names = list(weighted_losses)
    parameters = list(shared_parameters)
    vectors = {}
    for index, name in enumerate(names):
        vector = _task_gradient(
            weighted_losses[name], parameters, retain_graph=index + 1 < len(names))
        if not torch.isfinite(vector).all():
            raise ValueError(f"non-finite shared gradient for {name}")
        vectors[name] = vector
    norms = {name: float(vector.norm()) for name, vector in vectors.items()}
    cosine = {}
    for left in names:
        cosine[left] = {}
        for right in names:
            denominator = vectors[left].norm() * vectors[right].norm()
            cosine[left][right] = (None if not bool(denominator > 0) else
                                   float(torch.dot(vectors[left], vectors[right]) / denominator))
    total = torch.stack(list(vectors.values())).sum(0)
    one_step_state = torch.stack([
        vector for name, vector in vectors.items() if name.startswith("h1/") and name != "h1/contact"
    ]).sum(0)
    multistep_state = torch.stack([
        vector for name, vector in vectors.items() if name.startswith(("h4/", "h8/"))
    ]).sum(0)
    contact = vectors["h1/contact"]
    def vector_cosine(left, right):
        denominator = left.norm() * right.norm()
        return None if not bool(denominator > 0) else float(torch.dot(left, right) / denominator)
    return {
        "losses_after_current_weight": {name: float(loss.detach())
                                         for name, loss in weighted_losses.items()},
        "shared_gradient_norm_pre_clip": norms,
        "shared_gradient_cosine": cosine,
        "aggregates": {
            "total_norm_pre_clip": float(total.norm()),
            "would_trigger_global_clip_at_1": bool(total.norm() > 1.0),
            "one_step_state_norm": float(one_step_state.norm()),
            "multistep_state_norm": float(multistep_state.norm()),
            "contact_norm": float(contact.norm()),
            "contact_to_one_step_state_norm_ratio": float(
                contact.norm() / one_step_state.norm().clamp_min(1e-12)),
            "contact_to_all_state_norm_ratio": float(
                contact.norm() / (one_step_state + multistep_state).norm().clamp_min(1e-12)),
            "contact_vs_one_step_state_cosine": vector_cosine(contact, one_step_state),
            "contact_vs_multistep_state_cosine": vector_cosine(contact, multistep_state),
            "one_step_vs_multistep_state_cosine": vector_cosine(one_step_state, multistep_state),
        },
    }


def build_weighted_losses(trainer, rollout, begin: int) -> dict[str, torch.Tensor]:
    teacher = trainer.privileged_teacher
    if teacher is None or teacher.config.objective_version != 3:
        raise ValueError("B0 requires a v3 privileged Teacher")
    inputs = rollout.privileged_teacher_input
    current = rollout.privileged_teacher_core
    ids = rollout.privileged_teacher_env_ids
    if inputs is None or current is None or ids is None or len(ids) != rollout.actions.shape[0]:
        raise ValueError("rollout lacks complete Teacher supervision")
    starts = rollout.physics_episode_start.index_select(0, ids)
    dones = rollout.dones.squeeze(-1).index_select(0, ids)
    actions_all = rollout.actions if rollout.executed_actions is None else rollout.executed_actions
    actions = actions_all.index_select(0, ids)
    output = teacher(inputs, starts, actions, current)
    truth_all = normalized_core(
        rollout.next_physics_observation, trainer.dynamics.robot).index_select(0, ids)
    valid_all = rollout.next_physics_observation.core_valid.index_select(0, ids)
    transition = ~dones[:, begin:]
    delta_error = (output["delta_core"][:, begin:]
                   - (truth_all[:, begin:] - current[:, begin:]).detach())
    if teacher.config.delta_whitening:
        _, one_step_std = teacher.delta_statistics(1, delta_error)
        delta_error = delta_error / one_step_std
    losses = {}
    # teacher_supervision averages the five one-step groups, hence 1/5 each.
    for name, section in zip(GROUP_NAMES, core_groups(truth_all.shape[-1]), strict=True):
        mask = valid_all[:, begin:, section] & transition.unsqueeze(-1)
        losses[f"h1/{name}"] = 0.2 * _masked_mse(delta_error[..., section], mask)

    count = teacher.config.contact_slots
    labels = rollout.next_physics_observation.contact_features.index_select(
        0, ids)[:, begin:, :count, 10]
    contact_mask = (rollout.next_physics_observation.contact_mask.index_select(
        0, ids)[:, begin:, :count] & transition.unsqueeze(-1))
    contact_loss = F.binary_cross_entropy_with_logits(
        output["contact_logits"][:, begin:, :count], labels.detach(), reduction="none")
    losses["h1/contact"] = torch.where(contact_mask, contact_loss, 0.0).sum() / contact_mask.sum().clamp_min(1)

    future_horizons = tuple(h for h in teacher.config.prediction_horizons if h > 1)
    per_task_weight = teacher.config.multistep_weight / (len(future_horizons) * len(GROUP_NAMES))
    for horizon in future_horizons:
        length = actions.shape[1] - horizon + 1
        if length <= begin:
            continue
        valid_transition = _valid_horizon(starts, dones, horizon)[:, begin:]
        window = torch.stack(
            [actions[:, begin + offset:length + offset] for offset in range(horizon)], -2)
        predicted = teacher.predict_future_core(
            output["latent"][:, begin:length], window, current[:, begin:length], horizon=horizon)
        truth = truth_all[:, begin + horizon - 1:length + horizon - 1]
        target_valid = valid_all[:, begin + horizon - 1:length + horizon - 1]
        for name, section in zip(GROUP_NAMES, core_groups(truth.shape[-1]), strict=True):
            mask = target_valid[..., section] & valid_transition.unsqueeze(-1)
            error = ((predicted[..., section] - current[:, begin:length, section])
                     - (truth[..., section] - current[:, begin:length, section]).detach())
            if teacher.config.delta_whitening:
                _, std = teacher.delta_statistics(horizon, predicted)
                error = error / std[..., section]
            losses[f"h{horizon}/{name}"] = per_task_weight * _masked_mse(error, mask)
    return losses


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; B0 audit was not started")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    runtime = runtime_configuration(payload)
    contract = HoloMotionOnnxContract(**runtime["task_contract"])
    paths, excluded = _motion_paths(args.motion_dir, args.exclude_motion_dir)
    if args.include_motion_prefix:
        paths = [path for path in paths if any(
            path.name.startswith(prefix + "__") for prefix in args.include_motion_prefix)]
        if not paths:
            raise ValueError("motion-prefix restriction removed every candidate window")
    library = MotionLibrary.from_named_mjlab_files(paths, robot_id="k1_22dof", device=device)
    offsets = tuple(index * 0.02 for index in range(contract.future_frames + 2))
    sampler = MotionSampler(library, num_envs=args.num_envs, step_dt=0.02,
                            future_offsets_s=offsets)
    cfg = make_holomotion_k1_mjlab_cfg(args.asset_root, contract, num_envs=args.num_envs)
    cfg.seed = args.seed
    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        task = HoloMotionK1MJLabTask(
            HoloMotionK1MJLabAdapter(
                K1MJLabBridge(env, reference_steps=contract.future_frames + 2), contract),
            sampler,
            physics_randomization=HoloMotionPhysicsRandomizationConfig(
                **runtime["physics_randomization"]),
        )
        trainer = resume_k1_trainer(args.checkpoint, task,
                                    holomotion_root=args.holomotion_root)
        teacher = trainer.privileged_teacher
        teacher.eval()
        shared = _shared_parameters(teacher)
        before = parameter_stamp(teacher)
        begin = teacher.config.context_steps
        minimum_remaining = args.steps * sampler.step_dt + float(sampler.offsets_s[-1])
        trainer.reset(random_start=True, min_remaining_s=minimum_remaining)
        rollout = trainer.collect(args.steps)
        weighted_losses = build_weighted_losses(trainer, rollout, begin)
        diagnostics = gradient_report(weighted_losses, [parameter for _, parameter in shared])
        if before != parameter_stamp(teacher):
            raise RuntimeError("B0 audit unexpectedly changed Teacher parameters")
        document = {
            "schema_version": 1,
            "stage": "B0_gradient_diagnostics_only",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_iteration": payload.get("iter"),
            "teacher_updates": payload["privileged_teacher"]["updates"],
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "burn_in_steps": begin,
            "motion_candidates": len(paths),
            "included_motion_prefixes": args.include_motion_prefix,
            "excluded_training_count": len(excluded),
            "shared_parameter_scope": [name for name, _ in shared],
            "shared_parameter_count": sum(parameter.numel() for _, parameter in shared),
            "weights_unchanged": True,
            **diagnostics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
