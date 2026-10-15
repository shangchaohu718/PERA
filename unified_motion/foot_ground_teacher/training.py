"""Reproducible offline training primitives; this module never starts PPO."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import random

import numpy as np
import torch

from .objective import gradient_audit, teacher_losses
from .schema import FootGroundBatch, RegressionStatistics


REGRESSION_FIELDS = (
    "contact_point_local", "relative_velocity_normal",
    "relative_velocity_tangent", "slip_speed", "normal_force",
    "tangential_force", "contact_impulse", "foot_load_ratio",
    "delta_joint_velocity", "delta_root_angular_velocity",
    "joint_velocity_residual",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_batch(path: str | Path, device="cpu") -> FootGroundBatch:
    """Load one complete trajectory as a batch of size one."""
    with np.load(path, allow_pickle=False) as archive:
        values = {}
        for item in fields(FootGroundBatch):
            if item.name not in archive:
                if item.name in {"joint_velocity_residual_valid", "reference_intent"}:
                    values[item.name] = None
                    continue
                raise ValueError(f"{path} lacks {item.name}")
            values[item.name] = torch.from_numpy(archive[item.name]).unsqueeze(0).to(device)
    return FootGroundBatch(**values).validate()


def _masked_feature_columns(value: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """Select every output channel independently without mixing masked feet."""
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    width = value.shape[-1]
    flat_value = value.reshape(-1, width).float()
    flat_mask = mask.reshape(-1, width)
    return [flat_value[:, index][flat_mask[:, index]] for index in range(width)]


def fit_statistics(paths, *, std_floor=1e-5) -> RegressionStatistics:
    """Fit only on manifest train paths with the same masks used by losses."""
    chunks: dict[str, list[list[torch.Tensor]] | None] = {
        name: None for name in REGRESSION_FIELDS}
    for path in paths:
        batch = load_batch(path)
        valid = batch.valid_transition.unsqueeze(-1)
        contact = valid & batch.contact_valid
        for name in REGRESSION_FIELDS:
            value = getattr(batch, name)
            mask = contact if name in {
                "contact_point_local", "relative_velocity_normal",
                "relative_velocity_tangent", "slip_speed", "normal_force",
                "tangential_force",
            } else (valid & (batch.contact_valid | (batch.contact_mode == 3))) \
                if name == "contact_impulse" else batch.joint_velocity_residual_valid \
                if name == "joint_velocity_residual" else batch.valid_transition
            selected = _masked_feature_columns(value, mask)
            if chunks[name] is None:
                chunks[name] = [[] for _ in selected]
            for destination, column in zip(chunks[name], selected):
                if column.numel():
                    destination.append(column)
    mean, std, count = {}, {}, {}
    for name, columns in chunks.items():
        if columns is None or any(not values for values in columns):
            raise ValueError(f"training split has no valid {name} labels")
        joined = [torch.cat(values).double() for values in columns]
        mean[name] = tuple(float(values.mean()) for values in joined)
        std[name] = tuple(max(float(values.std(unbiased=False)), std_floor)
                          for values in joined)
        # A scalar count retains the conservative per-output support.
        count[name] = min(int(values.numel()) for values in joined)
    return RegressionStatistics(mean, std, count, std_floor)


def fit_contact_mode_class_weights(paths, *, classes=4) -> tuple[float, ...]:
    """Inverse-frequency mode weights fitted only from valid train transitions."""
    counts = torch.zeros(classes, dtype=torch.long)
    for path in paths:
        batch = load_batch(path)
        valid = batch.valid_transition.unsqueeze(-1).expand_as(batch.contact_mode)
        counts += torch.bincount(batch.contact_mode[valid], minlength=classes).cpu()
    if (counts == 0).any():
        raise ValueError(f"training split lacks contact mode(s): {counts.tolist()}")
    weights = counts.sum().double() / (classes * counts.double())
    weights /= weights.mean()
    return tuple(float(value) for value in weights)


def train_epoch(model, paths, optimizer, statistics, *, loss_config=None,
                replay_updates=1, audit_every=0, step_offset=0):
    """Train sequentially on full trajectories and report actual update count."""
    if replay_updates < 1:
        raise ValueError("replay_updates must be positive")
    model.train()
    records = []
    update = 0
    for path in paths:
        batch = load_batch(path, next(model.parameters()).device)
        for _ in range(replay_updates):
            output = model(batch.student_observation, batch.privileged_current,
                           batch.executed_action, batch.foot_current,
                           batch.episode_start, batch.reference_intent)
            losses = teacher_losses(output, batch, statistics, loss_config)
            audit = (gradient_audit(model, losses) if audit_every and
                     (step_offset + update) % audit_every == 0 else {})
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            update += 1
            records.append({"optimizer_update": step_offset + update,
                            "source": str(path), "total": float(losses["total"].detach()),
                            "gradient_norm/total": float(norm), **audit,
                            **{f"loss/{name}": float(value.detach())
                               for name, value in losses.items()
                               if name != "total" and not name.endswith("_group")},
                            **{name: float(losses[name].detach()) for name in (
                                "contact_group", "geometry_group", "kinematics_group",
                                "force_group", "dynamics_group")}})
    return records


def append_jsonl(path, records):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
