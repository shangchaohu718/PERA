"""Offline Teacher V1 training. This command cannot update an Actor or PPO."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from unified_motion.foot_ground_teacher.model import (  # noqa: E402
    FootGroundTeacher, FootGroundTeacherConfig,
)
from unified_motion.foot_ground_teacher.objective import (  # noqa: E402
    TASK_PROFILE_ACTIVE_LOSSES, FootGroundLossConfig,
)
from unified_motion.foot_ground_teacher.training import (  # noqa: E402
    append_jsonl, fit_contact_mode_class_weights, fit_statistics, load_batch,
    seed_everything, train_epoch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--contact-weight", type=float, choices=(1.0, .1, .03), required=True)
    parser.add_argument("--task-profile", choices=("v1", "v2_a1", "v2_a2_core3"),
                        default="v1")
    parser.add_argument("--replay-updates", type=int, default=1)
    parser.add_argument("--gradient-audit-every", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1 or args.replay_updates < 1 or args.seed < 0:
        parser.error("epochs/replay-updates must be positive and seed non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback run was started")
    manifest = json.loads(args.manifest.read_text())
    train_paths = [row["path"] for row in manifest["records"] if row["split"] == "train"]
    if not train_paths or not any(row["split"] == "validation" for row in manifest["records"]):
        raise ValueError("manifest needs non-empty train and validation splits")
    if not manifest.get("heldout_checkpoints") or not any(
            row["split"] == "test" and row["metadata"]["checkpoint_id"] in
            manifest["heldout_checkpoints"] for row in manifest["records"]):
        raise ValueError("manifest needs a held-out checkpoint test source")
    seed_everything(args.seed)
    probe = load_batch(train_paths[0])
    config = FootGroundTeacherConfig(
        student_observation_dim=probe.student_observation.shape[-1],
        privileged_current_dim=probe.privileged_current.shape[-1],
        foot_current_dim=probe.foot_current.shape[-1],
        reference_intent_dim=(0 if probe.reference_intent is None
                              else probe.reference_intent.shape[-1]),
    )
    model = FootGroundTeacher(config).to(args.device)
    statistics = fit_statistics(train_paths)
    mode_class_weights = fit_contact_mode_class_weights(train_paths)
    loss_config = FootGroundLossConfig(
        task_profile=args.task_profile,
        contact_weight=args.contact_weight,
        contact_mode_class_weights=mode_class_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    statistics.to_json(args.output_dir / "train_statistics.json")
    protocol = {
        "schema": 1, "seed": args.seed, "manifest": str(args.manifest.resolve()),
        "model": asdict(config), "loss": asdict(loss_config),
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "device": args.device,
        "replay_updates": args.replay_updates,
        "gradient_audit_every": args.gradient_audit_every,
        "optimizer": "AdamW", "ppo_updates": 0,
        "trajectory_order": "seeded_epoch_shuffle_v1",
        "contact_mode_class_weighting": "inverse_frequency_train_split",
        "future_action_contract": "executed action at t predicts labels at t+1",
        "task_profile": args.task_profile,
        "active_supervision": list(TASK_PROFILE_ACTIVE_LOSSES[args.task_profile]),
        "reference_intent_contract": (
            "none" if config.reference_intent_dim == 0 else
            "current reference phase and planned foot roles at t; no realized t+1 labels"
        ),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    updates = 0
    for epoch in range(args.epochs):
        epoch_paths = list(train_paths)
        random.Random(args.seed + epoch).shuffle(epoch_paths)
        records = train_epoch(model, epoch_paths, optimizer, statistics,
            loss_config=loss_config, replay_updates=args.replay_updates,
            audit_every=args.gradient_audit_every, step_offset=updates)
        updates += len(records)
        for row in records:
            row["epoch"] = epoch
        append_jsonl(args.output_dir / "metrics.jsonl", records)
    torch.save({"schema": 1, "model_config": asdict(config),
                "model_state_dict": model.state_dict(), "optimizer_updates": updates,
                "seed": args.seed, "contact_weight": args.contact_weight,
                "task_profile": args.task_profile,
                "active_supervision": list(TASK_PROFILE_ACTIVE_LOSSES[args.task_profile]),
                "statistics": asdict(statistics), "protocol": protocol},
               args.output_dir / "teacher.pt")
    print(json.dumps({"optimizer_updates": updates,
                      "checkpoint": str((args.output_dir / 'teacher.pt').resolve())}))


if __name__ == "__main__":
    main()
