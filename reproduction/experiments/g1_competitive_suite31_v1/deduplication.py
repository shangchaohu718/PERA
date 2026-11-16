"""Skip matching current-base tasks, without declaring them accepted.

A shared skill label is insufficient: MimicX stairs differs from Instinct
parkour stairs, and MimicX kungfu differs from PBHC roundhouse/martial clips.
"""

import json
from pathlib import Path

from pera.paths import ROOT


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def duplicate(task, base_sha256, exclude_run=None, root=ROOT):
    matches = []
    # These plans explicitly record the same frozen MLP checkpoint and exact
    # PBHC Roundhouse_kick source. They are not the official Transformer base.
    if task["id"] == "roundhouse" and task["family"] == "kick":
        for dirname in [
            "blink_competitive_roundhouse_ppo_v1_20261106",
            "blink_competitive_martial_ppo_v1_20261106",
        ]:
            folder = root / "reports" / dirname
            plan = read(folder / "plan.json")
            if plan.get("frozen_base_sha256") != base_sha256 or "Roundhouse_kick" not in plan.get(
                "reference_names", []
            ):
                continue
            reports = sorted(folder.glob("training*/status.json"))
            for path in reports:
                status = read(path)
                updates = status.get("actor_adam_updates", 0)
                if updates <= 0:
                    continue
                matches.append(
                    dict(
                        task="roundhouse",
                        kind="same_base_existing_training",
                        actual_updates=updates,
                        stage=status.get("stage"),
                        evidence=str(path),
                        pid=status.get("pid"),
                        accepted=False,
                        acceptance="Not inferred from training completion",
                    )
                )
    for path in sorted((root / "reports/g1_competitive_suite31_v1").glob("*/pipeline.json")):
        if exclude_run is not None and path.parent.resolve() == Path(exclude_run).resolve():
            continue
        pipeline = read(path)
        if pipeline.get("base_sha256") != base_sha256:
            continue
        entry = pipeline.get("tasks", {}).get(task["id"], {})
        native = read(Path(entry["output"]) / "report.json") if entry.get("output") else {}
        updates = max(
            entry.get("actual_Adam_updates", 0),
            native.get("actual_Adam_updates", 0),
        )
        if updates <= 0:
            continue
        matches.append(
            dict(
                task=task["id"],
                kind="same_base_suite_training_exists",
                actual_updates=updates,
                stage=native.get("status", entry.get("status")),
                evidence=str(path),
                checkpoint=entry.get("checkpoint"),
                accepted=bool(entry.get("accepted_for_integration", False)),
            )
        )
    if not matches:
        return None
    return dict(
        reason="Same base and same task already has optimizer updates; do not restart from scratch automatically",
        evidence=matches,
    )
