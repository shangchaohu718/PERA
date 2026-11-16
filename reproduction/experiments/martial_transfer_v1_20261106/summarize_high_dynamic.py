"""Aggregate all locked high-dynamic repeats, including negative results."""

import csv
import json
from pathlib import Path
import numpy as np
from experiments.martial_transfer_v1_20261106.prepare import ROOT, save, sha


def main():
    folder = ROOT / "reports/martial_high_dynamic_20261106"
    selection = json.loads((folder / "selection_v2/protocol.json").read_text())
    runs = json.loads((folder / "confirmation_runs.json").read_text())
    assert runs["stage"] == "complete" and len(runs["runs"]) == 3
    reports = [json.loads((Path(p) / "report.json").read_text()) for p in runs["runs"]]
    for report in reports:
        assert (
            report["training_updates"] == 0
            and report["public_parameters_unchanged"]
            and report["matched_initial_states_exact"]
        )
    names = [r["name"] for r in selection["rows"]]
    assert all([r["name"] for r in d["paired_clips"]] == names for d in reports)
    perclip = []
    metrics = [
        "base_mean_completion",
        "trained_mean_completion",
        "completion_change",
        "base_full_completion_rate",
        "trained_full_completion_rate",
        "full_completion_rate_change",
    ]
    for i, name in enumerate(names):
        row = selection["rows"][i]
        records = [d["paired_clips"][i] for d in reports]
        perclip.append(
            dict(
                name=name,
                pool=row["pool"],
                description=row["description"],
                **{key: float(np.mean([r[key] for r in records])) for key in metrics},
                individual_repeat_results=records,
            )
        )
    pools = {}
    for pool in dict.fromkeys(r["pool"] for r in perclip):
        chosen = [r for r in perclip if r["pool"] == pool]
        pools[pool] = dict(
            clips=len(chosen),
            physical_rollouts_per_condition=len(chosen) * 8 * 3,
            **{key: float(np.mean([r[key] for r in chosen])) for key in metrics},
            repeat_results=[r["pools"][pool] for r in reports],
        )
    hashes = {
        key: dict(
            file=row["file"], sha256=sha(row["file"]), unchanged=sha(row["file"]) == row["sha256"]
        )
        for key, row in selection["checkpoints"].items()
    }
    assert all(r["unchanged"] for r in hashes.values())
    report = dict(
        robot="unitree_g1_29dof",
        comparison="same base vs frozen base plus fixed2000-round martial plugin, trueTop2",
        reference_selection=str(folder / "selection_v2/protocol.json"),
        physical_repeats=3,
        replicas_per_clip_per_repeat=8,
        trained_expert_training_seeds=1,
        training_updates=0,
        pools=pools,
        per_clip=perclip,
        checkpoint_audit=hashes,
        run_reports=runs["runs"],
        conclusions=[
            "Fast-dance mean completion improves in these locked descriptive evaluations; other groups do not show general improvement.",
            "Report the complete32-motion set; the largest positive example was identified after evaluation.",
            "Repeated physical evaluations do not establish robustness across independent expert trainings.",
            "The existing expert was trained on7PBHC motions includingCharleston dance; this is unseen-motion transfer, not proof of martial-only cross-domain transfer.",
            "Completion only measures reaching reference end without termination, not accurate execution of every high-dynamic pose.",
        ],
    )
    save(folder / "report.json", report)
    with (folder / "per_clip.csv").open("w", newline="") as f:
        fields = ["name", "pool", "description"] + metrics
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(perclip)
    lines = [
        "G1 未训练高动态参考动作：固定基座 vs 加入已训练竞争专家",
        "32条，本地G1映射数据，4类各8条；同一清单3轮，每条每轮8个参考初态副本；零训练更新。",
        "",
    ]
    for pool, r in pools.items():
        lines.append(
            f"{pool}: 平均完成度 {r['base_mean_completion']:.2%} → {r['trained_mean_completion']:.2%}；整段完成率 {r['base_full_completion_rate']:.2%} → {r['trained_full_completion_rate']:.2%}"
        )
    lines.append("\n逐轮：")
    for pool, r in pools.items():
        lines.append(
            pool
            + ": "
            + ", ".join(
                f"{x['base_mean_completion']:.2%}→{x['trained_mean_completion']:.2%}"
                for x in r["repeat_results"]
            )
        )
    top = max(perclip, key=lambda r: r["completion_change"])
    lines.append(
        f"\n观察到的最大改善例（事后选例）：{top['name']}，平均完成度{top['base_mean_completion']:.2%}→{top['trained_mean_completion']:.2%}，整段完成率{top['base_full_completion_rate']:.2%}→{top['trained_full_completion_rate']:.2%}"
    )
    lines.append(
        "同一个已训练专家；物理复跑不是多个训练种子。训练动作中包含Charleston，不能据此声称仅武术训练实现跨领域泛化。"
    )
    (folder / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
