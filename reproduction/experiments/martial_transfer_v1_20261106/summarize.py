"""Paired, clip-balanced pilot results; never promote a plugin from test scores."""

import csv
import json
from pathlib import Path
import numpy as np
from experiments.martial_transfer_v1_20261106.prepare import REPORT, save, sha

CONDITIONS = ["base", "initialized", "trained", "trained_disabled"]


def cluster_interval(values, groups):
    unique = sorted(set(groups))
    indices = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    rng = np.random.default_rng(99117)
    means = []
    for _ in range(5000):
        take = rng.choice(unique, len(unique), replace=True)
        means.append(float(np.mean(np.concatenate([values[indices[g]] for g in take]))))
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    results = {name: json.loads((REPORT / name / "result.json").read_text()) for name in CONDITIONS}
    protocol_hash = sha(REPORT / "protocol.json")
    frames = {name: dict(np.load(REPORT / name / "frame_metrics.npz")) for name in CONDITIONS}
    reference = results["base"]["records"]
    initial = dict(np.load(REPORT / "base/initial_states.npz"))
    state_errors = {}
    for name, result in results.items():
        assert result["full_evaluation"] and result["completed_bookkeeping"]
        assert result["protocol_sha256"] == protocol_hash and result["public_unchanged"]
        assert [(r["clip"], r["case"]) for r in result["records"]] == [
            (r["clip"], r["case"]) for r in reference
        ]
        actual = dict(np.load(REPORT / name / "initial_states.npz"))
        error = max(float(np.max(np.abs(actual[k] - initial[k]))) for k in initial)
        state_errors[name] = error
        assert error == 0, ("Mismatched paired initial states", name, error)
    comparisons = []
    for name in CONDITIONS[1:]:
        for i, (base, row) in enumerate(zip(reference, results[name]["records"])):
            count = min(base["observed_steps"], row["observed_steps"])
            a = frames[name]["body_rms"][:count, i]
            b = frames["base"]["body_rms"][:count, i]
            comparisons.append(
                dict(
                    condition=name,
                    clip=row["clip"],
                    pool=row["pool"],
                    case=row["case"],
                    split_group=row["split_group"],
                    return_delta=row["normalized_return"] - base["normalized_return"],
                    survival_delta=int(row["full_clip_survival"]) - int(base["full_clip_survival"]),
                    body_common_prefix_delta_m=float(np.mean(a - b)),
                    common_prefix_steps=count,
                    body_common_prefix_base_m=float(np.mean(b)),
                    body_common_prefix_condition_m=float(np.mean(a)),
                    root_xy_common_prefix_delta_m=float(
                        np.mean(
                            frames[name]["root_xy"][:count, i]
                            - frames["base"]["root_xy"][:count, i]
                        )
                    ),
                    new_admission=row["new_admission"],
                    new_contribution=row["new_contribution"],
                )
            )
    csv_path = REPORT / "paired_cases.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    pools = sorted(set(r["pool"] for r in reference))
    summary = {}
    for pool in pools:
        names = sorted(set(r["clip"] for r in reference if r["pool"] == pool))
        item = dict(
            clips=len(names),
            initial_conditions=sum(r["pool"] == pool for r in reference),
            conditions={},
            paired_vs_base={},
        )
        for name in CONDITIONS:
            source = results[name]["records"]
            by_clip = {clip: [r for r in source if r["clip"] == clip] for clip in names}
            item["conditions"][name] = {
                "normalized_return": float(
                    np.mean(
                        [np.mean([r["normalized_return"] for r in rs]) for rs in by_clip.values()]
                    )
                ),
                "full_clip_survival": float(
                    np.mean(
                        [np.mean([r["full_clip_survival"] for r in rs]) for rs in by_clip.values()]
                    )
                ),
                "new_admission": float(
                    np.mean([np.mean([r["new_admission"] for r in rs]) for rs in by_clip.values()])
                ),
                "new_contribution": float(
                    np.mean(
                        [np.mean([r["new_contribution"] for r in rs]) for rs in by_clip.values()]
                    )
                ),
            }
        for name in CONDITIONS[1:]:
            selected = [r for r in comparisons if r["pool"] == pool and r["condition"] == name]
            perclip = []
            for clip in names:
                records = [r for r in selected if r["clip"] == clip]
                perclip.append(
                    dict(
                        clip=clip,
                        split_group=records[0]["split_group"],
                        **{
                            key: float(np.mean([r[key] for r in records]))
                            for key in (
                                "return_delta",
                                "survival_delta",
                                "body_common_prefix_delta_m",
                                "root_xy_common_prefix_delta_m",
                            )
                        },
                    )
                )
            groups = [r["split_group"] for r in perclip]
            metrics = {}
            for key in (
                "return_delta",
                "survival_delta",
                "body_common_prefix_delta_m",
                "root_xy_common_prefix_delta_m",
            ):
                values = np.asarray([r[key] for r in perclip])
                metrics[key] = dict(
                    mean=float(values.mean()),
                    source_group_bootstrap95=cluster_interval(values, groups),
                )
            metrics.update(
                clips_return_improved=sum(r["return_delta"] > 1e-4 for r in perclip),
                clips_return_worsened=sum(r["return_delta"] < -1e-4 for r in perclip),
                source_groups=len(set(groups)),
                per_clip=perclip,
            )
            item["paired_vs_base"][name] = metrics
        summary[pool] = item
    trained_metrics = summary["unseen_related"]["paired_vs_base"]["trained"]
    init_metrics = summary["unseen_related"]["paired_vs_base"]["initialized"]
    report = dict(
        protocol=json.loads((REPORT / "protocol.json").read_text()),
        paired_initial_state_max_abs_errors=state_errors,
        pools=summary,
        frozen_public_audit={
            name: {
                k: v
                for k, v in result.items()
                if k
                in (
                    "public_sha256_before",
                    "public_sha256_after",
                    "public_unchanged",
                    "old_selection_bias_unchanged",
                    "new_selection_bias_unchanged",
                    "disabled_vs_native_action_max_abs_error",
                    "elapsed_seconds",
                    "torch_peak_allocated_mib",
                )
            }
            for name, result in results.items()
        },
        observations=dict(
            related_trained_return_delta=trained_metrics["return_delta"]["mean"],
            related_initialized_return_delta=init_metrics["return_delta"]["mean"],
            related_trained_minus_initialized_return=trained_metrics["return_delta"]["mean"]
            - init_metrics["return_delta"]["mean"],
        ),
        claim_status="exploratory single-trained-expert result; not a multi-training-seed paper conclusion",
        selection_and_training_after_test="none; original checkpoints and default registry untouched",
        limitations=[
            "Different rollout lengths confound alive-only errors; main body comparison uses paired common alive prefix.",
            "All-frame return uses the full reference denominator; failures receive no later reward.",
            "Bootstrap samples source groups of clips, not independent policy training seeds.",
            "Top2 or positive contribution alone does not demonstrate useful transfer.",
            "Seven training controls includeCharleston dance; cross-source dance improvements do not establish all-domain transfer.",
            "Source metadata and joint signatures do not prove absence of every cross-dataset upstream mocap duplicate.",
        ],
    )
    save(REPORT / "report.json", report)
    lines = [
        "G1 冻结武术专家迁移：固定第2000轮、单训练种子探索评测",
        "四组：base / initialized / trained / trained_disabled。新测试32条，全部源动作全长。",
        "非正式论文结论；多初值不是多个训练种子。",
        "",
    ]
    for pool, item in summary.items():
        lines.append(pool + f" ({item['clips']} clips)")
        for name, m in item["conditions"].items():
            lines.append(
                f"  {name}: return={m['normalized_return']:.6f}, survival={m['full_clip_survival']:.2%}, admission={m['new_admission']:.2%}, contribution={m['new_contribution']:.2%}"
            )
        m = item["paired_vs_base"]["trained"]
        lines.append(
            f"  trained-base: return_delta={m['return_delta']['mean']:+.6f}, bootstrap95={m['return_delta']['source_group_bootstrap95']}, common_prefix_body_delta_m={m['body_common_prefix_delta_m']['mean']:+.6f}"
        )
        lines.append(
            f"  return improved/worsened clips: {m['clips_return_improved']}/{m['clips_return_worsened']}"
        )
        lines.append("")
    (REPORT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
