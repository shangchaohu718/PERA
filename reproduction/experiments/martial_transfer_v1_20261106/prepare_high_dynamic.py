"""Select local, unseen, high-dynamic G1 references by semantics and kinematics only."""

import collections
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np

from experiments.martial_transfer_v1_20261106.prepare import (
    ROOT,
    BASE_CONTRACT,
    OLD_HOLDOUT,
    SOURCE,
    group,
    sha,
    save,
)


def category(family, description):
    text = description.lower()
    if family == "aistpp":
        return "fast_dance"
    if family != "cmu":
        return None
    # No platform, ledge, rope, object or multi-person interaction is available
    # in the flat reference-tracking scene. This pool focuses on free motion.
    if re.search(
        r"ledge|platform|rope|ball|throw|catch|grab|window|handstand|backflip|cartwheel|vignette|stumble|fall|subjects",
        text,
    ):
        return None
    if re.search(r"spin|twist|turn|180|360", text):
        return "turn_spin"
    if re.search(r"kick|punch|boxing", text):
        return "kick_strike"
    if re.search(r"jump|hop", text):
        return "jump_hop"
    return None


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    dest = a.output / "motions"
    dest.mkdir()
    source = json.loads(SOURCE.read_text())
    base = json.loads(BASE_CONTRACT.read_text())
    hold = json.loads(OLD_HOLDOUT.read_text())
    previous = json.loads((ROOT / "reports/martial_transfer_v1_20261106/protocol.json").read_text())
    lookup = {
        (r["metadata"]["source_family"], r["metadata"]["source_id"]): r["metadata"]
        for r in source["clips"].values()
    }
    blocked = {
        group(
            r["source_family"], r["source_id"], lookup.get((r["source_family"], r["source_id"]), r)
        )
        for r in base["clips"] + hold["clips"]
    }
    descriptions = json.loads((ROOT / "resources/motions/cmu_metadata.json").read_text())["clips"]
    order = [source["dof_names"].index(n) for n in base["sdk_dof_names"]]
    candidates = []
    rejections = collections.Counter()
    fingerprints = {}
    for key, row in source["clips"].items():
        md = row["metadata"]
        family = md["source_family"]
        sid = md["source_id"]
        desc = descriptions.get(sid, {}).get("description", "") if family == "cmu" else sid
        pool = category(family, desc)
        if pool is None or row["length"] < 100 or md.get("mirrored_variant"):
            continue
        g = group(family, sid, md)
        if g in blocked:
            continue
        shard = Path(source["hdf5_shards"][row["shard"]]["file"])
        s = row["start"]
        length = row["length"]
        with h5py.File(shard) as f:
            q = np.asarray(f["ref_dof_pos"][s : s + length], dtype=np.float32)[:, order]
            p0 = np.asarray(f["ref_root_pos"][s : s + length], dtype=np.float32)
            quat = np.asarray(f["ref_root_rot"][s : s + length], dtype=np.float32)
        if (
            not all(np.isfinite(x).all() for x in (q, p0, quat))
            or np.max(np.abs(np.linalg.norm(quat, axis=1) - 1)) > 0.01
        ):
            rejections["nonfinite_or_bad_quaternion"] += 1
            continue
        velocity = np.diff(q, axis=0) * 50
        p95 = float(np.quantile(np.abs(velocity), 0.95))
        p99 = float(np.quantile(np.abs(velocity), 0.99))
        # Reject gross angular discontinuities rather than selecting corrupt
        # retargets merely because their apparent speed is enormous.
        if p95 < 1.5 or p99 > 40 or np.max(np.abs(np.diff(q, axis=0))) > 1.0:
            rejections["below_speed_floor_or_gross_joint_jump"] += 1
            continue
        ts = np.linspace(0, len(q) - 1, 64)
        fingerprints[key] = np.stack(
            [np.interp(ts, np.arange(len(q)), q[:, j]) for j in range(29)], 1
        )
        candidates.append(
            dict(
                key=key,
                pool=pool,
                split_group=g,
                source_id=sid,
                source_family=family,
                description=desc,
                frames=length,
                shard=str(shard),
                start=s,
                joint_abs_speed_p95_rad_s=p95,
                joint_abs_speed_p99_rad_s=p99,
                root_vertical_speed_p95_m_s=float(
                    np.quantile(np.abs(np.diff(p0[:, 2]) * 50), 0.95)
                ),
            )
        )
    chosen = []
    selected_fingerprints = []
    duplicate_rejections = []
    for pool in ("kick_strike", "jump_hop", "turn_spin", "fast_dance"):
        ranked = sorted(
            (r for r in candidates if r["pool"] == pool),
            key=lambda r: (-r["joint_abs_speed_p95_rad_s"], r["key"]),
        )
        counts = collections.Counter()
        selected = []
        for row in ranked:
            # At most2 CMU clips from one subject, and1 AIST choreography group.
            limit = 1 if pool == "fast_dance" else 2
            if counts[row["split_group"]] >= limit:
                continue
            fp = fingerprints[row["key"]]
            if any(
                float(np.sqrt(np.mean((fp - prev) ** 2))) < 0.03 for prev in selected_fingerprints
            ):
                duplicate_rejections.append(row["key"])
                continue
            counts[row["split_group"]] += 1
            selected.append(row)
            selected_fingerprints.append(fp)
            if len(selected) == 8:
                break
        assert len(selected) == 8, (pool, len(selected), len(ranked))
        chosen.extend(selected)
    rows = []
    for row in chosen:
        s = row["start"]
        length = row["frames"]
        with h5py.File(row["shard"]) as f:
            p0, quat, q = [
                np.asarray(f[n][s : s + length], dtype=np.float32)
                for n in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
            ]
        name = row["source_family"] + "__" + row["source_id"]
        path = dest / (name + ".npz")
        np.savez_compressed(
            path,
            root_pos=p0,
            root_quat_wxyz=quat[:, [3, 0, 1, 2]],
            joint_pos=q[:, order],
            fps=np.float64(50),
        )
        rows.append(
            dict(row, name=name, file=str(path), sha256=sha(path), full_existing_source_clip=True)
        )
    previous_test = json.loads(
        (ROOT / "reports/martial_transfer_v1_20261106/data_contract.json").read_text()
    )
    earlier = {
        (r["source_family"], r["source_id"])
        for r in previous_test["rows"]
        if r["pool"].startswith("unseen")
    }
    protocol = dict(
        robot="unitree_g1_29dof",
        training=False,
        checkpoints={k: previous["checkpoints"][k] for k in ("base", "trained")},
        replicas_per_clip=8,
        rows=rows,
        selection_locked_before_policy_evaluation=True,
        selection_rule="4 semantic categories x8, descending reference joint-speed p95; at most2 clips per CMU subject,1 per AIST choreography; no policy scores",
        source_manifest=str(SOURCE),
        source_manifest_sha256=sha(SOURCE),
        candidate_counts=dict(collections.Counter(r["pool"] for r in candidates)),
        rejections=dict(rejections),
        removed_near_duplicate_candidate_keys=duplicate_rejections,
        source_groups_absent_from_base_and_prior16_holdouts=True,
        trained_expert_reference_source="seven PBHC clips only; these CMU/AIST references were not used by the expert",
        existing_previous32_test_overlap=sum(
            (r["source_family"], r["source_id"]) in earlier for r in rows
        ),
        not_new_retargeting=True,
        reference_speed_unchanged=True,
        no_test_based_checkpoint_selection=True,
        original_train_contract_sha256=sha(BASE_CONTRACT),
        limitations=[
            "Source metadata disjointness does not certify all upstream cross-dataset mocap duplication.",
            "Reference-motion speed does not certify retarget physical quality or that the robot executes the intended high-dynamic skill.",
            "Completion measures survival to reference end; tracking fidelity is a separate question.",
        ],
    )
    save(a.output / "protocol.json", protocol)
    save(
        a.output / "candidate_kinematics.json",
        dict(candidates=candidates, rejections=dict(rejections)),
    )
    print(
        json.dumps(
            dict(
                output=str(a.output),
                clips=len(rows),
                seconds=sum(r["frames"] for r in rows) / 50,
                candidate_counts=protocol["candidate_counts"],
                minimum_joint_speed_p95=min(r["joint_abs_speed_p95_rad_s"] for r in rows),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
