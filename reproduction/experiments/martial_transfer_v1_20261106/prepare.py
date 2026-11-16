"""Lock a retrospective, fixed-checkpoint G1 transfer pilot before evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from pera.paths import ROOT

CODE = Path(__file__).resolve().parent
REPORT = ROOT / "reports/martial_transfer_v1_20261106"
SOURCE = ROOT / "reports/g1_curriculum_20261105/stage_3_complex/manifest.json"
BASE_CONTRACT = ROOT / "configs/data/train128.json"
OLD_HOLDOUT = BASE_CONTRACT.with_name("heldout16.json")
MARTIAL = ROOT / "reports/blink_competitive_martial_ppo_v1_20261106"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def group(family, sid, md):
    if family == "100style":
        return family + ":" + sid.split("_")[0]
    if family == "cmu":
        return family + ":subject_" + sid.split("_")[0]
    if family == "aistpp":
        parts = sid.split("_")
        return ":".join((family, parts[0], parts[-2], parts[-1]))
    if family == "bonus_seed":
        return family + ":" + md.get("action_label", sid).removesuffix("_M")
    return family + ":" + sid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE)
    parser.add_argument(
        "--cmu-metadata", type=Path, default=ROOT / "resources/motions/cmu_metadata.json"
    )
    args = parser.parse_args()
    source_path = args.source_manifest.resolve()
    if not source_path.is_file() or not args.cmu_metadata.is_file():
        parser.error("Transfer source manifest / CMU metadata absent; see docs/RESOURCES.md")
    assert not (REPORT / "protocol.json").exists(), "Never overwrite a locked experiment"
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "tmp").mkdir(exist_ok=True)
    dest = ROOT / "data/transfer"
    dest.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text())
    base = json.loads(BASE_CONTRACT.read_text())
    oldheld = json.loads(OLD_HOLDOUT.read_text())
    by_id = {
        (r["metadata"]["source_family"], r["metadata"]["source_id"]): (k, r)
        for k, r in source["clips"].items()
    }
    blocked = set()
    for row in base["clips"] + oldheld["clips"]:
        family, sid = row["source_family"], row["source_id"]
        md = by_id.get((family, sid), ("", {"metadata": row}))[1]["metadata"]
        blocked.add(group(family, sid, md))

    related = [
        ("cmu", sid)
        for sid in (
            "02_05",
            "111_19",
            "113_13",
            "135_04",
            "143_23",
            "143_24",
            "74_03",
            "75_16",
            "76_01",
            "87_01",
            "88_06",
            "90_05",
        )
    ]
    related += [("100style", sid) for sid in ("Kick_ID", "Kick_SR", "Punch_ID", "Punch_SR")]
    other = [
        ("100style", style + "_" + gait)
        for style in ("Neutral", "ArmsAboveHead", "GracefulArms", "Teapot")
        for gait in ("ID", "FW")
    ]
    # Selection uses only source identities, groups and lengths, never policy scores.
    candidates = sorted(
        (key for key in by_id if key[0] == "aistpp"),
        key=lambda x: hashlib.sha256((":".join(x)).encode()).hexdigest(),
    )
    chosen_groups = set()
    for family, sid in candidates:
        key, row = by_id[(family, sid)]
        g = group(family, sid, row["metadata"])
        if g in blocked or g in chosen_groups or row["length"] < 150:
            continue
        other.append((family, sid))
        chosen_groups.add(g)
        if len(other) == 16:
            break
    assert len(related) == len(other) == 16
    descriptions = json.loads(args.cmu_metadata.read_text())["clips"]
    rows = []
    for pool, items in [("unseen_related", related), ("unseen_other", other)]:
        for family, sid in items:
            key, row = by_id[(family, sid)]
            md = row["metadata"]
            g = group(family, sid, md)
            assert g not in blocked, (key, g)
            shard = Path(source["hdf5_shards"][row["shard"]]["file"])
            if not shard.is_absolute():
                shard = source_path.parent / shard
            start, length = row["start"], row["length"]
            with h5py.File(shard, "r") as f:
                p, quat, q = [
                    np.asarray(f[n][start : start + length], dtype=np.float32)
                    for n in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
                ]
            order = [source["dof_names"].index(n) for n in base["sdk_dof_names"]]
            q = q[:, order]
            assert all(np.isfinite(x).all() for x in (p, quat, q))
            assert np.max(np.abs(np.linalg.norm(quat, axis=1) - 1)) < 0.01
            path = dest / (family + "__" + sid + ".npz")
            assert not path.exists()
            np.savez_compressed(
                path,
                root_pos=p,
                root_quat_wxyz=quat[:, [3, 0, 1, 2]],
                joint_pos=q,
                fps=np.float64(50),
            )
            rows.append(
                dict(
                    name=path.stem,
                    pool=pool,
                    file=str(path),
                    sha256=sha(path),
                    source_family=family,
                    source_id=sid,
                    split_group=g,
                    frames=length,
                    source_start=start,
                    source_shard=str(shard),
                    source_manifest=str(source_path),
                    full_source_clip=True,
                    source_sha256=md.get("source_sha256"),
                    description=descriptions.get(sid, {}).get("description")
                    if family == "cmu"
                    else sid,
                    initial_cases=4,
                )
            )
    for row in base["clips"]:
        rows.append(
            dict(
                row,
                name=Path(row["file"]).stem,
                pool="old_training_retention",
                split_group="old:" + row["source_id"],
                initial_cases=1,
            )
        )

    # Training-motion controls use the exact already-normalized goals saved by
    # the original experiment. Reuse its tensor library to avoid another transform.
    # If no serialized motion library exists, prepare controls from the original
    # HDF5 and verify the same XY/yaw transform in the evaluation preflight.
    martial_manifest = MARTIAL / "references_h5/manifest.json"
    if not martial_manifest.exists():
        martial_manifest = ROOT / "reports/kungfubot_original_20261105/references_h5/manifest.json"
    manifest = json.loads(martial_manifest.read_text())
    for name, row in manifest["clips"].items():
        shard = Path(manifest["hdf5_shards"][row["shard"]]["file"])
        if not shard.is_absolute():
            shard = martial_manifest.parent / shard
        start, length = row.get("start", 0), row["length"]
        with h5py.File(shard, "r") as f:
            p, quat, q = [
                np.asarray(f[n][start : start + length], dtype=np.float32)
                for n in ("ref_root_pos", "ref_root_rot", "ref_dof_pos")
            ]
        assert row["metadata"]["dof_names"] == base["sdk_dof_names"]
        x, y, z, w = quat[0]
        yaw = np.arctan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)
        p[:, :2] -= p[0, :2].copy()
        cs, sn = np.cos(yaw), np.sin(yaw)
        px, py = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0], p[:, 1] = cs * px + sn * py, -sn * px + cs * py
        c, s = np.cos(-yaw / 2), np.sin(-yaw / 2)
        x, y, z, w = quat.T.copy()
        quat = np.stack((c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w), -1)
        path = dest / ("pbhc__" + name + ".npz")
        assert not path.exists()
        np.savez_compressed(
            path,
            root_pos=p,
            root_quat_wxyz=quat.astype(np.float32),
            joint_pos=q,
            fps=np.float64(50),
        )
        rows.append(
            dict(
                name=path.stem,
                pool="expert_training_control",
                file=str(path),
                sha256=sha(path),
                frames=length,
                split_group="pbhc:" + name,
                source_family="pbhc",
                source_id=name,
                initial_cases=4,
                full_source_clip=True,
                source_shard=str(shard),
                source_shard_sha256=sha(shard),
                transform="center initialXY, remove initialyaw; joint order unchanged",
            )
        )

    # Diagnostic of near-duplicate *whole-clip* joint trajectories. This does not
    # certify no shared upstream mocap across differently retargeted collections.
    def fingerprint(path):
        with np.load(path) as d:
            q = d["joint_pos"]
        t = np.linspace(0, len(q) - 1, 64)
        return np.stack([np.interp(t, np.arange(len(q)), q[:, j]) for j in range(29)], 1)

    trained = [
        r for r in rows if r["pool"] in ("old_training_retention", "expert_training_control")
    ]
    seen_fp = np.stack([fingerprint(r["file"]) for r in trained])
    near = []
    for row in rows[:32]:
        fp = fingerprint(row["file"])
        distance = np.sqrt(np.mean((seen_fp - fp) ** 2, axis=(1, 2)))
        i = int(distance.argmin())
        near.append(
            dict(
                test=row["name"],
                nearest_training=trained[i]["name"],
                joint_trajectory_rmse_rad=float(distance[i]),
            )
        )
        assert distance[i] > 0.02, ("Potential whole-clip duplicate", near[-1])
    data = dict(
        robot="unitree_g1_29dof",
        fps=50,
        joint_names=base["sdk_dof_names"],
        rows=rows,
        new_unseen_clips=32,
        old_retention_clips=128,
        training_controls=7,
        data_selection="source metadata only, fixed full clips; no policy-score filtering",
        source_manifest_sha256=sha(source_path),
        original_train_contract_sha256=sha(BASE_CONTRACT),
        previous16_holdouts_excluded=True,
        source_group_disjoint_from_pretraining=True,
        nearest_whole_clip_joint_signature=near,
        overlap_limitations="Source-group disjointness plus whole-clip signature diagnostic; does not certify all cross-dataset upstream human-motion provenance.",
    )
    save(REPORT / "data_contract.json", data)
    paths = dict(
        base=ROOT / "checkpoints/base/best.pt",
        initialized=MARTIAL / "training/initialized.pt",
        trained=MARTIAL / "training_resume_1000_to_2000/expert_002000.pt",
    )
    protocol = dict(
        version="martial_transfer_fixed2000_v1",
        robot="unitree_g1_29dof",
        stage="locked_before_evaluation",
        study="retrospective fixed-checkpoint pilot",
        independently_trained_experts=1,
        base_iteration=11000,
        expert_ppo_rounds=2000,
        expert_actor_optimizer_updates=48000,
        world_model_during_expert_training=False,
        routing="frozen8 + trained1, true unifiedTop2, frozen sharedFFN",
        conditions=["base", "initialized", "trained", "trained_disabled"],
        checkpoints={k: dict(file=str(p), sha256=sha(p)) for k, p in paths.items()},
        data_contract_sha256=sha(REPORT / "data_contract.json"),
        simulator="native G1 MuJoCo-Warp, same simulator and observation interface for all conditions",
        initial_cases=dict(
            case0="exact reference pose/velocity, phase0",
            cases1to3="fixed paired small perturbations, seeds41001,41002,41003; NOT training seeds",
            root_xy_uniform_m=0.005,
            joint_position_uniform_rad=0.01,
            joint_velocity_uniform_rad_s=0.02,
            root_velocity_uniform_m_s=0.01,
        ),
        selection="fixed last2000 expert before any new test; no test-based checkpoint selection",
        primary_metrics=[
            "clip-balanced normalized return",
            "full-clip survival",
            "body-anchor-aligned RMS error on common alive prefix versus base",
        ],
        secondary_metrics=[
            "root XY error",
            "joint RMS error",
            "new expert admission and contribution",
            "failure time",
            "phase-matched foot-height error",
        ],
        uncertainty="paired bootstrap over source groups; exploratory single training seed, no training-seed confidence claim",
        decision="report all improvements and regressions; no automatic acceptance or promotion",
        limitations=[
            "Existing expert saw all sevenPBHC motions includingCharleston dance.",
            "Existing2000 expert failed prior old-skill retention gate; no expectation of guaranteed positive transfer.",
            "No multiple-expert continual sequence or real-robot experiment.",
            "Pretraining/native and expert-training/Isaac differ; all transfer comparisons here are within one native simulator.",
        ],
    )
    save(REPORT / "protocol.json", protocol)
    print(
        json.dumps(
            dict(
                report=str(REPORT),
                clips=len(rows),
                eval_envs=sum(r["initial_cases"] for r in rows),
                unseen_seconds=sum(r["frames"] / 50 for r in rows[:32]),
                minimum_signature_rmse=min(r["joint_trajectory_rmse_rad"] for r in near),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
