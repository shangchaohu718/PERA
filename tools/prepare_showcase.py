"""Import existing local resources for the first shared-world demonstration; no downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source = args.source_root.resolve()
    release = "releases/2026-10-14-demonstrations"
    assets = source / "third_party/mimicx_release/MimicX-Assets"
    policies = source / "third_party/mimicx_release/MimicX-Policies"
    manifest = json.loads((policies / release / "manifest.json").read_text())
    native = source / "reports/continuous_four_scene_v1/native_stairs_probe_02/scene-export"
    copies = [
        (p, ROOT / "data/showcase/native_g1" / p.relative_to(native), None)
        for p in native.rglob("*")
        if p.is_file()
    ]
    common = json.loads((ROOT / "resources/manifest.json").read_text())
    copies += [
        (source / x["source"], ROOT / x["destination"], x["sha256"])
        for x in common["files"]
        if x["destination"].startswith("assets/g1/meshes/")
    ]
    tasks = []
    names = ("descent",)
    for index, name in enumerate(names):
        row = next(v for v in manifest["policies"] if v["task"] == name)
        target = Path("data/showcase") / name
        copies.append(
            (assets / row["motion"]["path"], ROOT / target / "motion.npz", row["motion"]["sha256"])
        )
        scene = None
        if row.get("scene"):
            scene = str(target / "scene.json")
            copies.append((assets / row["scene"]["path"], ROOT / scene, row["scene"]["sha256"]))
        # The native G1 entity uses the SDK's MuJoCo tree order; preparation
        # records the explicit joint list instead of guessing from array shapes.
        tasks.append(
            dict(
                id="mimicx_" + name,
                expert="mimicx_" + name,
                label={
                    "descent": "Stair descent",
                    "bench": "Bench",
                    "dance": "Dance",
                    "football": "Football motion",
                    "tennis": "Tennis swing",
                }[name],
                motion=str(target / "motion.npz"),
                scene=scene,
                origin=[float(index % 3 * 8), float(index // 3 * 7), 0.0],
                horizon=row["horizon"],
                context_dim=0,
                scope="Original reference with its fixed collision fixtures; shared-world port, not new skill acceptance",
            )
        )
    records = []
    for src, dst, expected in copies:
        actual = digest(src)
        if expected is not None and expected != actual:
            raise ValueError(f"Source fingerprint differs: {src.name}")
        if dst.exists() and digest(dst) != actual:
            raise ValueError(f"Existing destination differs: {dst.relative_to(ROOT)}")
        if args.apply and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        records.append(
            dict(
                source=str(src.relative_to(source)),
                path=str(dst.relative_to(ROOT)),
                sha256=actual,
                bytes=src.stat().st_size,
            )
        )
    cfg = dict(
        schema="pera.shared_world.v1",
        checkpoint="checkpoints/composed/pera_30_experts.pt",
        robot_scene="data/showcase/native_g1/scene.xml",
        physics_dt=0.005,
        control_dt=0.02,
        tasks=tasks,
        joint_names=json.loads((ROOT / "configs/robot.json").read_text())["joint_names"],
        resource_manifest="resources/showcase_manifest.json",
        default_tasks=[task["id"] for task in tasks],
    )
    if args.apply:
        (ROOT / "configs/showcase_descent.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (ROOT / "resources/showcase_manifest.json").write_text(
            json.dumps(
                dict(
                    schema="pera.showcase_resources.v1",
                    stage="local_review_only_not_uploaded",
                    files=records,
                    note="Existing local task assets imported without modification; source and dataset terms still apply",
                ),
                indent=2,
            )
            + "\n"
        )
    print(
        json.dumps(
            dict(
                apply=args.apply,
                files=len(records),
                bytes=sum(v["bytes"] for v in records),
                prepared_tasks=[v["id"] for v in tasks],
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
