"""Audit the local HoloMotion release inputs without network or simulator startup."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect(root, base_dir=None):
    policy = json.loads((root / "configs/demo_policy.json").read_text())
    weights = json.loads((root / "resources/holomotion_weights_manifest.json").read_text())
    config = json.loads((root / "configs/showcase.json").read_text())
    missing, changed = [], []
    expected = {}
    for row in weights["resources"]:
        expected[row["path"]] = row["sha256"]
    bank_path = root / "resources/holomotion_bank25.json"
    bank = json.loads(bank_path.read_text()) if bank_path.is_file() else None
    if bank is not None:
        if bank["contains_official_actor"]:
            raise ValueError("The release expert bank contains official Actor weights")
        expected[bank["path"]] = bank["sha256"]
    suite = json.loads((root / "configs/reproduction_suite15.json").read_text())
    manifests = set(config["resource_manifests"])
    for group in suite["groups"]:
        manifests.update(json.loads((root / group["config"]).read_text())["resource_manifests"])
    for name in sorted(manifests):
        for row in json.loads((root / name).read_text())["files"]:
            path = row.get("path", row.get("destination"))
            if path in expected and expected[path] != row["sha256"]:
                raise ValueError(f"Conflicting resource hashes: {path}")
            expected[path] = row["sha256"]
    verified_bytes = 0
    for name, expected_hash in expected.items():
        external_base = name.startswith("checkpoints/holomotion_v1.4/") and base_dir is not None
        path = (
            base_dir / Path(name).relative_to("checkpoints/holomotion_v1.4")
            if external_base
            else root / name
        ).resolve()
        if not external_base and not path.is_relative_to(root):
            raise ValueError(f"Resource path escapes project: {name}")
        if not path.is_file():
            missing.append(name)
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_hash:
            changed.append(name)
        else:
            verified_bytes += path.stat().st_size
    # Paths the current default demo launcher imports from its native root.
    # Their presence is necessary; this static audit cannot certify import closure.
    sources = json.loads((root / "resources/runtime_sources.json").read_text())
    runtime_paths = [row["path"] for project in sources["projects"] for row in project["files"]]
    runtime_paths.append("reproduction/experiments/ultra_box_v1/runtime.py")
    missing_runtime = [p for p in runtime_paths if not (root / p).is_file()]
    changed_runtime = []
    for project in sources["projects"]:
        for row in project["files"]:
            path = root / row["path"]
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                changed_runtime.append(row["path"])
    ready = not (missing or changed or missing_runtime or changed_runtime)
    return dict(
        status="static_inputs_present" if ready else "incomplete",
        base_family=policy["base_family"],
        pinned_expert_count=bank["expert_count"] if bank else len(policy["experts"]),
        demo_expert_count=len(config["tasks"]),
        unique_expected_resources=len(expected),
        verified_files=len(expected) - len(missing) - len(changed),
        verified_bytes=verified_bytes,
        missing_resources=missing,
        mismatched_resources=changed,
        missing_runtime_paths=missing_runtime,
        mismatched_runtime_paths=changed_runtime,
        standalone_inference_verified=False,
        clean_install_verified=False,
        training_reproduction_verified=False,
        public_downloads_verified=False,
        note="Static input audit only. A passing result does not establish reproduction.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--base-dir", type=Path, help="External official model directory")
    args = parser.parse_args()
    report = inspect(args.root.resolve(), args.base_dir.resolve() if args.base_dir else None)
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")
    return 0 if report["status"] == "static_inputs_present" else 2


if __name__ == "__main__":
    raise SystemExit(main())
