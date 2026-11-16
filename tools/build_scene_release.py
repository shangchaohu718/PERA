"""Build a verified local scene-replay archive, excluding official base weights."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PREFIX = "checkpoints/holomotion_v1.4/"


def release_files(root):
    selected = set()
    selection = json.loads((root / "configs/release.json").read_text())
    excluded = tuple(
        selection.get("excluded_source_paths", []) + selection.get("excluded_weight_manifests", [])
    )

    def excluded_source(name):
        return any(
            name == prefix or name.startswith(prefix.rstrip("/") + "/") for prefix in excluded
        )

    allowed_media = {
        row["path"]
        for row in json.loads((root / "provenance/README_MEDIA.json").read_text())["files"]
    }
    for name in [
        "src",
        "reproduction",
        "integrations/HoloMotion",
        "configs",
        "resources",
        "tools",
        "scripts",
        "tests",
        "docs",
        "licenses",
        "requirements",
        "provenance",
        "assets",
    ]:
        for path in (root / name).rglob("*"):
            if excluded_source(path.relative_to(root).as_posix()):
                continue
            if not path.is_file() or any(
                p in {"__pycache__", ".pytest_cache", ".ruff_cache"}
                for p in path.relative_to(root).parts
            ):
                continue
            if any(p.endswith(".egg-info") for p in path.relative_to(root).parts):
                continue
            if (
                path.suffix in {".pyc", ".pt", ".pth", ".safetensors", ".onnx", ".mp4", ".zip"}
                and path.relative_to(root).as_posix() not in allowed_media
            ):
                continue
            selected.add(path.relative_to(root).as_posix())
    for name in [
        "README.md",
        "README.zh-CN.md",
        "pyproject.toml",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "CITATION.cff",
        ".gitignore",
        ".gitattributes",
    ]:
        if (root / name).is_file():
            selected.add(name)
    expected = {}
    weights = json.loads((root / "resources/holomotion_weights_manifest.json").read_text())
    for row in weights["resources"]:
        if not row["path"].startswith(OFFICIAL_PREFIX):
            expected[row["path"]] = row["sha256"]
    bank_path = root / "resources/holomotion_bank25.json"
    if bank_path.is_file():
        bank = json.loads(bank_path.read_text())
        if bank["contains_official_actor"]:
            raise ValueError("The distributed expert bank must exclude official Actor weights")
        expected[bank["path"]] = bank["sha256"]
    config = json.loads((root / "configs/showcase.json").read_text())
    suite = json.loads((root / "configs/reproduction_suite15.json").read_text())
    manifests = set(config["resource_manifests"])
    for group in suite["groups"]:
        manifests.update(json.loads((root / group["config"]).read_text())["resource_manifests"])
    for manifest in sorted(manifests):
        for row in json.loads((root / manifest).read_text())["files"]:
            key = row["path"]
            if key in expected and expected[key] != row["sha256"]:
                raise ValueError(f"Conflicting asset hashes: {key}")
            expected[key] = row["sha256"]
    runtime = json.loads((root / "resources/runtime_sources.json").read_text())
    for project in runtime["projects"]:
        for row in project["files"]:
            expected[row["path"]] = row["sha256"]
    for name, expected_hash in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Resource escapes checkout: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_hash:
            raise ValueError(f"Missing or changed asset: {name}")
        selected.add(name)
    for name in selected:
        if excluded_source(name):
            raise ValueError(f"Excluded source included: {name}")
        if (
            name.startswith(OFFICIAL_PREFIX)
            or name.startswith("checkpoints/base/")
            or "pera_30_experts" in name
        ):
            raise ValueError(f"Excluded base included: {name}")
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = release_files(ROOT)
    if args.output.exists():
        parser.error("Output exists; use a new name to preserve earlier packages")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    payloads = {}
    stored_payloads = 0
    with tarfile.open(args.output, "w:gz", dereference=True) as archive:
        for name in files:
            data = (ROOT / name).read_bytes()
            item = tarfile.TarInfo("PERA/" + name)
            digest = hashlib.sha256(data).hexdigest()
            item.mode = 0o644
            # Only immutable binary resources share archive payloads; editable
            # source, configuration and license files remain independent files.
            deduplicate = Path(name).suffix.lower() in {
                ".npz",
                ".pt",
                ".onnx",
                ".stl",
                ".obj",
                ".png",
                ".jpg",
                ".jpeg",
            }
            if deduplicate and digest in payloads:
                item.type = tarfile.LNKTYPE
                item.linkname = payloads[digest]
                archive.addfile(item)
            else:
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
                stored_payloads += 1
                if deduplicate:
                    payloads[digest] = item.name
            entries.append(dict(path=name, bytes=len(data), sha256=digest))
        payload = (
            json.dumps(
                {
                    "schema": "pera.scene_archive.v1",
                    "files": entries,
                    "official_base_bundled": False,
                },
                indent=2,
            )
            + "\n"
        ).encode()
        info = tarfile.TarInfo("PERA/SCENE_RELEASE_MANIFEST.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(
        json.dumps(
            dict(
                files=len(files),
                stored_payload_files=stored_payloads,
                bytes=args.output.stat().st_size,
                sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                official_base_bundled=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
