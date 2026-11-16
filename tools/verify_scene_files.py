"""Verify every source/scene file declared by an extracted PERA scene archive."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root):
    manifest = json.loads((root / "SCENE_RELEASE_MANIFEST.json").read_text())
    missing, changed = [], []
    total = 0
    for row in manifest["files"]:
        name = row["path"]
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Archive file escapes checkout: {name}")
        if name.startswith("checkpoints/holomotion_v1.4/"):
            raise ValueError("Official base is bundled in this scene release")
        if not path.is_file():
            missing.append(name)
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
            changed.append(name)
        else:
            total += path.stat().st_size
    return dict(
        status="passed" if not (missing or changed) else "failed",
        files=len(manifest["files"]),
        verified_bytes=total,
        missing=missing,
        changed=changed,
        official_base_bundled=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
