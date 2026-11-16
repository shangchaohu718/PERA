"""Verify local model files against the release manifest; no Torch or network required."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "resources/holomotion_weights_manifest.json"
    )
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("Weight manifest is absent")
    manifest = json.loads(args.manifest.read_text())
    missing, mismatched = [], []
    verified_bytes = 0
    for row in manifest["resources"]:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT):
            parser.error("Model path escapes the project directory")
        if not path.is_file():
            missing.append(row["path"])
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
            mismatched.append(row["path"])
        else:
            verified_bytes += row["bytes"]
    print(
        json.dumps(
            dict(
                status="verified" if not missing and not mismatched else "incomplete_or_changed",
                expected_files=len(manifest["resources"]),
                verified_files=len(manifest["resources"]) - len(missing) - len(mismatched),
                verified_bytes=verified_bytes,
                missing=missing,
                mismatched=mismatched,
            ),
            indent=2,
        )
    )
    return 2 if missing or mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
