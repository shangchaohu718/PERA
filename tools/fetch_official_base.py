"""Install pinned models directly from HoloMotion; never mirror or fetch by default."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def matches(path, row):
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="External model directory, outside the source release",
    )
    parser.add_argument(
        "--download", action="store_true", help="Explicitly download missing official files"
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        parser.error(
            "Use a model directory outside the release; official weights must not enter the upload package"
        )
    descriptor = json.loads((ROOT / "resources/holomotion_base.json").read_text())
    for row in descriptor["files"]:
        relative = Path(row["path"]).relative_to("checkpoints/holomotion_v1.4")
        destination = output / relative
        if matches(destination, row):
            print(f"Verified; reusing {relative}", flush=True)
            continue
        if destination.exists():
            raise ValueError(
                f"Existing file differs from the pinned model; preserve and inspect it: {destination}"
            )
        print(
            f"{'Download' if args.download else 'Missing'} {relative}: {row['bytes']} bytes from {row['upstream_url']}",
            flush=True,
        )
        if not args.download:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        with (
            urllib.request.urlopen(row["upstream_url"], timeout=60) as source,
            temporary.open("wb") as target,
        ):
            while chunk := source.read(4 * 1024 * 1024):
                target.write(chunk)
        if not matches(temporary, row):
            raise ValueError(f"Downloaded model failed SHA256/size verification: {temporary}")
        temporary.replace(destination)
    missing = [
        row["path"]
        for row in descriptor["files"]
        if not matches(output / Path(row["path"]).relative_to("checkpoints/holomotion_v1.4"), row)
    ]
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
