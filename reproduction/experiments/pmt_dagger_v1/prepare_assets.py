"""Fetch only the pinned official assets needed for PMT big-map preflight."""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY = "aCodeDog/PMT-assets"


def read_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPOSITORY}"
    manifest_path = root / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = previous.get("revision") or read_json(api)["sha"]
    rows = read_json(f"{api}/tree/{revision}?recursive=true&limit=1000")
    files = {row["path"]: row for row in rows if row["type"] == "file"}
    wanted = [
        "checkpoints/pretrained/walkdance_bigmap_teacher.pt",
        "checkpoints/pretrained/ppoft_vision_bigmap_walkdance.pt",
        "assets/terrain/g1_29dof_big_map.stl",
        "SHA256SUMS.txt",
    ]
    raw = sorted(name for name in files if "/raw/" in name)
    for name in raw[: args.pairs]:
        wanted.extend(
            [
                name,
                name.replace("/raw/", "/optimized/").replace("_raw.npz", "_optimized.npz"),
            ]
        )
    report = {
        "repository": REPOSITORY,
        "revision": revision,
        "pairs": args.pairs,
        "status": "downloading",
        "files": [],
        "total_bytes": sum(files[name]["size"] for name in wanted),
        "downloads_torch_cuda_or_environments": False,
    }

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(manifest_path)

    save()
    for name in wanted:
        row = files[name]
        expected = row.get("lfs", {}).get("oid")
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        valid = target.is_file() and target.stat().st_size == row["size"]
        if valid and expected:
            valid = digest(target) == expected
        if not valid:
            url = (
                f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
                f"{revision}/{urllib.parse.quote(name, safe='/')}"
            )
            temporary = target.with_suffix(target.suffix + ".part")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        with temporary.open("wb") as stream:
                            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                                stream.write(chunk)
                    if temporary.stat().st_size != row["size"]:
                        raise ValueError(f"Size mismatch: {name}")
                    if expected and digest(temporary) != expected:
                        raise ValueError(f"SHA256 mismatch: {name}")
                    temporary.replace(target)
                    break
                except Exception:
                    if attempt == 2:
                        report["status"] = "download_failed"
                        report["failed_file"] = name
                        save()
                        raise
                    time.sleep(2)
        report["files"].append(
            {
                "path": name,
                "size": target.stat().st_size,
                "sha256": digest(target),
                "reused": valid,
            }
        )
        save()
        print(f"Verified {name}", flush=True)
    checksums = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        value, name = line.split(maxsplit=1)
        checksums[name.lstrip("*").removeprefix("./")] = value
    for row in report["files"]:
        if row["path"] != "SHA256SUMS.txt":
            if checksums[row["path"]] != row["sha256"]:
                raise ValueError(f"Published checksum mismatch: {row['path']}")
    report["status"] = "verified"
    save()


if __name__ == "__main__":
    main()
