#!/usr/bin/env python3
"""Fetch and hash-check the pinned MimicX runtime without executing it."""

import concurrent.futures
import hashlib
import json
import shutil
import threading
import time

import requests

from pera.paths import ROOT

SPEC = json.loads((ROOT / "resources/mimicx_release.json").read_text())
DEST = ROOT / "third_party/mimicx_release"
LOCAL = threading.local()


def fetch(repo, revision, relative, expected=None):
    base = DEST / ("MimicX-Assets" if repo.startswith("datasets/") else "MimicX-Policies")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Artifact path escapes release directory")
    if path.is_file() and expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return path
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    error = None
    for attempt in range(3):
        try:
            response = LOCAL.session.get(
                f"https://huggingface.co/{repo}/resolve/{revision}/{relative}",
                timeout=30,
            )
            response.raise_for_status()
            content = response.content
            if expected and hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Checksum mismatch: " + relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".download")
            temporary.write_bytes(content)
            temporary.replace(path)
            return path
        except (requests.RequestException, ValueError) as exc:
            error = exc
            time.sleep(attempt + 1)
    raise error


def main():
    assets_repo = "datasets/" + SPEC["assets_repository"]
    policy_repo = SPEC["policies_repository"]
    assets_rev, policy_rev = SPEC["assets_revision"], SPEC["policies_revision"]
    release = SPEC["release"]
    manifest_path = fetch(
        policy_repo,
        policy_rev,
        release + "/manifest.json",
        SPEC["manifest_sha256"],
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["assets_revision"] == assets_rev
    row = next(x for x in manifest["policies"] if x["task"] == "stairs")
    policy = row["checkpoint"]
    target = DEST / "MimicX-Policies" / policy["path"]
    local = ROOT / "checkpoints/teachers/mimicx_stairs/model.pt"
    if not target.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == policy["sha256"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
    fetch(policy_repo, policy_rev, policy["path"], policy["sha256"])
    records = []
    indexes = {
        release + "/FILES.json": SPEC["assets_index_sha256"],
        **SPEC["dependency_indexes"],
    }
    for relative, expected in indexes.items():
        index = fetch(assets_repo, assets_rev, relative, expected)
        records.extend(json.loads(index.read_text()))
    unique = {record["path"]: record for record in records}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [
            pool.submit(fetch, assets_repo, assets_rev, relative, record["sha256"])
            for relative, record in unique.items()
        ]
        for i, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            future.result()
            if i % 50 == 0:
                print(f"Checked {i}/{len(jobs)} assets", flush=True)
    result = {
        "task": "stairs",
        "assets_revision": assets_rev,
        "policies_revision": policy_rev,
        "verified_assets": len(unique),
        "teacher": row,
        "runtime_executed": False,
    }
    (DEST / "DOWNLOAD_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"verified_assets": len(unique), "output": str(DEST)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
