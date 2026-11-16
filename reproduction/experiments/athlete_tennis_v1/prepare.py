"""Fetch the pinned Quick Demo subset, reusing identical local robot meshes."""

import concurrent.futures
import fcntl
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from pera.paths import ROOT

MANIFEST = Path(__file__).with_name("source_manifest.json")
OUTPUT = ROOT / "reports/athlete_tennis_v1/preparation"
SOURCE = ROOT / "third_party/ATHLETE"
OUTPUT.mkdir(parents=True, exist_ok=True)
lock = (OUTPUT / "prepare.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
manifest = json.loads(MANIFEST.read_text())
started = time.monotonic()
state = {
    "status": "preparing",
    "repository": manifest["repository"],
    "commit": manifest["commit"],
    "kind": manifest["kind"],
    "files": [],
    "dependencies_installed": False,
    "environment_modified": False,
    "datasets_downloaded": False,
    "full_clone": False,
    "clone_note": "GitHub Git endpoint timed out; use exact Git-blob-verified files from raw.githubusercontent.com",
}
mesh_roots = [
    ROOT / p
    for p in (
        "HoloMotion/assets/robots/unitree/G1/meshes",
        "third_party/PMT/TCRS/assets/g1/meshes",
        "third_party/InstinctMJ/src/instinct_mj/assets/resources/unitree_g1/meshes",
        "third_party/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/assets",
    )
]


def blob_hash(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def save():
    state["elapsed_s"] = time.monotonic() - started
    p = OUTPUT / "report.tmp"
    p.write_text(json.dumps(state, indent=2) + "\n")
    p.replace(OUTPUT / "report.json")


def prepare(row):
    destination = SOURCE / row["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    reused = None
    url = f"https://raw.githubusercontent.com/HazinTihoa/athlete-humanoid-tennis/{manifest['commit']}/{row['path']}"
    if destination.exists():
        data = destination.read_bytes()
        assert blob_hash(data) == row["git_blob_sha"], f"Existing source differs: {destination}"
        reused = str(destination)
    else:
        for parent in mesh_roots if destination.suffix.lower() in (".stl", ".obj") else ():
            candidate = parent / destination.name
            if candidate.is_file():
                data = candidate.read_bytes()
                if blob_hash(data) == row["git_blob_sha"]:
                    destination.symlink_to(candidate)
                    reused = str(candidate)
                    break
        if reused is None:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=35) as response:
                        data = response.read()
                    assert len(data) == row["bytes"] and blob_hash(data) == row["git_blob_sha"], (
                        row["path"]
                    )
                    temporary = destination.with_name(destination.name + ".download")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (attempt + 1))
    return {**row, "url": url, "sha256": hashlib.sha256(data).hexdigest(), "reused_from": reused}


try:
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pending = [pool.submit(prepare, row) for row in manifest["files"]]
        for future in concurrent.futures.as_completed(pending):
            state["files"].append(future.result())
            save()
    published = json.loads((SOURCE / "deploy/policies/gpu3_2_model45000/manifest.json").read_text())
    for relative, row in published["files"].items():
        path = SOURCE / "deploy/policies/gpu3_2_model45000" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], relative
    state.update(
        status="completed",
        published_policy_checksums_passed=True,
        downloaded_bytes=sum(row["bytes"] for row in state["files"] if row["reused_from"] is None),
        reused_bytes=sum(row["bytes"] for row in state["files"] if row["reused_from"] is not None),
    )
except BaseException as error:
    state.update(status="failed", error=repr(error))
    raise
finally:
    save()
