"""Hash-check and import separately provided resources; plan-only by default."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument(
        "--group", action="append", choices=["base", "continue128", "ppo", "task-extras"]
    )
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    rows = json.loads((ROOT / "resources/manifest.json").read_text())["files"]
    groups = a.group or ["base"]
    rows = [x for x in rows if x["group"] in groups]
    plan = []
    for x in rows:
        src = a.source_root / x["source"]
        dst = ROOT / x["destination"]
        if not dst.resolve().is_relative_to(ROOT):
            raise ValueError("Destination escapes release")
        state = (
            "already_matching"
            if dst.is_file() and digest(dst) == x["sha256"]
            else "available"
            if src.is_file()
            else "source_missing"
        )
        plan.append(dict(destination=x["destination"], state=state, bytes=x["bytes"]))
        if a.apply:
            if state == "already_matching":
                continue
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite {dst}")
            if not src.is_file():
                raise FileNotFoundError(src)
            if digest(src) != x["sha256"]:
                raise ValueError(f"Source hash differs: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".importing")
            shutil.copyfile(src, tmp)
            if digest(tmp) != x["sha256"]:
                tmp.unlink()
                raise ValueError("Copied hash mismatch")
            os.replace(tmp, dst)
    print(json.dumps({"applied": a.apply, "files": len(plan), "plan": plan}, indent=2))


if __name__ == "__main__":
    main()
