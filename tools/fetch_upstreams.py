"""Fetch exact source commits, never environments, Torch or data by default."""

import argparse, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--only", nargs="+")
    a = p.parse_args()
    rows = json.loads((ROOT / "resources/upstreams.json").read_text())["git_repositories"]
    if a.only:
        unknown = set(a.only) - {r["name"] for r in rows}
        if unknown:
            p.error("Unknown upstreams: " + ",".join(sorted(unknown)))
        rows = [r for r in rows if r["name"] in a.only]
    print(json.dumps(rows, indent=2))
    if not a.fetch:
        return
    for r in rows:
        dst = ROOT / r["destination"]
        if not dst.resolve().is_relative_to(ROOT):
            raise ValueError("Destination escapes release")
        if dst.exists():
            rev = subprocess.check_output(
                ["git", "-C", str(dst), "rev-parse", "HEAD"], text=True
            ).strip()
            if rev != r["revision"]:
                raise RuntimeError(
                    f"{dst} has another revision; preserve it and choose a fresh destination"
                )
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", r["url"], str(dst)], check=True)
        subprocess.run(["git", "-C", str(dst), "checkout", "--detach", r["revision"]], check=True)
        if r.get("patch"):
            subprocess.run(["git", "-C", str(dst), "apply", str(ROOT / r["patch"])], check=True)


if __name__ == "__main__":
    main()
