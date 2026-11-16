"""Static package audit. No simulator, CUDA, network or model loading."""

import ast
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems = []
    media_manifest = ROOT / "provenance/README_MEDIA.json"
    allowed_videos = set()
    if media_manifest.is_file():
        for row in json.loads(media_manifest.read_text())["files"]:
            path = ROOT / row["path"]
            if path.parent != ROOT / "docs/media":
                problems.append("Unexpected README media path: " + row["path"])
                continue
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                problems.append("Missing or changed README media: " + row["path"])
            if path.suffix == ".mp4":
                allowed_videos.add(row["path"])
    base_training = ROOT / "src/blinkmimic/train.py"
    if base_training.exists():
        problems.append("Base-policy training is outside this release: src/blinkmimic/train.py")
    paths = list(ROOT.rglob("*.py"))
    code = []
    for p in paths:
        if any(x in p.parts for x in ["third_party", "vendor", "__pycache__", ".venv", "runs"]):
            continue
        text = p.read_text()
        code.append(p)
        try:
            ast.parse(text)
        except SyntaxError as e:
            problems.append(f"{p.relative_to(ROOT)}:{e.lineno}: {e.msg}")
        if re.search(r"""['"]/home/[A-Za-z0-9_.-]+/""", text):
            problems.append("User-specific path: " + str(p.relative_to(ROOT)))
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(
            x in p.parts
            for x in [
                "third_party",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "runs",
                "node_modules",
                "dist",
            ]
        )
    ]
    # Public source artifacts exclude explicitly ignored resources.
    forbidden = []
    for p in files:
        if (
            p.suffix in [".pt", ".pth", ".onnx", ".safetensors", ".mp4", ".zip"]
            and "checkpoints" not in p.parts
            and "resources" not in p.parts
            and p.relative_to(ROOT).parts[0] != "data"
            and str(p.relative_to(ROOT)) not in allowed_videos
            and not str(p.relative_to(ROOT)).startswith("website/public/media/")
        ):
            forbidden.append(str(p.relative_to(ROOT)))
    problems += ["Unexpected binary: " + p for p in forbidden]
    duplicates = collections.defaultdict(list)
    for p in code:
        if p.stat().st_size > 1500:
            duplicates[hashlib.sha256(p.read_bytes()).hexdigest()].append(str(p.relative_to(ROOT)))
    repeated = [v for v in duplicates.values() if len(v) > 1]
    problems += ["Duplicate source: " + str(v) for v in repeated]
    result = dict(
        python_files=len(code),
        base_training_included=base_training.exists(),
        duplicate_large_python_files=repeated,
        problems=problems,
        CUDA_initialized=False,
    )
    print(json.dumps(result, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
