"""One immutable official motion/terrain pair per independent scene expert."""

import hashlib
import json
import math
import shutil

from pera.paths import ROOT

DATASET = ROOT / "data/instinct_supermimic/20251116_50cm_kneeClimbStep1"
DEFAULT_TASK = "knee_climb_no_wall"

# Follow the released metadata.yaml, excluding unregistered archive extras.
TASKS = (
    ("diveroll_lab_no_wall", "无墙实验室前滚翻"),
    ("diveroll_ramp_no_wall", "无墙斜坡前滚翻"),
    ("knee_climb_wall", "有墙 50cm 台阶攀爬"),
    ("jump_sit_wall", "有墙台阶跳坐"),
    ("roll_vault_wall", "有墙台阶翻越"),
    ("diveroll_lab_wall", "有墙实验室前滚翻"),
    (DEFAULT_TASK, "无墙 50cm 台阶攀爬"),
    ("jump_sit_no_wall", "无墙台阶跳坐"),
    ("roll_vault_no_wall", "无墙台阶翻越"),
    ("diveroll_ramp_wall", "有墙斜坡前滚翻"),
)


def teacher_quality_acceptable(evaluation, allow_partial=False):
    """Keep real failure rates; the override never admits zero successes."""
    episodes = evaluation.get("episodes", 0)
    successes = evaluation.get("successful_episodes", 0)
    rate = evaluation.get("success_rate", float("nan"))
    return (
        isinstance(episodes, int)
        and isinstance(successes, int)
        and episodes >= 8
        and 1 <= successes <= episodes
        and math.isfinite(rate)
        and abs(rate - successes / episodes) < 1e-6
        and (allow_partial or rate >= 0.9)
    )


def get_task(task_id):
    for index, (name, label) in enumerate(TASKS):
        if task_id == name:
            return {
                "id": name,
                "label": label,
                "metadata_index": index,
                "domain": "instinct_climb" if name == DEFAULT_TASK else "instinct_" + name,
                "scene": "official_50cm_knee_climb_no_wall"
                if name == DEFAULT_TASK
                else "official_" + name,
            }
    raise ValueError(f"Unknown official task: {task_id}")


def official_pairs(dataset=DATASET):
    import yaml

    metadata = yaml.safe_load((dataset / "metadata.yaml").read_text())
    if len(metadata["motion_files"]) != len(TASKS):
        raise ValueError("Released task catalogue changed")
    terrains = {t["terrain_id"]: t for t in metadata["terrains"]}
    pairs = []
    for index, row in enumerate(metadata["motion_files"]):
        terrain = terrains[row["terrain_id"]]
        pair = {
            **get_task(TASKS[index][0]),
            "motion_file": row["motion_file"],
            "terrain_file": terrain["terrain_file"],
            "original_terrain_id": row["terrain_id"],
        }
        for key in ("motion_file", "terrain_file"):
            path = (dataset / pair[key]).resolve()
            if not path.is_relative_to(dataset.resolve()) or not path.is_file():
                raise ValueError(f"Missing/unsafe official asset: {path}")
            pair[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        pairs.append(pair)
    return pairs


def prepare_task_dataset(task_id, dataset=DATASET, destination=None):
    """Select one official pair; copy bytes and remap only its terrain ID."""
    import yaml

    get_task(task_id)
    task = next(p for p in official_pairs(dataset) if p["id"] == task_id)
    destination = destination or ROOT / "data/instinct_task_dagger_v1" / task_id
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for key in ("motion_file", "terrain_file"):
        name = task[key]
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = task[key + "_sha256"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Official asset copy changed: {target}")
        else:
            shutil.copy2(dataset / name, target)
        hashes[name] = expected
    metadata = {
        "motion_files": [{"motion_file": task["motion_file"], "terrain_id": 0}],
        "terrains": [{"terrain_file": task["terrain_file"], "terrain_id": 0}],
    }
    manifest = {
        "source": str(dataset),
        "release": "InstinctMJ_supermimic_v0.1",
        "task": task,
        "source_metadata_sha256": hashlib.sha256(
            (dataset / "metadata.yaml").read_bytes()
        ).hexdigest(),
        "files_sha256": hashes,
        "motion_and_mesh_modified": False,
        "uses_pmt_tcrs": False,
        "student_reference": "official_teacher_task_reference",
    }
    for name, text in [
        ("metadata.yaml", yaml.safe_dump(metadata, sort_keys=False)),
        ("manifest.json", json.dumps(manifest, indent=2) + "\n"),
    ]:
        target = destination / name
        if target.exists() and target.read_text() != text:
            raise ValueError(f"Task contract changed: {target}")
        target.write_text(text)
    return destination
