"""A clean source checkout must not advertise or package retired workflows."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from modumotion_g1 import training_dependencies
from pera import __main__ as cli


def test_scene_archive_honors_declared_exclusions(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "scene_builder", Path(__file__).resolve().parents[1] / "tools/build_scene_release.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    content = {
        "configs/release.json": {
            "excluded_weight_manifests": ["resources/weights_manifest.json"],
            "excluded_source_paths": ["AGENTS.md", "docs/history"],
        },
        "provenance/README_MEDIA.json": {"files": []},
        "resources/holomotion_weights_manifest.json": {"resources": []},
        "resources/runtime_sources.json": {"projects": []},
        "configs/showcase.json": {"resource_manifests": []},
        "configs/reproduction_suite15.json": {"groups": []},
        "resources/weights_manifest.json": {"obsolete": True},
    }
    for name, value in content.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    (tmp_path / "docs/history").mkdir(parents=True)
    (tmp_path / "docs/history/old.md").write_text("archived")
    (tmp_path / "AGENTS.md").write_text("internal instructions")
    (tmp_path / "README.md").write_text("current instructions")
    files = builder.release_files(tmp_path)
    assert "README.md" in files
    assert "resources/holomotion_weights_manifest.json" in files
    assert "resources/weights_manifest.json" not in files
    assert "AGENTS.md" not in files
    assert not any(name.startswith("docs/history/") for name in files)


@pytest.mark.parametrize("command", ["reproduce", "compose"])
def test_retired_cli_cannot_launch_training(monkeypatch, command):
    monkeypatch.setattr(sys, "argv", ["pera", command])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_inference_subset_is_not_silently_used_as_a_teacher(monkeypatch, tmp_path):
    monkeypatch.setattr(training_dependencies, "ROOT", tmp_path)
    before = list(sys.path)
    with pytest.raises(RuntimeError, match="full pinned upstream sources"):
        training_dependencies.require_instinct_teacher_sources()
    assert sys.path == before


def test_optional_isaac_adapter_explains_external_dependency(monkeypatch):
    monkeypatch.setattr(training_dependencies.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(RuntimeError, match="externally installed Isaac Lab"):
        training_dependencies.require_isaaclab()
