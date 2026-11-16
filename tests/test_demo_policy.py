"""The demo CLI must reject the archived base before starting a simulator."""

import sys

import pytest

from pera import __main__ as cli
from pera.paths import ROOT


def test_demo_rejects_self_trained_default_before_launch(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pera",
            "demo",
            "--config",
            str(ROOT / "configs/showcase_competitive_archived.json"),
            "--output",
            "runs/unused",
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A disallowed demonstration was launched")

    monkeypatch.setattr(cli.subprocess, "call", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "official HoloMotion" in capsys.readouterr().err
