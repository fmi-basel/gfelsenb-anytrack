"""Tests for the `anytrack` subcommand dispatcher (anytrack.dispatch)."""
from __future__ import annotations

import sys
import types

import pytest

import anytrack.dispatch as dispatch


@pytest.fixture(autouse=True)
def _preserve_argv():
    """The dispatcher rewrites sys.argv; restore it so tests don't leak state."""
    saved = sys.argv[:]
    yield
    sys.argv = saved


def test_no_args_prints_usage(capsys):
    assert dispatch.main([]) == 0
    out = capsys.readouterr().out
    assert "anytrack <command>" in out and "anytrack-run" in out


def test_help_prints_usage(capsys):
    for flag in ("-h", "--help", "help"):
        assert dispatch.main([flag]) == 0
        assert "commands:" in capsys.readouterr().out


def test_unknown_command_errors(capsys):
    assert dispatch.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().out


def test_routes_to_run_with_rewritten_argv(monkeypatch):
    captured = {}

    def fake_run():
        captured["argv"] = list(sys.argv)
        return 0

    monkeypatch.setattr("anytrack.run.main", fake_run)
    rc = dispatch.main(["run", "--video", "x.avi", "--dry-run"])
    assert rc == 0
    # subcommand token dropped; argv[0] looks like the standalone script.
    assert captured["argv"] == ["anytrack-run", "--video", "x.avi", "--dry-run"]


def test_routes_to_validate_and_propagates_exit_code(monkeypatch):
    captured = {}

    def fake_validate():
        captured["argv"] = list(sys.argv)
        return 3

    monkeypatch.setattr("anytrack.validate.main", fake_validate)
    rc = dispatch.main(["validate", "--video", "d"])
    assert rc == 3
    assert captured["argv"] == ["anytrack-validate", "--video", "d"]


def test_none_return_is_coerced_to_zero(monkeypatch):
    """GUI-style targets return None; the dispatcher must yield a 0 exit code."""
    fake = types.ModuleType("fake_gui_mod")
    hits = {}

    def _gui():          # like the real gui_main: launches the app, returns None
        hits["hit"] = True

    fake.gui_main = _gui
    monkeypatch.setitem(sys.modules, "fake_gui_mod", fake)
    monkeypatch.setitem(dispatch.SUBCOMMANDS, "gui", "fake_gui_mod:gui_main")

    assert dispatch.main(["gui"]) == 0
    assert hits.get("hit") is True


def test_subcommands_mirror_console_scripts():
    """Every subcommand maps to a 'module:function' string (kept ↔ pyproject)."""
    for name, target in dispatch.SUBCOMMANDS.items():
        assert target.count(":") == 1 and target.startswith("anytrack.")
    # The dispatcher must include the commands the task calls out explicitly.
    for expected in ("run", "validate", "gui"):
        assert expected in dispatch.SUBCOMMANDS
    assert dispatch.SUBCOMMANDS["gui"] == "anytrack.cli:gui_main"
