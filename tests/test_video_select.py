"""Tests for batch video discovery + selection (anytrack-run/-label --video DIR)."""
from __future__ import annotations

from anytrack.video_select import discover_videos, resolve_videos, _text_select


def test_discover_videos_dir_and_file(tmp_path):
    (tmp_path / "b.avi").write_bytes(b"")
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "c.MOV").write_bytes(b"")
    found = discover_videos(tmp_path)
    assert [p.name for p in found] == ["a.mp4", "b.avi", "c.MOV"]     # sorted, video exts only
    f = tmp_path / "a.mp4"
    assert discover_videos(f) == [f]                                  # single file → itself


def test_text_select_parsing(monkeypatch):
    labels = [f"v{i}" for i in range(6)]
    monkeypatch.setattr("builtins.input", lambda *_: "1,3-5")
    assert _text_select(labels) == [0, 2, 3, 4]                       # 1 and range 3-5
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert _text_select(labels) == [0, 1, 2, 3, 4, 5]                 # blank → all


def test_resolve_single_file_passthrough(tmp_path):
    f = tmp_path / "only.avi"
    f.write_bytes(b"")
    # a single file path is returned as-is (no validation/prompt — caller validates)
    assert resolve_videos(f, lambda v: (False, "should not be called")) == [f]


def test_resolve_dir_validator_filters_and_confirms(tmp_path, monkeypatch):
    for n in ("keep1.avi", "keep2.avi", "drop.avi"):
        (tmp_path / n).write_bytes(b"")
    val = lambda v: (not v.name.startswith("drop"), "ok" if not v.name.startswith("drop") else "bad")
    monkeypatch.setattr("builtins.input", lambda *_: "y")             # process all
    chosen = resolve_videos(tmp_path, val)
    assert [p.name for p in chosen] == ["keep1.avi", "keep2.avi"]     # invalid dropped
