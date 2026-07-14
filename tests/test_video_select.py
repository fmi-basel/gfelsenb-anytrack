"""Tests for batch video discovery + selection (anytrack-run/-label --video DIR)."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from anytrack.video_select import (UNINDEXED_REASON, discover_videos,
                                   resolve_videos, validate_openable, _text_select)


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


def test_resolve_dir_prints_remux_hint_for_unindexed(tmp_path, monkeypatch, capsys):
    (tmp_path / "good.avi").write_bytes(b"")
    (tmp_path / "broken.avi").write_bytes(b"")
    val = lambda v: (True, "5 frames") if v.name == "good.avi" else (False, UNINDEXED_REASON)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    chosen = resolve_videos(tmp_path, val)
    assert [p.name for p in chosen] == ["good.avi"]
    out = capsys.readouterr().out
    assert f"ffmpeg -i {tmp_path / 'broken.avi'} -c copy" in out       # full-path fix command
    assert "good.avi -c copy" not in out                               # only broken files listed


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_validate_openable_detects_unfinalized_avi(tmp_path):
    """An interrupted recording (no idx1 index, zeroed RIFF/frame counts) still
    decodes frame-by-frame but reports 0 frames — it must be flagged for remux,
    not conflated with a genuinely empty file."""
    good = tmp_path / "good.avi"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=size=64x64:rate=10:duration=1",
                    "-c:v", "mpeg4", str(good)], check=True)
    ok, reason = validate_openable(good)
    assert ok and reason == "10 frames"

    # Emulate the unfinalized file: drop the trailing idx1 index chunk and zero
    # every header field a finalizer would have written (RIFF size, avih/dmlh
    # total-frame counts, strh dwLength) — what a killed recorder leaves behind.
    data = bytearray(good.read_bytes())
    idx1 = data.rfind(b"idx1")
    avih = data.find(b"avih")
    strh = data.find(b"strh")
    assert idx1 > 0 and avih > 0 and strh > 0
    data = data[:idx1]
    zero = b"\x00\x00\x00\x00"
    data[4:8] = zero                                    # RIFF size
    data[avih + 8 + 16:avih + 8 + 20] = zero            # avih dwTotalFrames
    data[strh + 8 + 32:strh + 8 + 36] = zero            # strh dwLength
    dmlh = data.find(b"dmlh")
    if dmlh > 0:
        data[dmlh + 8:dmlh + 12] = zero                 # ODML dwTotalFrames
    broken = tmp_path / "broken.avi"
    broken.write_bytes(bytes(data))

    ok, reason = validate_openable(broken)
    assert not ok and reason == UNINDEXED_REASON
