"""Tests for anytrack-validate (preflight input validation)."""
from __future__ import annotations

import json

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.validate import validate_input, FileReport, PASS, FAIL, WARN
import anytrack.validate as vmod


def _write_video(path, n=20, h=64, w=64):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for _ in range(n):
        wr.write(np.full((h, w, 3), 200, np.uint8))
    wr.release()
    return True


def _write_timing(path, n):
    pd.DataFrame({"frame": np.arange(n), "dt_s": [1.0 / 30.0] * n}).to_csv(path, index=False)


def _status(rep: FileReport, name: str) -> str:
    return next(c.status for c in rep.checks if c.name == name)


def _make_valid(tmp_path):
    """A video + matching timing; returns (video_path, reported_frame_count)."""
    video = tmp_path / "clip.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    from anytrack.io import probe_video
    _, frames, _, _ = probe_video(video)
    _write_timing(video.with_suffix(".csv"), n=frames)
    return video, frames


def test_missing_video_fails_without_encoder(tmp_path):
    """Encoder-free: a nonexistent video fails the existence check cleanly."""
    rep = validate_input(tmp_path / "nope.avi")
    assert _status(rep, "video file exists") == FAIL
    assert rep.failed and not rep.is_ok()


def test_valid_input_all_pass(tmp_path):
    video, frames = _make_valid(tmp_path)
    rep = validate_input(video)
    assert rep.is_ok() and not rep.failed and not rep.warned
    assert _status(rep, "video opens") == PASS
    assert _status(rep, "first frame decodes") == PASS
    assert _status(rep, "timing CSV parses") == PASS
    assert _status(rep, "frame/timing match") == PASS


def test_missing_timing_fails(tmp_path):
    video = tmp_path / "clip.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    rep = validate_input(video)                 # no <stem>.csv written
    assert _status(rep, "timing CSV exists") == FAIL
    assert rep.failed and not rep.is_ok()


def test_unparseable_timing_fails(tmp_path):
    video, frames = _make_valid(tmp_path)
    # Overwrite the timing CSV with one that lacks the required columns.
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(video.with_suffix(".csv"), index=False)
    rep = validate_input(video)
    assert _status(rep, "timing CSV parses") == FAIL
    assert not rep.is_ok()


def test_frame_timing_mismatch_is_warning(tmp_path):
    """A row-count mismatch warns (usable) but fails under strict."""
    video, frames = _make_valid(tmp_path)
    _write_timing(video.with_suffix(".csv"), n=frames + 5)   # deliberate mismatch
    rep = validate_input(video)
    assert _status(rep, "frame/timing match") == WARN
    assert rep.warned and not rep.failed
    assert rep.is_ok(strict=False)          # usable by default
    assert not rep.is_ok(strict=True)       # fatal under --strict


def test_deep_decodes_last_frame(tmp_path):
    video, frames = _make_valid(tmp_path)
    rep = validate_input(video, deep=True)
    assert _status(rep, "last frame decodes") == PASS


def test_main_returns_zero_on_valid_and_one_on_invalid(tmp_path, capsys):
    video, frames = _make_valid(tmp_path)
    assert vmod.main(["--video", str(video)]) == 0

    # Remove the timing CSV -> the run must fail.
    video.with_suffix(".csv").unlink()
    assert vmod.main(["--video", str(video)]) == 1
    out = capsys.readouterr().out
    assert "timing CSV" in out and "0/1" in out


def test_main_strict_flag_fails_on_warning(tmp_path):
    video, frames = _make_valid(tmp_path)
    _write_timing(video.with_suffix(".csv"), n=frames + 5)   # mismatch -> warning
    assert vmod.main(["--video", str(video)]) == 0           # warning is non-fatal
    assert vmod.main(["--video", str(video), "--strict"]) == 1


def test_main_directory_batch_and_json(tmp_path, capsys):
    good, frames = _make_valid(tmp_path)                     # tmp_path/clip.avi (+csv)
    bad = tmp_path / "broken.avi"
    if not _write_video(bad):
        pytest.skip("No MJPG encoder available")             # no timing CSV for 'bad'

    rc = vmod.main(["--video", str(tmp_path), "--json"])
    assert rc == 1                                           # one input (broken) failed
    payload = json.loads(capsys.readouterr().out)
    by_video = {r["video"]: r for r in payload["reports"]}
    assert by_video[str(good)]["ok"] is True
    assert by_video[str(bad)]["ok"] is False
