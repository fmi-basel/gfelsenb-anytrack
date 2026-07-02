"""Tests for anytrack.cli_progress (tqdm-driven progress hook)."""
from __future__ import annotations

from anytrack.cli_progress import TqdmProgress, header, step, ok, info


def _drive(p: TqdmProgress) -> None:
    # Fast path
    p("status", {"stage": "preprocessing"})
    p("preprocessing", {"roi": "single-pass", "percent": 0.0})
    p("preprocessing", {"roi": "complete", "percent": 1.0})
    p("status", {"stage": "tracking"})
    p("tracking", {"status": "starting", "n_rois": 4, "n_workers": 4})
    for i in range(1, 5):
        p("tracking", {"status": "progress", "completed": i, "total": 4, "percent": i / 4})
    p("tracking", {"status": "complete", "n_tracks": 4})
    # Legacy path
    p("started", {"total_frames": 100})
    for f in range(0, 101, 25):
        p("progress", {"frame_count": f, "percent": f / 100})
    p("done", {})


def test_tqdm_progress_quiet_handles_all_events():
    p = TqdmProgress(quiet=True)
    _drive(p)  # must not raise
    p.close()


def test_tqdm_progress_loud_handles_all_events():
    p = TqdmProgress(quiet=False)
    _drive(p)  # renders bars to stderr; must not raise
    p.close()


def test_tqdm_progress_survives_garbage_payloads():
    p = TqdmProgress(quiet=True)
    p("tracking", {})            # missing status
    p("progress", {})            # missing frame_count
    p("unknown-event", {"x": 1})  # unknown event
    p("status", {})              # missing stage
    p.close()


def test_format_helpers_do_not_raise(capsys):
    header("title")
    step("doing thing")
    ok("done thing")
    info("detail")
    out = capsys.readouterr().out
    assert "title" in out and "doing thing" in out
