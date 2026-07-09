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


def test_liveline_shows_per_stage_elapsed_and_eta():
    """Each thread shows its current stage's elapsed time, plus an ETA once a
    frame total is known (not for a bare phase-text stage)."""
    from anytrack.cli_progress import _LiveLine
    ll = _LiveLine(1, 4, "clip.avi")
    ll._apply("frames", {"stage": "track", "n": 40, "total": 100})
    assert "left" in ll.render("|")            # ETA shown when fraction is known

    ll2 = _LiveLine(1, 4, "clip.avi")
    ll2._set_phase("roi", "detecting ROIs…")     # no frame total → elapsed only
    out = ll2.render("|")
    assert "detecting ROIs" in out and "left" not in out


def test_batch_group_header_shows_total_and_eta():
    """The aggregate header reports done/total, elapsed, and a batch ETA."""
    from anytrack.cli_progress import BatchProgressGroup
    g = BatchProgressGroup(8)
    assert "~… left" in g._header()            # no ETA before the first video finishes
    g._committed = 2
    h = g._header()
    assert "2/8" in h and "done" in h and "elapsed" in h and "left" in h


def test_name_column_width_fits_full_filename():
    from anytrack.cli_progress import _LiveLine, BatchProgressGroup
    from anytrack.run import _name_col_width
    name = "localsearch_2025-12-11T09_57_32.avi"   # 35 chars

    assert name not in _LiveLine(1, 8, name).render("|")            # old default clips at 26
    assert name in _LiveLine(1, 8, name, name_w=len(name)).render("|")  # override shows full name

    assert _name_col_width([name]) >= 16 and _name_col_width(["ab"]) >= 16   # floors
    g = BatchProgressGroup(8, name_w=_name_col_width([name]))       # threads to every slot
    assert g.name_w == _name_col_width([name])
