"""Tests for the anytrack-debug core (anytrack.gui.debug_app) — no display needed.

The Tk shell is exercised only for import/CLI wiring; the detection/tracking core
(background, stages, per-stage rendering, tracking) is validated on a synthetic
video (bright arena + a dark blob moving left→right)."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.gui import debug_app as D


def _synthetic_video(path, n=60):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (240, 240), True)
    if not wr.isOpened():
        wr.release()
        return False
    for i in range(n):
        f = np.zeros((240, 240), np.uint8)
        cv2.circle(f, (120, 120), 100, 200, -1)      # bright arena disk
        cv2.circle(f, (70 + i, 120), 7, 20, -1)      # dark "fly" moving right
        wr.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def _cfg():
    cfg = AnyTrackConfig()
    cfg.roi_downscale = 2
    cfg.gmm_n_samples = 20
    cfg.expected_fly_area_min = 5
    cfg.expected_fly_area_max = 2000
    return cfg


def test_debug_core_stages_and_tracking(tmp_path):
    vid = tmp_path / "syn.avi"
    if not _synthetic_video(vid):
        pytest.skip("no MJPG encoder in this OpenCV build")
    cfg = _cfg()
    roi = CircleROI(name="arena_01", cx=120, cy=120, r=100)

    # background excludes the moving blob (bright arena)
    bg = D.build_roi_background(vid, roi, cfg)
    assert bg is not None and bg.dtype == np.uint8
    assert np.median(bg) > 150

    # stages on a mid frame: exactly one candidate (the fly)
    cap = cv2.VideoCapture(str(vid)); cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, fr = cap.read(); cap.release()
    assert ok
    roi_gray = D.crop_roi_gray(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), roi, cfg.roi_downscale)
    st = D.detect_stages(roi_gray, bg, cfg)
    assert set(np.unique(st["mask"])).issubset({0, 255})
    assert len(st["candidates"]) >= 1

    # every stage renders to a 3-channel BGR image of the ROI size
    for stage in D.STAGES:
        img = D.render_stage(stage, roi_gray, bg, cfg, None, 30)
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.shape[:2] == roi_gray.shape[:2]

    # tracking follows the rightward-moving fly
    df = D.track_roi(vid, roi, bg, cfg)
    assert {"frame", "xs", "ys", "x", "y", "n_candidates", "jump_px"} <= set(df.columns)
    assert len(df) >= 40
    corr = np.corrcoef(df["frame"].to_numpy(), df["xs"].to_numpy())[0, 1]
    assert corr > 0.9  # xs increases with frame → tracks the moving fly

    # rendering the track overlay must use the "xs"/"ys" columns (both are reserved
    # pandas method names) — render "final"/"candidates" WITH the tracks df.
    for stage in ("final", "candidates"):
        img = D.render_stage(stage, roi_gray, bg, cfg, df, 30)
        assert img.ndim == 3 and img.shape[2] == 3

    # one-decode-pass multi-arena tracker agrees with the per-arena tracker
    multi = D.track_all_rois(vid, [roi], {roi.name: bg}, cfg)
    assert set(multi) == {roi.name} and len(multi[roi.name]) >= 40


def test_debug_background_fallback(tmp_path):
    """build_roi_background returns None on an unreadable video (worker copes)."""
    cfg = _cfg()
    roi = CircleROI(name="arena_01", cx=120, cy=120, r=100)
    assert D.build_roi_background(tmp_path / "nope.avi", roi, cfg) is None


def test_debug_cli_help(capsys):
    """The anytrack-debug entry point parses args (no display needed)."""
    with pytest.raises(SystemExit) as e:
        D.main(["--help"])
    assert e.value.code == 0
    assert "detection/tracking inspector" in capsys.readouterr().out
