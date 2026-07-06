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


def test_debug_bg_methods_and_gating(tmp_path):
    """percentile & fg_excluded backgrounds build; contrast_gate / darkness_weight
    tracking runs (dwell-contamination A/B knobs)."""
    vid = tmp_path / "syn.avi"
    if not _synthetic_video(vid):
        pytest.skip("no MJPG encoder in this OpenCV build")
    cfg = _cfg()
    roi = CircleROI(name="arena_01", cx=120, cy=120, r=100)
    assert set(D.BG_METHODS) == {"adaptive", "gmm", "percentile", "fg_excluded"}

    perc = D.build_roi_background(vid, roi, cfg, method="percentile", percentile=90)
    assert perc is not None and perc.dtype == np.uint8 and np.median(perc) > 150

    adap = D.build_roi_background(vid, roi, cfg, method="adaptive")
    assert adap is not None and adap.dtype == np.uint8 and np.median(adap) > 150

    tracks = D.track_roi(vid, roi, perc, cfg)
    fge = D.build_roi_background(vid, roi, cfg, method="fg_excluded", percentile=90, tracks=tracks)
    assert fge is not None and fge.shape == perc.shape

    # gate + darkness-weighted linking still produces a valid, monotonic track
    gated = D.track_roi(vid, roi, perc, cfg, contrast_gate=10, darkness_weight=1.0)
    assert len(gated) >= 30
    assert np.corrcoef(gated["frame"].to_numpy(), gated["xs"].to_numpy())[0, 1] > 0.8


def test_deghost_lifts_fly_keeps_fixture():
    """de-ghost lifts a baked-in fly (dark in bg but bright somewhere in the stack)
    to arena brightness, while keeping a genuinely static dark fixture and clean
    pixels unchanged — no video/ffmpeg needed."""
    sc, n = 100, 100
    stack = np.full((n, sc, sc), 200, np.uint8)
    for k in range(n):                       # dwell fly at (30,30): present 95% of frames
        if k % 20 != 0:
            stack[k, 25:35, 25:35] = 20
    stack[:, 65:75, 65:75] = 25              # static fixture at (70,70): every frame

    bg = D._bg_from_stack(stack, AnyTrackConfig(), "percentile", 90)  # p90 bakes the fly in
    assert bg[30, 30] < 60 and bg[70, 70] < 60

    dg = D.deghost(bg, stack, percentile=98, margin=15)
    assert dg[30, 30] > 150     # fly lifted to arena (arena visible in the 5% clean frames)
    assert dg[70, 70] < 60      # static fixture preserved
    assert abs(int(dg[50, 50]) - int(bg[50, 50])) <= 2   # clean arena untouched


def test_deghost_neighbor_fill_matches_floor_not_nearmax():
    """neighbor fill borrows the true floor from nearby clean pixels (unbiased),
    where the per-pixel near-max overshoots: at a dwell spot whose few fly-absent
    frames happen to be bright glints, nearmax fills toward the glint while neighbor
    fills to the surrounding floor level."""
    sc, n = 100, 100
    stack = np.full((n, sc, sc), 200, np.uint8)   # floor = 200 everywhere
    for k in range(n):                            # fly dwells at (30,30) 90% of frames;
        if k % 10 != 0:                           # the 10 fly-absent frames are bright
            stack[k, 25:35, 25:35] = 20           # glints (255), so p98 overshoots
        else:
            stack[k, 25:35, 25:35] = 255

    bg = D._bg_from_stack(stack, AnyTrackConfig(), "percentile", 90)  # bakes the fly in
    assert bg[30, 30] < 80

    near = D.deghost(bg, stack, percentile=98, margin=15, fill="nearmax")
    nb = D.deghost(bg, stack, percentile=98, margin=15, fill="neighbor")
    assert near[30, 30] > 230                     # near-max chases the glint
    assert 185 <= nb[30, 30] <= 215               # neighbor ≈ true floor (200)
    assert nb[30, 30] < near[30, 30]              # neighbor does not overshoot
    assert abs(int(nb[50, 50]) - int(bg[50, 50])) <= 2   # clean arena untouched


def test_build_roi_background_deghost_flag(tmp_path):
    """apply_deghost is plumbed through build_roi_background end-to-end."""
    vid = tmp_path / "syn.avi"
    if not _synthetic_video(vid):
        pytest.skip("no MJPG encoder in this OpenCV build")
    cfg = _cfg()
    roi = CircleROI(name="arena_01", cx=120, cy=120, r=100)
    bg = D.build_roi_background(vid, roi, cfg, method="percentile", apply_deghost=True,
                                deghost_margin=15)
    assert bg is not None and bg.dtype == np.uint8 and np.median(bg) > 150


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
