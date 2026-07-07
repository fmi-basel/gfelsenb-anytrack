"""Headless tests for the anytrack-arena-debug core (no Tk)."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.tracker import CentroidTracker
from anytrack.detector import EllipseCandidate
from anytrack.preprocess import check_ffmpeg_available  # noqa: F401 (env parity)
from anytrack.gui import arena_debug as ad


def test_tracker_records_prediction_and_gate():
    """step() records the prediction point + acceptance gate (record-only)."""
    tr = CentroidTracker(center_xy=(50.0, 50.0), max_jump=20.0, miss_tolerance=5, use_kalman=True)
    c = EllipseCandidate(x=52.0, y=48.0, angle_deg=0, cv_angle_deg=0, major=6, minor=4, area=20)
    tr.step([c])
    assert tr.last_pred is not None and len(tr.last_pred) == 2
    assert tr.last_gate == 20.0                      # misses==0 → base gate
    tr.step([])                                      # miss: gate still 20 (misses was 0), misses→1
    assert tr.last_gate == pytest.approx(20.0)
    tr.step([])                                      # now misses==1 → gate widens
    assert tr.last_gate == pytest.approx(40.0)       # 20 * (1 + 1 miss)


def _base_cfg():
    return AnyTrackConfig(fast_mode=False, arena_detection_enabled=False,
                          bgdiff_type="dark", thr_method="fixed", thr_fixed=40,
                          expected_fly_area_min=5, expected_fly_area_max=4000,
                          roi_downscale=1, gmm_n_samples=10)


def _dbg_frame(sc):
    return dict(frame=3, chosen=(sc * 0.5, sc * 0.5), pred=(sc * 0.45, sc * 0.5),
                gate=8.0, missed=False,
                cands=[(sc * 0.5, sc * 0.5, 8.0, 5.0, 30.0)])


def test_base_stage_shapes_and_types():
    cfg = _base_cfg()
    sc = 60
    roi_gray = np.full((sc, sc), 200, np.uint8)
    cv2.circle(roi_gray, (30, 30), 5, 20, -1)
    bg = np.full((sc, sc), 200, np.uint8)
    for stage in ad.STAGES:
        img = ad._base_stage(stage, roi_gray, bg, cfg)
        assert img.shape == (sc, sc, 3) and img.dtype == np.uint8


def test_overlay_toggle_changes_pixels():
    """Enabling an overlay must change the rendered image; disabling all leaves
    only the base stage."""
    cfg = _base_cfg()
    sc = 60
    roi_gray = np.full((sc, sc), 200, np.uint8)
    bg = np.full((sc, sc), 200, np.uint8)
    dbg = _dbg_frame(sc)
    off = {k: False for k, _, _ in ad.OVERLAYS}
    base = ad.render_arena_frame(roi_gray, bg, cfg, dbg, [], stage="raw",
                                 overlays_on=off, colors=ad.DEFAULT_COLORS)
    assert np.array_equal(base, ad._base_stage("raw", roi_gray, bg, cfg))  # nothing drawn
    for key in ("candidates", "centroid", "prediction", "gate"):
        on = dict(off); on[key] = True
        img = ad.render_arena_frame(roi_gray, bg, cfg, dbg, [], stage="raw",
                                    overlays_on=on, colors=ad.DEFAULT_COLORS)
        assert not np.array_equal(img, base), f"overlay {key} drew nothing"


def test_overlay_color_is_respected():
    """A candidate drawn in a custom colour paints pixels of that colour."""
    cfg = _base_cfg()
    sc = 80
    roi_gray = np.full((sc, sc), 200, np.uint8)
    bg = np.full((sc, sc), 200, np.uint8)
    dbg = _dbg_frame(sc)
    custom = (255, 0, 0)   # BGR blue
    on = {k: (k == "candidates") for k, _, _ in ad.OVERLAYS}
    img = ad.render_arena_frame(roi_gray, bg, cfg, dbg, [], stage="raw",
                                overlays_on=on, colors={"candidates": custom})
    # some pixel is (near) pure blue
    blue = (img[:, :, 0] > 200) & (img[:, :, 1] < 60) & (img[:, :, 2] < 60)
    assert blue.any()


def test_trail_up_to():
    dbg = [dict(chosen=(float(i), float(i))) for i in range(10)]
    dbg[5]["chosen"] = None                          # a missed frame is skipped
    full = ad.trail_up_to(dbg, 9, -1)
    assert full == [(i, i) for i in range(10) if i != 5]
    capped = ad.trail_up_to(dbg, 9, 3)               # frames 6..9 inclusive (matches QC overlay)
    assert capped == [(6, 6), (7, 7), (8, 8), (9, 9)]


def test_hex_bgr_roundtrip():
    assert ad._hex_to_bgr("#ff8800") == (0, 136, 255)   # r=ff,g=88,b=00 → BGR
    assert ad._bgr_to_hex((0, 136, 255)) == "#ff8800"


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
def test_track_arena_debug_records_state(tmp_path):
    """The debug tracking pass records per-frame candidates + prediction + gate,
    and follows a moving fly."""
    sc = 80
    n, size = 24, 120
    vp = tmp_path / "in.avi"
    wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (size, size), isColor=True)
    if not wr.isOpened():
        wr.release(); pytest.skip("no MJPG encoder")
    xs = np.linspace(30, 90, n)
    for i in range(n):
        g = np.full((size, size), 200, np.uint8)
        cv2.circle(g, (int(xs[i]), 60), 5, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()

    cfg = _base_cfg()
    roi = CircleROI(name="arena_01", cx=60, cy=60, r=55)
    bg = ad.build_roi_background(str(vp), roi, cfg, method="adaptive")
    assert bg is not None
    dbg = ad.track_arena_debug(str(vp), roi, bg, cfg)
    assert len(dbg) == n
    assert all("cands" in d and "gate" in d for d in dbg)
    detected = [d for d in dbg if d["chosen"] is not None]
    assert len(detected) >= n // 2                    # tracks the moving blob
    # the chosen centroid moves left→right with the fly
    xs_tracked = [d["chosen"][0] for d in detected]
    assert xs_tracked[-1] - xs_tracked[0] > 10
    assert any(d["pred"] is not None for d in dbg)     # prediction recorded
