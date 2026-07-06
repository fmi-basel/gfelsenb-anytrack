"""Tests for anytrack.preprocess single-pass ROI extraction (2a)."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from anytrack.preprocess import (extract_roi_videos, check_ffmpeg_available,
                                 cleanup_roi_videos, build_roi_backgrounds_uniform,
                                 roi_crop_geometry, model_background, _arena_mask)
from anytrack.models import CircleROI
from anytrack.config import AnyTrackConfig


def _cfg_adaptive():
    cfg = AnyTrackConfig()
    cfg.roi_downscale = 2
    return cfg


def test_model_background_easy_is_fast_tier0():
    """A roaming fly (never dwells) → tier 0 (plain percentile), no ghost, no flag."""
    sc, n = 100, 100
    stack = np.full((n, sc, sc), 200, np.uint8)
    for k in range(n):                                 # fly sweeps across, never lingers
        x = 20 + (k % 60)
        stack[k, 48:52, x:x + 4] = 20
    bg, info = model_background(stack, _cfg_adaptive())
    assert info["tier"] == 0
    assert info["n_ghost_final"] == 0
    assert np.median(bg[_arena_mask(sc)]) > 150


def test_model_background_dwell_escalates_and_is_ghost_free():
    """A long-dwell fly (present 92% of frames, leaves 8%) → tier 1 de-ghost lifts it,
    leaving a ghost-free background at the true floor."""
    sc, n = 100, 100
    stack = np.full((n, sc, sc), 200, np.uint8)
    for k in range(n):
        if k % 12 != 0:                                 # dwell at (30,30) 92% of frames
            stack[k, 26:34, 26:34] = 20
    bg, info = model_background(stack, _cfg_adaptive())
    assert info["tier"] >= 1 and info["method"].endswith("deghost")
    assert info["n_ghost_initial"] > 0 and info["n_ghost_final"] == 0
    assert bg[30, 30] > 150                             # dwell spot recovered to floor


def test_model_background_default_leaves_never_moved_fly_baked():
    """By default (T3 off) a never-moved fly stays baked in: no temporal statistic can
    recover it and the opt-in spatial fill is off (it would also fill fixtures)."""
    sc, n = 100, 100
    stack = np.full((n, sc, sc), 200, np.uint8)
    stack[:, 44:56, 44:56] = 20                         # compact dark fly, every frame
    bg, info = model_background(stack, _cfg_adaptive())
    assert info["n_stationary_filled"] == 0             # T3 is opt-in
    assert bg[50, 50] < 80                               # fly remains baked


def test_fill_stationary_opt_in_fills_a_compact_dark_spot():
    """The opt-in T3 spatial fill (fill_stationary) replaces a compact local dark spot
    with the surrounding floor when explicitly enabled — a capability, not a default
    (on real arenas it also fills the central port, so it stays off by default)."""
    sc, n = 100, 100
    stack = np.full((n, sc, sc), 200, np.uint8)
    stack[:, 44:56, 44:56] = 20
    cfg = _cfg_adaptive(); cfg.bg_model_fill_stationary = True
    bg, info = model_background(stack, cfg)
    assert info["n_stationary_filled"] > 0 and info["tier"] == 3
    assert bg[50, 50] > 150                              # dark spot replaced by floor


def _write_video(path, n=20, h=200, w=200):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for i in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (50 + i, 50), 5, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
def test_single_pass_extracts_all_rois(tmp_path):
    vid = tmp_path / "in.avi"
    if not _write_video(vid):
        pytest.skip("no MJPG encoder in this OpenCV build")

    # r=39 -> crop 78 -> downscaled 39 (ODD) without rounding; libx264 rejects
    # odd dims, so this guards the even-rounding fix in single-pass extraction.
    rois = [
        CircleROI(name="r0", cx=50, cy=50, r=39),
        CircleROI(name="r1", cx=150, cy=150, r=39),
    ]
    out_dir = tmp_path / "roi_out"
    # use_hw_encode=False keeps CI off VideoToolbox; single-pass forces libx264 anyway.
    res = extract_roi_videos(vid, rois, output_dir=out_dir, downscale=2, use_hw_encode=False)
    try:
        assert set(res.keys()) == {"r0", "r1"}
        for pr in res.values():
            assert pr.video_path.exists(), f"missing sub-video for {pr.roi_name}"
            cap = cv2.VideoCapture(str(pr.video_path))
            assert cap.isOpened()
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            assert n_frames > 0
            assert width % 2 == 0 and width > 0  # even dims (libx264 requires this)
            assert pr.scale_factor == 2.0
    finally:
        cleanup_roi_videos(res)


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
def test_build_roi_backgrounds_uniform_excludes_moving_fly(tmp_path):
    """The uniform per-ROI GMM must exclude a fly that roams the arena (no bake-in),
    and return correctly-shaped uint8 backgrounds for each ROI."""
    vid = tmp_path / "in.avi"
    if not _write_video(vid, n=40, h=200, w=200):  # bright arena, dark blob moves each frame
        pytest.skip("no MJPG encoder in this OpenCV build")

    cfg = AnyTrackConfig()
    cfg.roi_downscale = 2
    cfg.gmm_n_samples = 30
    roi = CircleROI(name="r0", cx=50, cy=50, r=39)
    _, _, _, scaled = roi_crop_geometry(roi, cfg.roi_downscale)

    fly_masks = {}
    out = build_roi_backgrounds_uniform(vid, [roi], cfg, fly_masks=fly_masks)
    assert out is not None and set(out.keys()) == {"r0"}
    bg = out["r0"]
    assert bg.shape == (scaled, scaled) and bg.dtype == np.uint8
    # arena is bright (~200); the roaming dark blob (~20) must NOT be baked in:
    assert np.median(bg) > 150
    assert (bg < 100).mean() < 0.05  # <5% dark pixels — no persistent blob

    # fly_masks side-channel: same shape, boolean, and the roaming blob makes some
    # pixels bimodal (2-component GMM) → a non-empty fly footprint.
    m = fly_masks["r0"]
    assert m.shape == (scaled, scaled) and m.dtype == np.bool_
    assert m.any()


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
def test_build_roi_backgrounds_uniform_fallback(tmp_path):
    """Returns None (worker then builds its own GMM) on unreadable input / no ROIs."""
    cfg = AnyTrackConfig()
    roi = CircleROI(name="r0", cx=50, cy=50, r=39)
    assert build_roi_backgrounds_uniform(tmp_path / "does_not_exist.avi", [roi], cfg) is None
    vid = tmp_path / "ok.avi"
    if _write_video(vid, n=10):
        assert build_roi_backgrounds_uniform(vid, [], cfg) is None  # no ROIs
