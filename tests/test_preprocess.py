"""Tests for anytrack.preprocess single-pass ROI extraction (2a)."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from anytrack.preprocess import (extract_roi_videos, check_ffmpeg_available,
                                 cleanup_roi_videos, build_roi_backgrounds_uniform,
                                 roi_crop_geometry)
from anytrack.models import CircleROI
from anytrack.config import AnyTrackConfig


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

    out = build_roi_backgrounds_uniform(vid, [roi], cfg)
    assert out is not None and set(out.keys()) == {"r0"}
    bg = out["r0"]
    assert bg.shape == (scaled, scaled) and bg.dtype == np.uint8
    # arena is bright (~200); the roaming dark blob (~20) must NOT be baked in:
    assert np.median(bg) > 150
    assert (bg < 100).mean() < 0.05  # <5% dark pixels — no persistent blob


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
def test_build_roi_backgrounds_uniform_fallback(tmp_path):
    """Returns None (worker then builds its own GMM) on unreadable input / no ROIs."""
    cfg = AnyTrackConfig()
    roi = CircleROI(name="r0", cx=50, cy=50, r=39)
    assert build_roi_backgrounds_uniform(tmp_path / "does_not_exist.avi", [roi], cfg) is None
    vid = tmp_path / "ok.avi"
    if _write_video(vid, n=10):
        assert build_roi_backgrounds_uniform(vid, [], cfg) is None  # no ROIs
