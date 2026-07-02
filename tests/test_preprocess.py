"""Tests for anytrack.preprocess single-pass ROI extraction (2a)."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from anytrack.preprocess import extract_roi_videos, check_ffmpeg_available, cleanup_roi_videos
from anytrack.models import CircleROI


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
