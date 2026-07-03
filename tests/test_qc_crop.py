"""Tests for the QC overlay crop options (--crop-roi / --follow)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.qc import _resolve_roi_bbox, render_overlay


def test_resolve_roi_bbox_uses_geometry():
    video = SimpleNamespace(rois=[CircleROI("arena_01", 100, 100, 40)])
    df = pd.DataFrame({"roi": ["arena_01"], "x": [100.0], "y": [100.0]})
    bbox = _resolve_roi_bbox("arena_01", video, df)
    assert bbox == (60.0, 60.0, 140.0, 140.0)             # cx±r


def test_resolve_roi_bbox_fallback_square_min():
    # No geometry + a near-stationary fly -> a padded square window, not a sliver.
    video = SimpleNamespace(rois=[])
    df = pd.DataFrame({"roi": ["r0"] * 3, "x": [100.0, 101.0, 100.5], "y": [50.0, 50.5, 50.0]})
    x0, y0, x1, y1 = _resolve_roi_bbox("r0", video, df)
    assert abs((x1 - x0) - (y1 - y0)) < 1e-6              # square
    assert (x1 - x0) >= 300.0                             # floored, usable
    assert _resolve_roi_bbox("missing", video, df) is None


def _write_video(path, n=6, h=200, w=200):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for _ in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (100, 100), 5, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def _tracks(n=6):
    return pd.DataFrame({"roi": ["r0"] * n, "track_id": [0] * n, "frame": range(n),
                         "x": [100.0] * n, "y": [100.0] * n, "t_s": [i / 30 for i in range(n)],
                         "area": [50.0] * n})


def test_render_overlay_crop_roi_dims(tmp_path):
    vp = tmp_path / "v.avi"
    if not _write_video(vp):
        pytest.skip("No MJPG encoder available")
    cfg = AnyTrackConfig(); cfg.qc_overlay_downscale = 1; cfg.qc_overlay_stride = 1
    video = SimpleNamespace(video_path=vp, width=200, height=200, frame_count=6,
                            fps_nominal=30.0, rois=[CircleROI("r0", 100, 100, 40)])
    p, n = render_overlay(video, _tracks(), cfg, tmp_path / "o.mp4")   # uncropped baseline
    if p is None:
        pytest.skip("No overlay encoder available")
    cap = cv2.VideoCapture(str(p)); w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
    assert w0 == 200

    # Cropped overlays scale the source region UP to crop_output_size (square),
    # independent of the source window — so annotations stay crisp.
    p2, _ = render_overlay(video, _tracks(), cfg, tmp_path / "roi.mp4",
                           crop_roi="r0", crop_output_size=160)
    cap = cv2.VideoCapture(str(p2))
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    assert (cw, ch) == (160, 160)

    p3, _ = render_overlay(video, _tracks(), cfg, tmp_path / "foll.mp4",
                           follow="r0", follow_size=64, crop_output_size=160)
    cap = cv2.VideoCapture(str(p3))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    assert (fw, fh) == (160, 160)                        # output size, not the 64px window
