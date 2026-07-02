"""
Regression tests for the shared detector/tracker refactor (Milestone A1/A2).

These use synthetic in-memory images and a small generated video, so they run
without ANYTRACK_TEST_VIDEO / ffmpeg.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.detector import (
    EllipseCandidate,
    build_morph_kernels,
    extract_ellipses,
    debug_frame,
    centroid_contrast,
)
from anytrack.tracker import CentroidTracker
from anytrack.models import CircleROI, VideoAsset
from anytrack.tracking import track_video


def _det_cfg(**overrides) -> AnyTrackConfig:
    """A config tuned for the synthetic dark-blob-on-bright scenes below."""
    cfg = AnyTrackConfig(
        gmm_n_samples=15,
        arena_detection_enabled=False,
        bgdiff_type="dark",
        thr_method="fixed",
        thr_fixed=50,
        morph_open=3,
        morph_close=5,
        max_centroids_per_roi=1,
        expected_fly_area_min=30,
        expected_fly_area_max=4000,
        max_jump_px=40.0,
        miss_tolerance=15,
        use_kalman=True,
        fast_mode=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _dark_blob(h=80, w=80, cx=40, cy=40, r=6, bg=210, fg=20) -> np.ndarray:
    img = np.full((h, w), bg, dtype=np.uint8)
    cv2.circle(img, (int(cx), int(cy)), int(r), int(fg), -1)
    return img


# ---------------------------------------------------------------------------
# EllipseCandidate contract (the field the legacy bug dropped)
# ---------------------------------------------------------------------------

def test_ellipse_candidate_has_all_fields():
    fields = [f.name for f in dataclasses.fields(EllipseCandidate)]
    assert fields == [
        "x", "y", "angle_deg", "cv_angle_deg", "major", "minor", "area", "contour"
    ]
    # contour is the only optional field.
    c = EllipseCandidate(x=1, y=2, angle_deg=3, cv_angle_deg=4, major=5, minor=6, area=7)
    assert c.contour is None


# ---------------------------------------------------------------------------
# detector.extract_ellipses / debug_frame on a synthetic scene
# ---------------------------------------------------------------------------

def test_extract_ellipses_detects_dark_blob():
    cfg = _det_cfg()
    img = _dark_blob()
    bg = np.full_like(img, 210)
    kopen, kclose = build_morph_kernels(cfg)

    cands = extract_ellipses(img, bg, cfg, kernel_open=kopen, kernel_close=kclose)
    assert len(cands) == 1
    c = cands[0]
    assert abs(c.x - 40) < 3 and abs(c.y - 40) < 3
    assert c.area >= cfg.expected_fly_area_min
    assert c.major >= c.minor
    assert np.isfinite(c.cv_angle_deg)   # populated, not dropped
    assert c.contour is not None


def test_debug_frame_matches_extract_ellipses():
    cfg = _det_cfg()
    img = _dark_blob()
    bg = np.full_like(img, 210)

    dbg = debug_frame(img, bg, cfg)
    assert dbg.diff.shape == img.shape
    assert dbg.mask.shape == img.shape
    assert set(np.unique(dbg.mask)).issubset({0, 255})
    assert len(dbg.candidates) == 1


def test_extract_ellipses_respects_mask():
    cfg = _det_cfg()
    img = _dark_blob()
    bg = np.full_like(img, 210)
    # Mask that excludes the blob entirely -> no candidate survives.
    mask = np.zeros_like(img)
    cv2.circle(mask, (10, 10), 5, 255, -1)
    cands = extract_ellipses(img, bg, cfg, mask=mask)
    assert cands == []


# ---------------------------------------------------------------------------
# tracker.CentroidTracker semantics
# ---------------------------------------------------------------------------

def _cand(x, y):
    return EllipseCandidate(
        x=x, y=y, angle_deg=0.0, cv_angle_deg=90.0, major=10.0, minor=5.0, area=50.0
    )


def test_centroid_tracker_assigns_nearest_and_smooths():
    tr = CentroidTracker(center_xy=(50, 50), max_jump=40, miss_tolerance=2, use_kalman=True)
    chosen, xf, yf = tr.step([_cand(51, 49)])
    assert chosen is not None
    assert abs(xf - 51) < 5 and abs(yf - 49) < 5
    assert tr.misses == 0


def test_centroid_tracker_rejects_far_candidate():
    tr = CentroidTracker(center_xy=(50, 50), max_jump=40, miss_tolerance=5, use_kalman=True)
    tr.step([_cand(50, 50)])          # init near center
    chosen, xf, yf = tr.step([_cand(300, 300)])  # jump beyond max_jump
    assert chosen is None and xf is None and yf is None
    assert tr.misses == 1


def test_centroid_tracker_miss_then_reinit():
    tr = CentroidTracker(center_xy=(50, 50), max_jump=40, miss_tolerance=2, use_kalman=True)
    tr.step([_cand(50, 50)])
    tr.step([])                       # miss 1
    assert tr.misses == 1
    tr.step([])                       # miss 2 (== tolerance, no reinit yet)
    assert tr.misses == 2
    tr.step([])                       # miss 3 (> tolerance -> reinit, reset)
    assert tr.misses == 0


def test_centroid_tracker_no_kalman_returns_raw_position():
    tr = CentroidTracker(center_xy=(50, 50), max_jump=40, miss_tolerance=2, use_kalman=False)
    chosen, xf, yf = tr.step([_cand(53, 48)])
    assert (xf, yf) == (53, 48)


# ---------------------------------------------------------------------------
# End-to-end legacy path — regression guard for the cv_angle_deg / contour bug.
# Before the fix, track_video raised TypeError on the first detection because
# EllipseCandidate was rebuilt with 7 positional args (contour missing).
# ---------------------------------------------------------------------------

def _write_synthetic_video(path, n=30, h=80, w=80):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not writer.isOpened():
        writer.release()
        return False
    for i in range(n):
        cx = 20 + int(round(i * (40.0 / (n - 1))))  # move right along y=40
        gray = _dark_blob(h=h, w=w, cx=cx, cy=40, r=6)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    return True


def test_legacy_track_video_runs_and_populates_cv_angle(tmp_path):
    video_path = tmp_path / "synthetic.avi"
    if not _write_synthetic_video(video_path):
        pytest.skip("No MJPG encoder available in this OpenCV build")

    n = 30
    timing = pd.DataFrame({"frame": np.arange(n), "t_s": np.arange(n) / 30.0})
    video = VideoAsset(
        video_path=video_path,
        timing_csv_path=video_path.with_suffix(".csv"),
        timing=timing,
        rois=[CircleROI(name="roi0", cx=40, cy=40, r=39)],
    )
    cfg = _det_cfg()

    # Must not raise (the bug raised TypeError here).
    result = track_video(video, cfg)
    df = result.to_dataframe()

    assert not df.empty
    assert "cv_angle_deg" in df.columns
    assert df["cv_angle_deg"].notna().all()
    # Tier-2 QC columns are populated end-to-end by the tracker.
    assert "n_candidates" in df.columns and "contrast" in df.columns
    assert (df["n_candidates"] >= 1).all()
    assert df["contrast"].notna().any()
    # Tracked the moving object across most frames, roughly along y=40.
    assert len(df) >= n // 2
    assert df["y"].between(30, 50).mean() > 0.8
    assert df["x"].max() - df["x"].min() > 10  # captured the rightward motion


def test_centroid_contrast():
    cfg = _det_cfg()  # bgdiff_type="dark"
    bg = np.full((50, 50), 200, np.uint8)
    gray = bg.copy()
    cv2.circle(gray, (25, 25), 4, 20, -1)  # dark blob on bright background

    # Dark object -> positive contrast (bg brighter than object) at the center.
    c = centroid_contrast(gray, bg, 25, 25, cfg)
    assert c > 100

    # Off-image centroid -> NaN.
    assert np.isnan(centroid_contrast(gray, bg, -5, -5, cfg))

    # "bright" mode inverts the sign for the same dark blob.
    cfg_bright = _det_cfg(bgdiff_type="bright")
    assert centroid_contrast(gray, bg, 25, 25, cfg_bright) < -100
