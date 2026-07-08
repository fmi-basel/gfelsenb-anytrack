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


# ---------------------------------------------------------------------------
# Arena-disk masking: no contour/centroid detected outside the arena circle
# ---------------------------------------------------------------------------

def test_inscribed_disk_mask_geometry():
    from anytrack.detector import inscribed_disk_mask
    m = inscribed_disk_mask((100, 100))
    assert m.shape == (100, 100) and m.dtype == np.uint8
    assert m[50, 50] == 255          # centre inside the disk
    assert m[2, 2] == 0 and m[97, 97] == 0   # corners outside


def test_roi_mask_scaled_matches_downscaled_arena():
    from anytrack.roi import roi_mask_scaled
    roi = CircleROI(name="a", cx=100, cy=100, r=40)
    x0, y0 = 60, 60                  # roi_crop_geometry offset (cx-r, cy-r)
    m = roi_mask_scaled((40, 40), roi, (x0, y0), scale=2.0)   # scaled crop is 40×40
    assert m.shape == (40, 40)
    assert m[20, 20] == 255          # arena centre → (20, 20) at scale 2
    assert m[1, 1] == 0              # corner masked out


def test_extract_ellipses_masks_blob_outside_arena():
    """A blob in the crop's corner (outside the arena disk) is dropped by the
    mask, while the centred blob survives — the fix's core behaviour."""
    cfg = _det_cfg()
    bg = np.full((100, 100), 210, np.uint8)
    img = bg.copy()
    cv2.circle(img, (50, 50), 7, 20, -1)   # real fly, centre of the arena
    cv2.circle(img, (8, 8), 6, 20, -1)     # artifact in the corner, outside the disk
    kopen, kclose = build_morph_kernels(cfg)

    from anytrack.detector import inscribed_disk_mask
    unmasked = extract_ellipses(img, bg, cfg, kernel_open=kopen, kernel_close=kclose)
    assert len(unmasked) == 2              # both blobs detected without a mask

    mask = inscribed_disk_mask(img.shape[:2])
    masked = extract_ellipses(img, bg, cfg, mask=mask, kernel_open=kopen, kernel_close=kclose)
    assert len(masked) == 1                # corner blob removed
    assert abs(masked[0].x - 50) < 3 and abs(masked[0].y - 50) < 3


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


def test_legacy_track_video_with_drift_correction(tmp_path):
    """The opt-in bg_drift_correction path runs end-to-end and still tracks."""
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
    cfg = _det_cfg(bg_drift_correction=True, bg_asym_update=True)
    df = track_video(video, cfg).to_dataframe()
    assert not df.empty
    assert len(df) >= n // 2
    assert df["y"].between(30, 50).mean() > 0.8
    assert df["x"].max() - df["x"].min() > 10
    # drift correction records a per-frame drift for the QC plot
    assert "bg_drift" in df.columns and df["bg_drift"].notna().any()


def test_tracker_reacquires_after_jump_away_from_center():
    """A fly that loses lock while living away from the ROI center must be
    re-acquired on its actual candidate, not stall at the (wrong) ROI center.

    Regression for the arena_01 miss bug: re-init used to snap to the ROI
    center, so an off-center object was gated out forever after any loss.
    """
    from anytrack.tracker import CentroidTracker

    def cand(x, y, area=50.0):
        return [EllipseCandidate(x=x, y=y, angle_deg=0.0, cv_angle_deg=0.0,
                                 major=8.0, minor=4.0, area=area)]

    tr = CentroidTracker(center_xy=(100.0, 100.0), max_jump=10.0,
                         miss_tolerance=3, use_kalman=True)
    # Locked near (200,200), then a jump to (250,200) breaks lock; the object
    # then stays there — far from the ROI center (100,100).
    seq = [(200.0, 200.0)] * 5 + [(250.0, 200.0)] * 15
    got = [tr.step(cand(x, y))[0] is not None for x, y in seq]

    assert got[:5] == [True] * 5          # initial lock
    assert sum(got[-5:]) == 5             # fully re-acquired off-center (was 0 before the fix)


def test_extract_ellipses_surfaces_multiple_candidates():
    """The detector must return more than one blob so the tracker can
    disambiguate the fly from static artifacts (regression for arena_01)."""
    cfg = _det_cfg()  # detect_max_candidates defaults to 5
    bg = np.full((140, 140), 200, np.uint8)
    gray = bg.copy()
    cv2.circle(gray, (35, 35), 7, 20, -1)    # blob A
    cv2.circle(gray, (100, 100), 7, 20, -1)  # blob B
    cands = extract_ellipses(gray, bg, cfg)
    assert len(cands) >= 2


def test_tracker_ignores_static_artifact_once_locked():
    """Once locked on the moving fly, the tracker must keep picking it and
    ignore a larger, static artifact blob 150 px away."""
    from anytrack.tracker import CentroidTracker

    def cands(*pts):
        return [EllipseCandidate(x=x, y=y, angle_deg=0.0, cv_angle_deg=0.0,
                                 major=8.0, minor=4.0, area=a) for (x, y, a) in pts]

    tr = CentroidTracker(center_xy=(50.0, 50.0), max_jump=15.0,
                         miss_tolerance=15, use_kalman=True)
    for x in (40.0, 42.0, 44.0):               # lock on the fly (only candidate)
        tr.step(cands((x, 50.0, 60.0)))
    # Now a LARGER static artifact appears; the fly keeps moving.
    got = []
    for x in (46.0, 48.0, 50.0, 52.0, 54.0, 56.0):
        chosen, _, _ = tr.step(cands((x, 50.0, 60.0), (200.0, 200.0, 200.0)))
        got.append((chosen.x, chosen.y) if chosen else None)
    assert all(g is not None and g[1] == 50.0 and g[0] < 100 for g in got)


def test_tracker_fast_recovery_via_growing_gate():
    """A jump beyond max_jump is re-acquired within a few frames via the
    growing acceptance gate, without waiting out miss_tolerance.

    Regression for arena_01's residual transient gaps: with a large
    miss_tolerance the old tracker stayed lost for many frames after each fast
    excursion; the growing gate recovers in ~a couple of frames.
    """
    from anytrack.tracker import CentroidTracker

    def cand(x, y):
        return [EllipseCandidate(x=x, y=y, angle_deg=0.0, cv_angle_deg=0.0,
                                 major=8.0, minor=4.0, area=50.0)]

    tr = CentroidTracker(center_xy=(100.0, 100.0), max_jump=10.0,
                         miss_tolerance=100, use_kalman=True)
    for _ in range(4):
        tr.step(cand(100.0, 100.0))            # lock
    # 30px jump (> max_jump) held for a few frames; miss_tolerance is huge, so
    # recovery must come from the growing gate, not the miss_tolerance re-acquire.
    got = [tr.step(cand(130.0, 100.0))[0] is not None for _ in range(6)]
    assert got[-1] is True                     # re-locked
    assert sum(got) >= 3                        # recovered within a couple frames


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
