"""
Legacy single-pass tracker (``cfg.fast_mode = False``).

Iterates the full video once, tracking every ROI per frame. Detection and
centroid linking are delegated to the shared ``detector`` and ``tracker``
modules; this file only orchestrates the frame loop and coordinate handling.
"""
from __future__ import annotations

from typing import Dict, Optional, Callable
import threading
import numpy as np
import cv2

from .models import VideoAsset, FlyTrack, EllipseObservation, TrackingResult
from .background import build_background
from .roi import roi_mask
from .config import AnyTrackConfig
from .detector import (
    EllipseCandidate,
    build_morph_kernels,
    extract_ellipses,
)
from .tracker import CentroidTracker


def track_video(
    video: VideoAsset,
    cfg: AnyTrackConfig,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_every: int = 1,
) -> TrackingResult:
    # 1) background
    bg_model = build_background(
        str(video.video_path),
        n_samples=cfg.gmm_n_samples,
        bic_improvement=cfg.gmm_bic_improvement,
        min_std=cfg.gmm_min_std,
        reg_covar=cfg.gmm_reg_covar,
        lowp=cfg.gmm_lowp,
        arena_detection=cfg.arena_detection_enabled,
        arena_min_area_frac=cfg.arena_min_area_frac,
        arena_blur_sigma=cfg.arena_blur_sigma,
    )
    bg = bg_model.gmm

    # 2) Ensure we have ROIs
    if not video.rois:
        raise ValueError("No ROIs found/defined on video. Run ROI detection/definition first.")

    # Per-ROI single track + tracker (extend later for multi-target).
    tracks: Dict[str, FlyTrack] = {}
    trackers: Dict[str, CentroidTracker] = {}
    for roi in video.rois:
        tracks[roi.name] = FlyTrack(roi_name=roi.name, track_id=1)
        trackers[roi.name] = CentroidTracker(
            center_xy=(roi.cx, roi.cy),
            max_jump=cfg.max_jump_px,
            miss_tolerance=cfg.miss_tolerance,
            use_kalman=cfg.use_kalman,
        )

    cap = cv2.VideoCapture(str(video.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video.video_path}")

    timing = video.timing
    if timing is None:
        cap.release()
        raise ValueError("VideoAsset has no timing table loaded.")

    total = int(len(timing))
    # Notify started
    try:
        if progress_hook is not None:
            progress_hook("started", {"total_frames": total})
    except Exception:
        pass

    # Pre-allocate morphology kernels (avoid creating per-frame)
    kernel_open, kernel_close = build_morph_kernels(cfg)

    # Pre-compute ROI masks and background crops (they don't change per-frame)
    roi_precomputed: Dict[str, dict] = {}
    for roi in video.rois:
        x0 = int(max(0, roi.cx - roi.r))
        y0 = int(max(0, roi.cy - roi.r))
        x1 = int(roi.cx + roi.r)
        y1 = int(roi.cy + roi.r)
        bg_roi = bg[y0:y1, x0:x1]
        # Pre-compute mask for this ROI
        roi_shape = (y1 - y0, x1 - x0)
        mask = roi_mask(roi_shape, roi, (x0, y0))
        roi_precomputed[roi.name] = {
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "bg_roi": bg_roi,
            "mask": mask,
        }

    # Extract timing data as numpy arrays (avoid iterrows overhead)
    frame_indices = timing["frame"].values.astype(np.int32)
    t_s_values = timing["t_s"].values.astype(np.float64)

    # Check if frames are consecutive for sequential reading optimization
    frames_consecutive = np.all(np.diff(frame_indices) == 1) if len(frame_indices) > 1 else True
    if frames_consecutive and len(frame_indices) > 0:
        # Seek to first frame once, then read sequentially
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))

    for i in range(total):
        # cooperative cancellation check
        if cancel_event is not None and cancel_event.is_set():
            break

        frame_idx = int(frame_indices[i])
        t_s = float(t_s_values[i])

        # Sequential read (no seeking) if frames are consecutive
        if not frames_consecutive:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for roi in video.rois:
            pre = roi_precomputed[roi.name]
            x0, y0 = pre["x0"], pre["y0"]
            roi_gray = gray[pre["y0"]:pre["y1"], pre["x0"]:pre["x1"]]
            bg_roi = pre["bg_roi"]
            mask = pre["mask"]

            cand = extract_ellipses(
                roi_gray, bg_roi, cfg,
                mask=mask,
                kernel_open=kernel_open,
                kernel_close=kernel_close,
            )

            # Candidates are ROI-local; convert to full-frame before linking so
            # the tracker (initialized at the full-frame ROI center) stays in
            # one coordinate system.
            cand_global = [
                EllipseCandidate(
                    x=c.x + x0,
                    y=c.y + y0,
                    angle_deg=c.angle_deg,
                    cv_angle_deg=c.cv_angle_deg,
                    major=c.major,
                    minor=c.minor,
                    area=c.area,
                    contour=c.contour,
                )
                for c in cand
            ]

            chosen, x_f, y_f = trackers[roi.name].step(cand_global)
            if chosen is None:
                continue

            obs = EllipseObservation(
                frame=frame_idx,
                t_s=t_s,
                x=float(x_f),
                y=float(y_f),
                angle_deg=float(chosen.angle_deg),
                cv_angle_deg=float(chosen.cv_angle_deg),
                major=float(chosen.major),
                minor=float(chosen.minor),
                area=float(chosen.area),
                contour_n=int(len(chosen.contour)) if chosen.contour is not None else 0,
            )
            tracks[roi.name].observations.append(obs)

        # report progress (throttled by progress_every)
        if progress_hook is not None and ((i % max(1, progress_every)) == 0):
            percent = float(i + 1) / float(max(1, total))
            progress_hook(
                "progress",
                {
                    "frame_idx": frame_idx,
                    "frame_count": int(i + 1),
                    "t_s": t_s,
                    "percent": percent,
                },
            )

    cap.release()
    return TrackingResult(video=video, tracks=list(tracks.values()), background=bg)
