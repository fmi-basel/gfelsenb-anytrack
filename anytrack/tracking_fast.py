"""
Fast parallel tracking for pre-cropped ROI videos.

Optimized for small pre-extracted ROI videos with sequential frame reading.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import cv2
import pandas as pd

from .models import CircleROI, FlyTrack, EllipseObservation, TrackingResult, VideoAsset
from .background import build_background
from .config import AnyTrackConfig
from .preprocess import PreprocessResult


@dataclass
class _EllipseCandidate:
    """Ellipse candidate in local ROI coordinates."""
    x: float
    y: float
    angle_deg: float
    major: float
    minor: float
    area: float


class _Kalman2D:
    """Constant velocity Kalman filter for position tracking."""

    def __init__(self, x: float, y: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)

    def predict(self) -> Tuple[float, float]:
        s = self.kf.predict()
        return float(s[0, 0]), float(s[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        m = np.array([[x], [y]], np.float32)
        s = self.kf.correct(m)
        return float(s[0, 0]), float(s[1, 0])


def _extract_ellipses(
    gray: np.ndarray,
    bg_gray: np.ndarray,
    cfg: AnyTrackConfig,
    kernel_open: Optional[np.ndarray] = None,
    kernel_close: Optional[np.ndarray] = None,
) -> List[_EllipseCandidate]:
    """Extract ellipse candidates from a frame."""
    diff = cv2.absdiff(gray, bg_gray)

    # Threshold
    if cfg.thr_method == "fixed":
        _, bw = cv2.threshold(diff, int(cfg.thr_fixed), 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology
    if kernel_open is not None:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open)
    if kernel_close is not None:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close)

    # Find contours
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[_EllipseCandidate] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < cfg.expected_fly_area_min or area > cfg.expected_fly_area_max:
            continue
        if len(c) < 5:
            continue

        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        major = float(max(MA, ma))
        minor = float(min(MA, ma))

        candidates.append(_EllipseCandidate(
            x=float(x),
            y=float(y),
            angle_deg=float(angle),
            major=major,
            minor=minor,
            area=area,
        ))

    # Sort by area (largest first)
    candidates.sort(key=lambda e: e.area, reverse=True)
    return candidates[:max(1, cfg.max_centroids_per_roi)]


def _greedy_assign(
    pred: Tuple[float, float],
    candidates: List[_EllipseCandidate],
    max_jump: float,
) -> Optional[_EllipseCandidate]:
    """Assign closest candidate within max_jump distance."""
    if not candidates:
        return None

    px, py = pred
    best = None
    best_d = float("inf")

    for c in candidates:
        d = float(np.hypot(c.x - px, c.y - py))
        if d < best_d:
            best_d = d
            best = c

    if best is not None and best_d <= max_jump:
        return best
    return None


def track_roi_video(
    roi_video_path: Path,
    preprocess_result: PreprocessResult,
    timing: pd.DataFrame,
    cfg: AnyTrackConfig,
    roi_background: Optional[np.ndarray] = None,
) -> FlyTrack:
    """
    Track a single pre-cropped ROI video.

    Args:
        roi_video_path: Path to the cropped ROI video
        preprocess_result: PreprocessResult with scaling info
        timing: Timing DataFrame with frame and t_s columns
        cfg: Tracking configuration
        roi_background: Optional pre-cropped background (avoids per-ROI background building)

    Returns:
        FlyTrack with observations in original frame coordinates
    """
    scale = preprocess_result.scale_factor
    x0, y0 = preprocess_result.crop_offset
    roi = preprocess_result.original_roi

    # Use pre-cropped background if provided, otherwise build from ROI video
    if roi_background is not None:
        bg = roi_background
    else:
        bg_model = build_background(
            str(roi_video_path),
            method=cfg.bg_method,
            k_min=cfg.bg_min_frames,
            k_max=cfg.bg_max_frames,
            converge_eps=cfg.bg_converge_eps,
            fg_thresh=cfg.bg_fg_thresh,
            inpaint_ghosts=cfg.bg_inpaint_ghosts,
            ghost_area_min=cfg.bg_ghost_area_min,
            ghost_area_max=cfg.bg_ghost_area_max,
        )
        bg = bg_model.image

    # Pre-allocate morphology kernels
    kernel_open = None
    kernel_close = None
    if cfg.morph_open > 0:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_open, cfg.morph_open)
        )
    if cfg.morph_close > 0:
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close)
        )

    # Scale max_jump for downscaled video
    scaled_max_jump = cfg.max_jump_px / scale

    # Open video
    cap = cv2.VideoCapture(str(roi_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open ROI video: {roi_video_path}")

    track = FlyTrack(roi_name=roi.name, track_id=1)
    kf: Optional[_Kalman2D] = None
    misses = 0

    # Center of ROI in scaled coordinates
    scaled_cx = roi.r / scale
    scaled_cy = roi.r / scale

    # Extract timing as arrays
    frame_indices = timing["frame"].values.astype(np.int32)
    t_s_values = timing["t_s"].values.astype(np.float64)

    for i in range(len(frame_indices)):
        frame_idx = int(frame_indices[i])
        t_s = float(t_s_values[i])

        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        candidates = _extract_ellipses(
            gray, bg, cfg,
            kernel_open=kernel_open,
            kernel_close=kernel_close,
        )

        # Initialize Kalman if needed
        if kf is None:
            if candidates:
                kf = _Kalman2D(candidates[0].x, candidates[0].y)
            else:
                kf = _Kalman2D(scaled_cx, scaled_cy)

        # Predict position
        if cfg.use_kalman:
            px, py = kf.predict()
        else:
            px, py = scaled_cx, scaled_cy

        # Assign candidate
        chosen = _greedy_assign((px, py), candidates, scaled_max_jump)

        if chosen is None:
            misses += 1
            if misses > cfg.miss_tolerance:
                kf = _Kalman2D(scaled_cx, scaled_cy)
                misses = 0
            continue

        misses = 0

        # Update Kalman
        if cfg.use_kalman:
            x_f, y_f = kf.update(chosen.x, chosen.y)
        else:
            x_f, y_f = chosen.x, chosen.y

        # Scale coordinates back to original frame
        obs = EllipseObservation(
            frame=frame_idx,
            t_s=t_s,
            x=float(x_f * scale + x0),
            y=float(y_f * scale + y0),
            angle_deg=float(chosen.angle_deg),
            major=float(chosen.major * scale),
            minor=float(chosen.minor * scale),
            area=float(chosen.area * scale * scale),
            contour_n=0,  # Not available in fast mode
        )
        track.observations.append(obs)

    cap.release()
    return track


def _track_roi_worker(args: tuple) -> FlyTrack:
    """Worker function for multiprocessing."""
    roi_video_path, preprocess_result, timing_dict, cfg_dict, bg_data = args

    # Reconstruct objects from serializable forms
    from .config import AnyTrackConfig
    from .preprocess import PreprocessResult
    from .models import CircleROI

    cfg = AnyTrackConfig(**cfg_dict)
    timing = pd.DataFrame(timing_dict)

    # Reconstruct PreprocessResult
    pr = PreprocessResult(
        roi_name=preprocess_result["roi_name"],
        video_path=Path(preprocess_result["video_path"]),
        original_roi=CircleROI(**preprocess_result["original_roi"]),
        scale_factor=preprocess_result["scale_factor"],
        crop_offset=tuple(preprocess_result["crop_offset"]),
    )

    # Reconstruct background from serialized data if provided
    roi_background = None
    if bg_data is not None:
        bg_bytes, bg_shape, bg_dtype = bg_data
        roi_background = np.frombuffer(bg_bytes, dtype=bg_dtype).reshape(bg_shape)

    return track_roi_video(Path(roi_video_path), pr, timing, cfg, roi_background=roi_background)


def track_parallel(
    preprocess_results: Dict[str, PreprocessResult],
    timing: pd.DataFrame,
    cfg: AnyTrackConfig,
    n_workers: Optional[int] = None,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    roi_backgrounds: Optional[Dict[str, np.ndarray]] = None,
) -> List[FlyTrack]:
    """
    Track all ROIs in parallel using multiprocessing.

    Args:
        preprocess_results: Dict mapping roi_name -> PreprocessResult
        timing: Timing DataFrame
        cfg: Tracking configuration
        n_workers: Number of worker processes (default: CPU count)
        progress_hook: Optional callback for progress updates
        roi_backgrounds: Optional dict mapping roi_name -> pre-cropped background image

    Returns:
        List of FlyTrack objects
    """
    if not preprocess_results:
        return []

    if n_workers is None:
        n_workers = min(len(preprocess_results), mp.cpu_count())

    # Prepare arguments for workers (must be serializable)
    timing_dict = timing.to_dict(orient="list")
    cfg_dict = {
        k: v for k, v in cfg.__dict__.items()
        if not k.startswith("_")
    }

    worker_args = []
    for roi_name, pr in preprocess_results.items():
        pr_dict = {
            "roi_name": pr.roi_name,
            "video_path": str(pr.video_path),
            "original_roi": {
                "name": pr.original_roi.name,
                "cx": pr.original_roi.cx,
                "cy": pr.original_roi.cy,
                "r": pr.original_roi.r,
                "n_targets": pr.original_roi.n_targets,
            },
            "scale_factor": pr.scale_factor,
            "crop_offset": list(pr.crop_offset),
        }
        # Serialize background for multiprocessing if provided
        bg_data = None
        if roi_backgrounds is not None and roi_name in roi_backgrounds:
            bg = roi_backgrounds[roi_name]
            bg_data = (bg.tobytes(), bg.shape, str(bg.dtype))
        worker_args.append((str(pr.video_path), pr_dict, timing_dict, cfg_dict, bg_data))

    if progress_hook:
        progress_hook("tracking", {
            "status": "starting",
            "n_rois": len(preprocess_results),
            "n_workers": n_workers,
        })

    # Run parallel tracking
    tracks: List[FlyTrack] = []

    if n_workers == 1:
        # Single worker - no multiprocessing overhead
        for i, args in enumerate(worker_args):
            track = _track_roi_worker(args)
            tracks.append(track)
            if progress_hook:
                progress_hook("tracking", {
                    "status": "progress",
                    "completed": i + 1,
                    "total": len(worker_args),
                    "percent": (i + 1) / len(worker_args),
                })
    else:
        # Use multiprocessing pool
        with mp.Pool(n_workers) as pool:
            for i, track in enumerate(pool.imap_unordered(_track_roi_worker, worker_args)):
                tracks.append(track)
                if progress_hook:
                    progress_hook("tracking", {
                        "status": "progress",
                        "completed": i + 1,
                        "total": len(worker_args),
                        "percent": (i + 1) / len(worker_args),
                    })

    if progress_hook:
        progress_hook("tracking", {
            "status": "complete",
            "n_tracks": len(tracks),
        })

    return tracks


def track_video_fast(
    video: VideoAsset,
    cfg: AnyTrackConfig,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    cleanup: bool = True,
) -> TrackingResult:
    """
    Fast tracking using pre-cropped ROI videos and parallel processing.

    Args:
        video: VideoAsset with ROIs defined
        cfg: Tracking configuration (should have fast_mode settings)
        progress_hook: Optional callback for progress updates
        cleanup: Whether to delete temp ROI videos after tracking

    Returns:
        TrackingResult with all tracks
    """
    from .preprocess import extract_roi_videos, cleanup_roi_videos

    if not video.rois:
        raise ValueError("No ROIs defined on video")

    if video.timing is None:
        raise ValueError("VideoAsset has no timing table")

    # Get fast mode settings from config
    downscale = getattr(cfg, "roi_downscale", 2)
    n_workers = getattr(cfg, "n_tracking_workers", None)
    use_hw_encode = getattr(cfg, "use_hw_encode", True)

    # Extract ROI videos
    if progress_hook:
        progress_hook("status", {"stage": "preprocessing"})

    preprocess_results = extract_roi_videos(
        input_video=video.video_path,
        rois=video.rois,
        downscale=downscale,
        use_hw_encode=use_hw_encode,
        progress_hook=progress_hook,
    )

    # Run parallel tracking (each worker builds its own background from ROI video)
    if progress_hook:
        progress_hook("status", {"stage": "tracking"})

    try:
        tracks = track_parallel(
            preprocess_results=preprocess_results,
            timing=video.timing,
            cfg=cfg,
            n_workers=n_workers,
            progress_hook=progress_hook,
        )
    finally:
        # Cleanup temp files
        if cleanup:
            cleanup_roi_videos(preprocess_results)

    return TrackingResult(video=video, tracks=tracks, background=None)