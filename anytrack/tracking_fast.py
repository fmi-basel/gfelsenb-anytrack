"""
Fast parallel tracking for pre-cropped ROI videos.

Optimized for small pre-extracted ROI videos with sequential frame reading.
Detection and centroid linking are delegated to the shared ``detector`` and
``tracker`` modules.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
import numpy as np
import cv2
import pandas as pd

from .models import CircleROI, FlyTrack, EllipseObservation, TrackingResult, VideoAsset
from .background import build_background, BackgroundState
from .config import AnyTrackConfig
from .preprocess import PreprocessResult
from .detector import extract_ellipses, build_morph_kernels, centroid_contrast
from .tracker import CentroidTracker
from .coordinates import scaled_to_full


def track_roi_video(
    roi_video_path: Path,
    preprocess_result: PreprocessResult,
    timing: pd.DataFrame,
    cfg: AnyTrackConfig,
    roi_background: Optional[np.ndarray] = None,
    return_timing: bool = False,
):
    """
    Track a single pre-cropped ROI video.

    Args:
        roi_video_path: Path to the cropped ROI video
        preprocess_result: PreprocessResult with scaling info
        timing: Timing DataFrame with frame and t_s columns
        cfg: Tracking configuration
        roi_background: Optional pre-cropped background (avoids per-ROI background building)
        return_timing: If True, return (FlyTrack, {bg_build_s, track_s, n_frames}) for
            per-stage benchmarking; otherwise return the FlyTrack (production default).

    Returns:
        FlyTrack, or (FlyTrack, timing_dict) when return_timing=True.
    """
    scale = preprocess_result.scale_factor
    x0, y0 = preprocess_result.crop_offset
    roi = preprocess_result.original_roi

    # Use pre-cropped background if provided, otherwise build from ROI video
    bg_build_s = 0.0
    if roi_background is not None:
        bg = roi_background
    else:
        _t_bg = time.perf_counter()
        bg_model = build_background(
            str(roi_video_path),
            n_samples=cfg.gmm_n_samples,
            bic_improvement=cfg.gmm_bic_improvement,
            min_std=cfg.gmm_min_std,
            reg_covar=cfg.gmm_reg_covar,
            lowp=cfg.gmm_lowp,
            arena_detection=False,  # No arena detection for individual ROI videos
            arena_min_area_frac=cfg.arena_min_area_frac,
            arena_blur_sigma=0.0,  # No blur for ROI videos
        )
        bg = bg_model.gmm
        bg_build_s = time.perf_counter() - _t_bg

    # Pre-allocate morphology kernels
    kernel_open, kernel_close = build_morph_kernels(cfg)

    # Scale linking params for the downscaled ROI video.
    scaled_max_jump = cfg.max_jump_px / scale
    scaled_cx = roi.r / scale
    scaled_cy = roi.r / scale

    # Open video
    cap = cv2.VideoCapture(str(roi_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open ROI video: {roi_video_path}")

    track = FlyTrack(roi_name=roi.name, track_id=1)
    tracker = CentroidTracker(
        center_xy=(scaled_cx, scaled_cy),
        max_jump=scaled_max_jump,
        miss_tolerance=cfg.miss_tolerance,
        use_kalman=cfg.use_kalman,
    )

    # Opt-in illumination-drift correction (built per worker; not picklable).
    bg_state = None
    if getattr(cfg, "bg_drift_correction", False):
        bg_state = BackgroundState(bg, cfg, protect_radius=cfg.bg_protect_radius_px / scale)

    # Extract timing as arrays
    frame_indices = timing["frame"].values.astype(np.int32)
    t_s_values = timing["t_s"].values.astype(np.float64)

    n_proc = 0
    _t_track = time.perf_counter()
    for i in range(len(frame_indices)):
        frame_idx = int(frame_indices[i])
        t_s = float(t_s_values[i])

        ok, frame = cap.read()
        if not ok:
            break
        n_proc += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Drift-correct the background against the tracked object's last position
        # (opt-in); otherwise use the static background.
        bg_use = bg if bg_state is None else bg_state.corrected(gray, center=tracker.last)
        candidates = extract_ellipses(
            gray, bg_use, cfg,
            kernel_open=kernel_open,
            kernel_close=kernel_close,
        )

        chosen, x_f, y_f = tracker.step(candidates)
        if bg_state is not None:
            bg_state.note_foreground(chosen.contour if chosen is not None else None, gray.shape)
        if chosen is None:
            continue

        # Scale coordinates back to original full-frame coordinates.
        x_full, y_full = scaled_to_full(x_f, y_f, scale, x0, y0)
        # QC diagnostics (chosen is in scaled-local coords, matching gray/bg).
        contrast = centroid_contrast(gray, bg, chosen.x, chosen.y, cfg)
        obs = EllipseObservation(
            frame=frame_idx,
            t_s=t_s,
            x=float(x_full),
            y=float(y_full),
            angle_deg=float(chosen.angle_deg),
            cv_angle_deg=float(chosen.cv_angle_deg),
            major=float(chosen.major * scale),
            minor=float(chosen.minor * scale),
            area=float(chosen.area * scale * scale),
            contour_n=0,  # Not available in fast mode
            n_candidates=len(candidates),
            contrast=contrast,
        )
        track.observations.append(obs)

    track_s = time.perf_counter() - _t_track
    cap.release()
    if return_timing:
        return track, {"bg_build_s": bg_build_s, "track_s": track_s, "n_frames": n_proc}
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
