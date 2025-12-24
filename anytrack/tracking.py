from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable
import threading
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

from .models import VideoAsset, CircleROI, FlyTrack, EllipseObservation, TrackingResult
from .background import build_background
from .roi import crop_roi, roi_mask
from .config import AnyTrackConfig

@dataclass
class EllipseCandidate:
    x: float
    y: float
    angle_deg: float
    major: float
    minor: float
    area: float
    contour: np.ndarray

class Kalman2D:
    # Constant velocity Kalman filter using OpenCV
    def __init__(self, x: float, y: float):
        self.kf = cv2.KalmanFilter(4, 2)
        # state: [x, y, vx, vy]
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)

    def predict(self) -> Tuple[float, float]:
        s = self.kf.predict()
        return float(s[0, 0]), float(s[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        m = np.array([[x], [y]], np.float32)
        s = self.kf.correct(m)
        return float(s[0, 0]), float(s[1, 0])

def _build_morph_kernels(cfg: AnyTrackConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pre-allocate morphology structuring elements."""
    kernel_open = None
    kernel_close = None
    if cfg.morph_open > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_open, cfg.morph_open))
    if cfg.morph_close > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close))
    return kernel_open, kernel_close


def _threshold_fg(
    diff: np.ndarray,
    cfg: AnyTrackConfig,
    kernel_open: Optional[np.ndarray] = None,
    kernel_close: Optional[np.ndarray] = None,
) -> np.ndarray:
    if cfg.thr_method == "fixed":
        _, bw = cv2.threshold(diff, int(cfg.thr_fixed), 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if kernel_open is not None:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open)
    if kernel_close is not None:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close)
    return bw

def _extract_ellipses(
    roi_gray: np.ndarray,
    bg_roi_gray: np.ndarray,
    cfg: AnyTrackConfig,
    mask: Optional[np.ndarray] = None,
    kernel_open: Optional[np.ndarray] = None,
    kernel_close: Optional[np.ndarray] = None,
) -> List[EllipseCandidate]:
    # Compute background difference based on type
    bgdiff_type = getattr(cfg, 'bgdiff_type', 'dark')
    if bgdiff_type == "dark":
        # Only pixels darker than background
        diff = cv2.subtract(bg_roi_gray, roi_gray)
    elif bgdiff_type == "bright":
        # Only pixels brighter than background
        diff = cv2.subtract(roi_gray, bg_roi_gray)
    else:  # "absolute"
        # Absolute difference (original behavior)
        diff = cv2.absdiff(roi_gray, bg_roi_gray)

    if mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=mask)
    bw = _threshold_fg(diff, cfg, kernel_open, kernel_close)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand: List[EllipseCandidate] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < cfg.expected_fly_area_min or area > cfg.expected_fly_area_max:
            continue
        if len(c) < 5:
            continue
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        major = float(max(MA, ma))
        minor = float(min(MA, ma))
        # Fix angle: OpenCV has y-axis flipped, so geometric angle = -opencv_angle (in degrees)
        geometric_angle = (-float(angle)) % 360.0
        cand.append(EllipseCandidate(
            x=float(x), y=float(y), angle_deg=geometric_angle,
            major=major, minor=minor, area=area, contour=c
        ))
    # Sort: largest area first (often best)
    cand.sort(key=lambda e: e.area, reverse=True)
    return cand[: max(1, cfg.max_centroids_per_roi)]

def _greedy_assign(pred: Tuple[float, float], candidates: List[EllipseCandidate], max_jump: float) -> Optional[EllipseCandidate]:
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

    # 2) Ensure we have ROIs
    if not video.rois:
        raise ValueError("No ROIs found/defined on video. Run ROI detection/definition first.")

    # Create per-ROI single track (extend later for multi-target)
    tracks: Dict[str, FlyTrack] = {}
    kalman: Dict[str, Optional[Kalman2D]] = {}
    misses: Dict[str, int] = {}

    for roi in video.rois:
        tracks[roi.name] = FlyTrack(roi_name=roi.name, track_id=1)
        kalman[roi.name] = None
        misses[roi.name] = 0

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

    # === PHASE 1 OPTIMIZATIONS ===
    # Pre-allocate morphology kernels (avoid creating per-frame)
    kernel_open, kernel_close = _build_morph_kernels(cfg)

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
            x0, y0, x1, y1 = pre["x0"], pre["y0"], pre["x1"], pre["y1"]
            roi_gray = gray[y0:y1, x0:x1]
            bg_roi = pre["bg_roi"]
            mask = pre["mask"]

            cand = _extract_ellipses(
                roi_gray, bg_roi, cfg,
                mask=mask,
                kernel_open=kernel_open,
                kernel_close=kernel_close,
            )

            kf = kalman[roi.name]
            if kf is None:
                # initialize with best candidate (or ROI center)
                if cand:
                    kf = Kalman2D(cand[0].x + x0, cand[0].y + y0)
                else:
                    kf = Kalman2D(roi.cx, roi.cy)
                kalman[roi.name] = kf

            px, py = kf.predict() if cfg.use_kalman else (roi.cx, roi.cy)
            # candidates are in ROI-local coords; convert
            cand_global = [
                EllipseCandidate(c.x + x0, c.y + y0, c.angle_deg, c.major, c.minor, c.area, c.contour)
                for c in cand
            ]
            chosen = _greedy_assign((px, py), cand_global, cfg.max_jump_px)

            if chosen is None:
                misses[roi.name] += 1
                if misses[roi.name] > cfg.miss_tolerance:
                    # re-init near ROI center
                    kalman[roi.name] = Kalman2D(roi.cx, roi.cy)
                    misses[roi.name] = 0
                continue

            misses[roi.name] = 0
            if cfg.use_kalman:
                x_f, y_f = kf.update(chosen.x, chosen.y)
            else:
                x_f, y_f = chosen.x, chosen.y

            obs = EllipseObservation(
                frame=frame_idx,
                t_s=t_s,
                x=float(x_f),
                y=float(y_f),
                angle_deg=float(chosen.angle_deg),
                major=float(chosen.major),
                minor=float(chosen.minor),
                area=float(chosen.area),
                contour_n=int(len(chosen.contour)),
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
