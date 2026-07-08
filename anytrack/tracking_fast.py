"""
Fast parallel tracking for pre-cropped ROI videos.

Optimized for small pre-extracted ROI videos with sequential frame reading.
Detection and centroid linking are delegated to the shared ``detector`` and
``tracker`` modules.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Callable
import numpy as np
import cv2
import pandas as pd


class _StreamingUnavailable(Exception):
    """Raised when the decode-once FIFO path cannot run; caller falls back to files."""


class _FrameCounter:
    """Minimal ``.value`` counter for the single-worker (in-process) frame bar.

    Mirrors the ``.value`` interface of a ``multiprocessing.Manager().Value`` so
    the same worker/poller code drives both the in-process and pool paths.
    """
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0


class _FrameProgress:
    """Shared cross-worker frame counter + poller feeding the tracking bar.

    Reused by ``track_parallel`` (file path) and the streaming path. Under
    ``n_workers == 1`` the counter is an in-process object; otherwise a picklable
    ``Manager().Value`` proxy the workers increment. A daemon poller turns the
    count into ``("frames", {stage: track, ...})`` events. Entirely best-effort —
    on any setup failure ``shared`` stays ``(None, None)`` and tracking runs
    without a bar.
    """

    def __init__(self, progress_hook, n_workers, total_frames):
        self.total = int(total_frames)
        self.shared = (None, None)
        self._manager = None
        self._poller = None
        self._stop = None
        if progress_hook is None:
            return
        try:
            self._stop = threading.Event()
            if n_workers == 1:            # in-process: plain object + thread lock
                counter, lock = _FrameCounter(), threading.Lock()
            else:                         # pool: picklable Manager proxies
                self._manager = mp.Manager()
                counter, lock = self._manager.Value("i", 0), self._manager.Lock()
            self.shared = (counter, lock)
            total = self.total

            def _poll(_c=counter, _stop=self._stop):
                while not _stop.wait(0.15):
                    try:
                        n = int(_c.value)
                    except Exception:
                        return
                    try:
                        progress_hook("frames", {"stage": "track", "n": n, "total": total})
                    except Exception:
                        pass

            self._poller = threading.Thread(target=_poll, daemon=True)
            self._poller.start()
        except Exception:
            self.shared = (None, None)
            self._manager = self._poller = self._stop = None

    def finish(self, progress_hook):
        """Stop the poller, snap the bar to 100%, tear down the Manager."""
        if self._stop is not None:
            self._stop.set()
        if self._poller is not None:
            self._poller.join(timeout=0.5)
        if progress_hook is not None:
            try:
                progress_hook("frames", {"stage": "track", "n": self.total, "total": self.total})
            except Exception:
                pass
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass


from .models import CircleROI, FlyTrack, EllipseObservation, TrackingResult, VideoAsset
from .background import build_background, BackgroundState
from .config import AnyTrackConfig
from .preprocess import PreprocessResult
from .detector import extract_ellipses, build_morph_kernels, centroid_contrast
from .tracker import CentroidTracker
from .coordinates import scaled_to_full
from .roi import roi_mask_scaled


def _rescale_detect_params(cfg: AnyTrackConfig, scale: float) -> AnyTrackConfig:
    """Rescale area/morph thresholds from the reference downscale to ``scale``.

    ``expected_fly_area_min/max`` and ``morph_open/close`` are tuned in downscaled
    pixels at ``detect_params_ref_downscale`` (default 2). At another downscale the
    object's linear size changes by ``ref/scale``, so areas scale by that squared
    and morph kernels linearly — keeping the detector in physical units so
    ``roi_downscale=4`` (or 1) detects the same flies without hand-retuning.
    """
    import dataclasses
    ref = int(getattr(cfg, "detect_params_ref_downscale", 2) or 2)
    if scale == ref:
        return cfg
    r = ref / float(scale)
    return dataclasses.replace(
        cfg,
        expected_fly_area_min=int(max(1, round(cfg.expected_fly_area_min * r * r))),
        expected_fly_area_max=int(max(1, round(cfg.expected_fly_area_max * r * r))),
        morph_open=int(max(1, round(cfg.morph_open * r))),
        morph_close=int(max(1, round(cfg.morph_close * r))),
    )


def _track_from_frames(
    frame_iter,
    preprocess_result: PreprocessResult,
    timing: pd.DataFrame,
    cfg: AnyTrackConfig,
    bg: Optional[np.ndarray],
    on_frames: Optional[Callable[[int], None]] = None,
    return_timing: bool = False,
    stride: int = 1,
):
    """Track one ROI from an iterator of **grayscale** frames against ``bg``.

    The detection + linking core shared by the file path (``track_roi_video``,
    frames from an mp4 via cv2) and the streaming path (``_track_roi_stream``,
    raw gray frames from a FIFO — no ``cvtColor``). ``frame_iter`` yields 2-D
    uint8 arrays; iteration stops at the first ``None``/exhaustion or when the
    timing table is consumed. Returns a FlyTrack, or ``(FlyTrack, {bg_build_s:
    0.0, track_s, n_frames})`` when ``return_timing`` (callers patch bg_build_s).
    """
    scale = preprocess_result.scale_factor
    x0, y0 = preprocess_result.crop_offset
    roi = preprocess_result.original_roi

    # Keep area/morph in physical units when the downscale differs from the
    # reference the thresholds were tuned at (so downscale=4 detects correctly).
    cfg = _rescale_detect_params(cfg, scale)

    # With stride > 1 the fly moves up to stride× farther between tracked frames,
    # so widen the max allowed jump accordingly (else the tracker rejects the real
    # detection and drifts). Skipped frames are interpolated downstream.
    stride = max(1, int(stride))

    kernel_open, kernel_close = build_morph_kernels(cfg)
    tracker = CentroidTracker(
        center_xy=(roi.r / scale, roi.r / scale),
        max_jump=cfg.max_jump_px * stride / scale,
        miss_tolerance=cfg.miss_tolerance,
        use_kalman=cfg.use_kalman,
    )
    track = FlyTrack(roi_name=roi.name, track_id=1)

    # Opt-in illumination-drift correction (built per worker; not picklable).
    bg_state = None
    if getattr(cfg, "bg_drift_correction", False) and bg is not None:
        bg_state = BackgroundState(bg, cfg, protect_radius=cfg.bg_protect_radius_px / scale)

    frame_indices = timing["frame"].values.astype(np.int32)
    t_s_values = timing["t_s"].values.astype(np.float64)

    # With stride > 1 the decoder emits only every Nth frame; the k-th emitted
    # frame is original timing row k*stride. Skipped frames are interpolated later
    # (trajectory.resample_to_frames in the session).
    rows = range(0, len(frame_indices), stride)

    # Circular arena-disk mask so detection can't fire in the square crop's
    # corners (outside the arena). Built once, lazily, from the first frame's
    # actual shape (matches the decoded crop even when it was edge-clamped).
    want_arena_mask = getattr(cfg, "detect_arena_mask", True)
    arena_mask = None

    n_proc = 0
    _reported = 0
    _t_track = time.perf_counter()
    it = iter(frame_iter)
    for i in rows:
        gray = next(it, None)
        if gray is None:
            break
        frame_idx = int(frame_indices[i])
        t_s = float(t_s_values[i])
        n_proc += 1
        # Feed the shared frame counter in coarse batches (cheap: a few IPC calls
        # per ROI instead of one per frame) so the tracking bar advances smoothly.
        if on_frames is not None and n_proc - _reported >= 256:
            on_frames(n_proc - _reported)
            _reported = n_proc

        if bg is None:
            continue  # no background could be built → cannot track this ROI

        if want_arena_mask and arena_mask is None:
            arena_mask = roi_mask_scaled(gray.shape[:2], roi, (x0, y0), scale)

        # Drift-correct the background against the tracked object's last position
        # (opt-in); otherwise use the static background.
        bg_use = bg if bg_state is None else bg_state.corrected(gray, center=tracker.last)
        candidates = extract_ellipses(
            gray, bg_use, cfg,
            mask=arena_mask,
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
            bg_drift=(bg_state.last_drift if bg_state is not None else float("nan")),
        )
        track.observations.append(obs)

    if on_frames is not None and n_proc > _reported:
        on_frames(n_proc - _reported)   # flush the final partial batch

    track_s = time.perf_counter() - _t_track
    if return_timing:
        return track, {"bg_build_s": 0.0, "track_s": track_s, "n_frames": n_proc}
    return track


def _build_roi_gmm(samples: np.ndarray, cfg: AnyTrackConfig) -> Optional[np.ndarray]:
    """GMM background from in-memory grayscale samples (streaming path).

    A FIFO can't be seeked to sample uniformly like ``build_background`` does, so
    the streaming worker buffers its first frames and fits the (vectorized) GMM
    over them directly. Fine for a static arena; ``bg_drift_correction`` handles
    slow illumination change at run time.
    """
    if samples is None or len(samples) == 0:
        return None
    try:
        from .background import fit_gmm_background
        gmm_bg, _ = fit_gmm_background(
            samples,
            bic_improvement=cfg.gmm_bic_improvement,
            lowp=cfg.gmm_lowp,
            min_std=cfg.gmm_min_std,
            reg_covar=cfg.gmm_reg_covar,
        )
        return gmm_bg
    except Exception:
        return None


def _cv2_gray_frames(cap):
    """Yield grayscale frames from an open cv2.VideoCapture until exhausted."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def track_roi_video(
    roi_video_path: Path,
    preprocess_result: PreprocessResult,
    timing: pd.DataFrame,
    cfg: AnyTrackConfig,
    roi_background: Optional[np.ndarray] = None,
    return_timing: bool = False,
    on_frames: Optional[Callable[[int], None]] = None,
):
    """
    Track a single pre-cropped ROI **file** (mp4). Thin wrapper: resolve the
    background, then run the shared :func:`_track_from_frames` core over cv2-decoded
    grayscale frames.

    Args:
        roi_video_path: Path to the cropped ROI video
        preprocess_result: PreprocessResult with scaling info
        timing: Timing DataFrame with frame and t_s columns
        cfg: Tracking configuration
        roi_background: Optional pre-cropped background (avoids per-ROI background building)
        return_timing: If True, return (FlyTrack, {bg_build_s, track_s, n_frames}).
        on_frames: Optional callback ``on_frames(delta)`` feeding the shared frame counter.

    Returns:
        FlyTrack, or (FlyTrack, timing_dict) when return_timing=True.
    """
    # Use pre-cropped background if provided, otherwise build from the ROI video.
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

    cap = cv2.VideoCapture(str(roi_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open ROI video: {roi_video_path}")
    try:
        result = _track_from_frames(
            _cv2_gray_frames(cap), preprocess_result, timing, cfg, bg,
            on_frames=on_frames, return_timing=return_timing,
        )
    finally:
        cap.release()

    if return_timing:
        track, td = result
        td["bg_build_s"] = bg_build_s
        return track, td
    return result


def _track_roi_worker(args: tuple) -> FlyTrack:
    """Worker function for multiprocessing."""
    # A 6th element (shared frame counter + lock) is optional so older/other
    # callers passing a 5-tuple keep working.
    if len(args) == 6:
        roi_video_path, preprocess_result, timing_dict, cfg_dict, bg_data, shared = args
    else:
        roi_video_path, preprocess_result, timing_dict, cfg_dict, bg_data = args
        shared = (None, None)

    # Reconstruct objects from serializable forms
    from .config import AnyTrackConfig
    from .preprocess import PreprocessResult
    from .models import CircleROI

    cfg = AnyTrackConfig(**cfg_dict)

    # Pin OpenCV to a single internal thread per worker: N worker processes each
    # spinning OpenCV's own thread pool oversubscribes the cores. This is a
    # standalone tracking win and a prerequisite for cross-video concurrency.
    _nthreads = int(getattr(cfg, "cv_threads_per_worker", 1) or 0)
    if _nthreads > 0:
        try:
            cv2.setNumThreads(_nthreads)
        except Exception:
            pass

    timing = pd.DataFrame(timing_dict)

    # Build the frame-progress callback from the shared counter (if any). Under
    # 'spawn' these are Manager proxies; in single-worker mode they are plain
    # in-process objects. Increments are best-effort — never break tracking.
    counter, lock = shared
    on_frames = None
    if counter is not None:
        def on_frames(delta, _c=counter, _l=lock):
            try:
                with _l:
                    _c.value += int(delta)
            except Exception:
                pass

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

    return track_roi_video(Path(roi_video_path), pr, timing, cfg,
                           roi_background=roi_background, on_frames=on_frames)


def _read_exact(fh, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes from a raw pipe, or None at EOF / short frame."""
    buf = fh.read(n)
    if not buf:
        return None
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if not chunk:          # EOF mid-frame → treat as stream end
            return None
        buf += chunk
    return buf


def _track_roi_stream(fifo_path, w, h, pr, timing, cfg, roi_background, on_frames, stride=1):
    """Track one ROI by reading raw ``w×h`` gray frames from a FIFO (no cvtColor).

    Opens the FIFO by path — blocks until FFmpeg opens it for writing — then feeds
    frames to :func:`_track_from_frames`. When no background is supplied, buffers
    the first ``gmm_n_samples`` frames to fit the GMM, then tracks all frames.
    With ``stride > 1`` the FIFO already carries only every Nth frame. Always
    drains any remaining bytes so FFmpeg never blocks on a full pipe.
    """
    frame_bytes = int(w) * int(h)
    fh = open(fifo_path, "rb", buffering=0)   # blocks until the writer (ffmpeg) opens

    def _read_frame():
        raw = _read_exact(fh, frame_bytes)
        if raw is None:
            return None
        return np.frombuffer(raw, np.uint8).reshape(int(h), int(w))

    try:
        if roi_background is not None:
            bg = roi_background
            buffered = []
        else:
            buffered = []
            for _ in range(int(getattr(cfg, "gmm_n_samples", 100))):
                f = _read_frame()
                if f is None:
                    break
                buffered.append(f)
            bg = _build_roi_gmm(np.stack(buffered), cfg) if buffered else None

        def frame_iter():
            for f in buffered:      # replay the frames consumed to build the bg
                yield f
            while True:
                f = _read_frame()
                if f is None:
                    return
                yield f

        return _track_from_frames(frame_iter(), pr, timing, cfg, bg,
                                  on_frames=on_frames, stride=stride)
    finally:
        try:                        # drain leftover frames so ffmpeg can finish
            while fh.read(1 << 20):
                pass
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass


def _track_roi_stream_worker(args: tuple) -> FlyTrack:
    """Multiprocessing worker for the streaming path: reads its FIFO and tracks."""
    fifo_path, w, h, pr_dict, timing_dict, cfg_dict, bg_data, shared, stride = args

    from .config import AnyTrackConfig
    from .preprocess import PreprocessResult
    from .models import CircleROI

    cfg = AnyTrackConfig(**cfg_dict)
    _nthreads = int(getattr(cfg, "cv_threads_per_worker", 1) or 0)
    if _nthreads > 0:
        try:
            cv2.setNumThreads(_nthreads)
        except Exception:
            pass

    timing = pd.DataFrame(timing_dict)
    pr = PreprocessResult(
        roi_name=pr_dict["roi_name"],
        video_path=Path(pr_dict["video_path"]),
        original_roi=CircleROI(**pr_dict["original_roi"]),
        scale_factor=pr_dict["scale_factor"],
        crop_offset=tuple(pr_dict["crop_offset"]),
    )

    counter, lock = shared
    on_frames = None
    if counter is not None:
        def on_frames(delta, _c=counter, _l=lock):
            try:
                with _l:
                    _c.value += int(delta)
            except Exception:
                pass

    roi_background = None
    if bg_data is not None:
        bg_bytes, bg_shape, bg_dtype = bg_data
        roi_background = np.frombuffer(bg_bytes, dtype=bg_dtype).reshape(bg_shape)

    return _track_roi_stream(Path(fifo_path), w, h, pr, timing, cfg, roi_background,
                             on_frames, stride=stride)


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

    # Shared frame counter for a frame-granular tracking bar (best-effort).
    total_frames = int(len(timing) * len(preprocess_results))
    fp = _FrameProgress(progress_hook, n_workers, total_frames)

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
        worker_args.append((str(pr.video_path), pr_dict, timing_dict, cfg_dict, bg_data, fp.shared))

    if progress_hook:
        progress_hook("tracking", {
            "status": "starting",
            "n_rois": len(preprocess_results),
            "n_workers": n_workers,
        })

    # Run parallel tracking
    tracks: List[FlyTrack] = []

    try:
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
    finally:
        fp.finish(progress_hook)

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
    roi_backgrounds: Optional[Dict[str, np.ndarray]] = None,
) -> TrackingResult:
    """
    Fast tracking dispatcher. Prefers the decode-once FIFO streaming path
    (``cfg.roi_stream_decode``, needs ``os.mkfifo``); on any streaming failure it
    falls back to the proven file-based path — never worse than before.
    """
    if not video.rois:
        raise ValueError("No ROIs defined on video")
    if video.timing is None:
        raise ValueError("VideoAsset has no timing table")

    # Streaming needs one live FIFO reader per ROI simultaneously; don't spawn
    # more readers than cores (fall back to the file path for many-ROI videos).
    stream_ok = (getattr(cfg, "roi_stream_decode", True) and hasattr(os, "mkfifo")
                 and len(video.rois) <= mp.cpu_count())
    if stream_ok:
        try:
            return track_video_fast_streaming(video, cfg, progress_hook, roi_backgrounds)
        except _StreamingUnavailable as e:
            warnings.warn(f"streaming decode unavailable ({e}); using file-based ROI extraction")

    return _track_video_fast_files(video, cfg, progress_hook, cleanup, roi_backgrounds)


def _track_video_fast_files(
    video: VideoAsset,
    cfg: AnyTrackConfig,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    cleanup: bool = True,
    roi_backgrounds: Optional[Dict[str, np.ndarray]] = None,
) -> TrackingResult:
    """Original path: FFmpeg encodes N ROI mp4s, workers re-decode + track them."""
    from .preprocess import extract_roi_videos, cleanup_roi_videos

    downscale = getattr(cfg, "roi_downscale", 2)
    n_workers = getattr(cfg, "n_tracking_workers", None)
    use_hw_encode = getattr(cfg, "use_hw_encode", True)

    if progress_hook:
        progress_hook("status", {"stage": "preprocessing"})

    preprocess_results = extract_roi_videos(
        input_video=video.video_path,
        rois=video.rois,
        downscale=downscale,
        use_hw_encode=use_hw_encode,
        progress_hook=progress_hook,
        total_frames=int(getattr(video, "frame_count", 0) or 0),
    )

    if progress_hook:
        progress_hook("status", {"stage": "tracking"})

    try:
        tracks = track_parallel(
            preprocess_results=preprocess_results,
            timing=video.timing,
            cfg=cfg,
            n_workers=n_workers,
            progress_hook=progress_hook,
            roi_backgrounds=roi_backgrounds,
        )
    finally:
        if cleanup:
            cleanup_roi_videos(preprocess_results)

    return TrackingResult(video=video, tracks=tracks, background=None)


def track_video_fast_streaming(
    video: VideoAsset,
    cfg: AnyTrackConfig,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    roi_backgrounds: Optional[Dict[str, np.ndarray]] = None,
) -> TrackingResult:
    """Decode-once path: ONE FFmpeg decode pipes raw gray crops into N FIFO readers.

    Eliminates the crop encode + disk write + re-decode + per-frame cvtColor. Uses
    exactly ``len(rois)`` workers — one reader per FIFO — because FFmpeg opens all
    outputs before decoding, so every FIFO must have a live reader simultaneously
    (fewer readers would deadlock). Raises :class:`_StreamingUnavailable` on any
    failure (after cleanup) so the caller can fall back to files.
    """
    from .preprocess import build_stream_command

    rois = list(video.rois)
    downscale = getattr(cfg, "roi_downscale", 2)
    stride = max(1, int(getattr(cfg, "track_stride", 1) or 1))
    n_workers = len(rois)   # one reader per FIFO (mandatory — see docstring)

    tmpdir = Path(tempfile.mkdtemp(prefix="anytrack_stream_"))
    fifo_paths = {roi.name: tmpdir / f"roi_{roi.name}.gray" for roi in rois}
    try:
        for p in fifo_paths.values():
            os.mkfifo(p)
    except Exception as e:
        _rmtree_quiet(tmpdir)
        raise _StreamingUnavailable(f"mkfifo failed: {e!r}")

    hw = _hw_decode_flags(video.video_path, cfg)
    cmd, results, sizes = build_stream_command(
        video.video_path, rois, downscale, fifo_paths, hwaccel_flags=hw, stride=stride)

    if progress_hook:
        progress_hook("status", {"stage": "tracking"})
        progress_hook("tracking", {"status": "starting", "n_rois": len(rois), "n_workers": n_workers})

    # Only every ``stride``-th frame is decoded + tracked (rest interpolated later).
    strided = (len(video.timing) + stride - 1) // stride
    total_frames = int(strided * len(rois))
    fp = _FrameProgress(progress_hook, n_workers, total_frames)

    timing_dict = video.timing.to_dict(orient="list")
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    worker_args = []
    for roi in rois:
        pr = results[roi.name]
        pr_dict = {
            "roi_name": pr.roi_name,
            "video_path": str(pr.video_path),
            "original_roi": {"name": roi.name, "cx": roi.cx, "cy": roi.cy,
                             "r": roi.r, "n_targets": roi.n_targets},
            "scale_factor": pr.scale_factor,
            "crop_offset": list(pr.crop_offset),
        }
        bg_data = None
        if roi_backgrounds is not None and roi.name in roi_backgrounds:
            bg = roi_backgrounds[roi.name]
            bg_data = (bg.tobytes(), bg.shape, str(bg.dtype))
        sz = sizes[roi.name]
        worker_args.append((str(fifo_paths[roi.name]), sz, sz, pr_dict,
                            timing_dict, cfg_dict, bg_data, fp.shared, stride))

    proc = None
    stderr_chunks: List[str] = []
    tracks: List[FlyTrack] = []
    try:
        # stdin=DEVNULL so the ffmpeg decode can't put the terminal into raw mode
        # (its interactive 'q' handler), which would leave the shell needing `stty sane`.
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        def _drain():
            try:
                for line in proc.stderr:
                    stderr_chunks.append(line)
            except Exception:
                pass
        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()

        # One reader per FIFO, all live at once (n_workers == len(rois)).
        with mp.Pool(n_workers) as pool:
            for i, tr in enumerate(pool.imap_unordered(_track_roi_stream_worker, worker_args)):
                tracks.append(tr)
                if progress_hook:
                    progress_hook("tracking", {"status": "progress",
                                               "completed": i + 1, "total": len(worker_args)})

        proc.wait(timeout=600)
        drainer.join(timeout=1.0)
        if proc.returncode not in (0, None):
            raise _StreamingUnavailable(
                f"ffmpeg exit {proc.returncode}: {''.join(stderr_chunks)[-400:]}")
    except _StreamingUnavailable:
        raise
    except Exception as e:
        raise _StreamingUnavailable(f"streaming tracking failed: {e!r}")
    finally:
        fp.finish(progress_hook)
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        for p in fifo_paths.values():
            try:
                os.unlink(p)
            except Exception:
                pass
        _rmtree_quiet(tmpdir)

    if progress_hook:
        progress_hook("tracking", {"status": "complete", "n_tracks": len(tracks)})
    return TrackingResult(video=video, tracks=tracks, background=None)


def _rmtree_quiet(path: Path) -> None:
    try:
        for child in Path(path).iterdir():
            try:
                child.unlink()
            except Exception:
                pass
        Path(path).rmdir()
    except Exception:
        pass


def _hw_decode_flags(video_path, cfg):
    """Hardware-decode FFmpeg flags for this source (probed + cached), or []."""
    try:
        from .preprocess import hw_decode_flags
        return hw_decode_flags(video_path, cfg)
    except Exception:
        return []
