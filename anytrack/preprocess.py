"""
FFmpeg-based video preprocessing for fast tracking mode.

Extracts cropped, optionally downscaled ROI sub-videos for parallel processing.
"""
from __future__ import annotations

import subprocess
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import tempfile

from .models import CircleROI


def _run_ffmpeg_with_progress(
    cmd: List[str],
    total_frames: int,
    progress_hook: Optional[Callable[[str, dict], None]],
    timeout: float = 600.0,
) -> Tuple[int, str]:
    """Run an FFmpeg command, streaming ``frame=`` progress as frame events.

    The command must include ``-progress pipe:1`` so FFmpeg writes ``frame=N``
    lines to stdout as it decodes; each is surfaced as
    ``("frames", {"stage": "preprocess", "n", "total"})``. stderr is drained on a
    side thread (so a full pipe never deadlocks) and returned for error
    reporting. A watchdog kills the process after ``timeout`` seconds, letting
    the caller fall back exactly as the old blocking path did on a nonzero code.

    Returns ``(returncode, stderr_text)``.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_chunks: List[str] = []

    def _drain() -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                stderr_chunks.append(line)
        except Exception:
            pass

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.daemon = True
    watchdog.start()

    total = int(total_frames) if total_frames else 0
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line.startswith("frame=") and progress_hook is not None:
                try:
                    n = int(line.split("=", 1)[1].strip())
                    progress_hook("frames", {"stage": "preprocess", "n": n, "total": total})
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        proc.wait()
        watchdog.cancel()
        drainer.join(timeout=1.0)

    return proc.returncode, "".join(stderr_chunks)


@dataclass
class PreprocessResult:
    """Result of ROI video extraction."""
    roi_name: str
    video_path: Path
    original_roi: CircleROI
    scale_factor: float  # To map coordinates back (1.0 = no scaling)
    crop_offset: tuple[int, int]  # (x0, y0) in original frame


def get_ffmpeg_path() -> str:
    """Get FFmpeg executable path."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg to use fast mode:\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )
    return ffmpeg


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def get_ffprobe_path() -> Optional[str]:
    """Path to ffprobe, or None if not installed."""
    return shutil.which("ffprobe")


def _probe_codec(video_path) -> Optional[str]:
    """Video stream codec name via ffprobe (e.g. 'h264', 'mpeg4'), or None."""
    ffprobe = get_ffprobe_path()
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=np=1:nq=1",
             str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        return (out.stdout or "").strip() or None
    except Exception:
        return None


# Cache the hardware-decode decision per (backend, codec): whether -hwaccel
# actually initializes on this machine for this codec. Probed once (cheap).
_HW_DECODE_CACHE: Dict[tuple, list] = {}


def _hwaccel_probe_ok(video_path, flags) -> bool:
    """Decode one frame with ``flags`` to confirm the hwaccel initializes."""
    try:
        r = subprocess.run(
            [get_ffmpeg_path(), "-v", "error", *flags, "-i", str(video_path),
             "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def hw_decode_flags(video_path, cfg) -> list:
    """FFmpeg input flags for hardware decode of this source, or ``[]``.

    Honors ``cfg.use_hw_decode`` / ``cfg.hw_decode_backend`` (auto|videotoolbox|
    none). The decision is probed once per (backend, codec) by decoding a single
    frame and cached — if the hwaccel can't initialize (e.g. VideoToolbox has no
    mpeg4 decoder), we return ``[]`` and FFmpeg decodes in software as before.
    Passing ``-hwaccel`` is harmless when unsupported (FFmpeg silently uses SW),
    but probing avoids relying on that and lets callers know the true state.
    """
    if not getattr(cfg, "use_hw_decode", True):
        return []
    backend = getattr(cfg, "hw_decode_backend", "auto") or "auto"
    if backend == "none":
        return []
    if backend == "auto":
        backend = "videotoolbox"
    key = (backend, _probe_codec(video_path))
    if key not in _HW_DECODE_CACHE:
        flags = ["-hwaccel", backend]
        _HW_DECODE_CACHE[key] = flags if _hwaccel_probe_ok(video_path, flags) else []
    return list(_HW_DECODE_CACHE[key])


def roi_crop_geometry(roi: CircleROI, downscale: int) -> Tuple[int, int, int, int]:
    """Square crop + downscaled size for one ROI — shared by every extract path.

    Returns ``(crop_size, x0, y0, scaled_size)``. ``crop_size`` is rounded down to
    a multiple of ``2*downscale`` so the downscaled output has EVEN dimensions
    (libx264/yuv420p rejects odd width/height, which would fail the encode).
    ``scaled_size == crop_size`` when ``downscale == 1``. Cropping the full-frame
    background with the same geometry (Phase 5) reproduces a worker's ROI
    background exactly.
    """
    crop_size = int(2 * roi.r)
    crop_size -= crop_size % (2 * downscale)
    x0 = int(max(0, roi.cx - roi.r))
    y0 = int(max(0, roi.cy - roi.r))
    scaled_size = crop_size // downscale if downscale > 1 else crop_size
    return crop_size, x0, y0, scaled_size


def crop_roi_backgrounds(bg_full, rois, downscale):
    """Per-ROI backgrounds cropped from a full-frame background (Phase 5 reuse).

    Uses the same :func:`roi_crop_geometry` as extraction so each crop lines up
    with the tracker's frames, letting workers skip rebuilding a GMM and — for the
    streaming path — giving a uniformly-sampled background instead of the
    first-N-frames one. Returns ``{roi_name: uint8 (scaled_size, scaled_size)}``.
    """
    import numpy as np
    import cv2

    out = {}
    H, W = bg_full.shape[:2]
    for roi in rois:
        crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)
        crop = bg_full[y0:min(y0 + crop_size, H), x0:min(x0 + crop_size, W)]
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:  # ROI off-frame (rare)
            padded = np.zeros((crop_size, crop_size), dtype=bg_full.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
        if scaled_size != crop_size:
            crop = cv2.resize(crop, (scaled_size, scaled_size), interpolation=cv2.INTER_AREA)
        out[roi.name] = np.ascontiguousarray(crop)
    return out


def build_stream_command(input_video, rois, downscale, fifo_paths, hwaccel_flags=None):
    """Build the decode-once FFmpeg command for the streaming path.

    ONE decode of the source, split into N streams, each cropped+scaled+``gray``
    and written as raw gray video to its FIFO (``fifo_paths[roi.name]``). Returns
    ``(cmd, results, sizes)`` where ``results`` maps roi_name → PreprocessResult
    (video_path = the FIFO) and ``sizes`` maps roi_name → scaled_size (the raw
    gray frame is ``scaled_size × scaled_size`` bytes). ``hwaccel_flags`` (e.g.
    ``["-hwaccel", "videotoolbox"]``) are inserted before ``-i``.
    """
    ffmpeg = get_ffmpeg_path()
    filter_parts: List[str] = []
    output_args: List[str] = []
    results: Dict[str, PreprocessResult] = {}
    sizes: Dict[str, int] = {}

    for i, roi in enumerate(rois):
        crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)
        fstr = f"[v{i}]crop={crop_size}:{crop_size}:{x0}:{y0}"
        if downscale > 1:
            fstr += f",scale={scaled_size}:{scaled_size}"
        fstr += f",format=gray[out{i}]"
        filter_parts.append(fstr)
        output_args.extend([
            "-map", f"[out{i}]", "-f", "rawvideo", "-pix_fmt", "gray",
            str(fifo_paths[roi.name]),
        ])
        results[roi.name] = PreprocessResult(
            roi_name=roi.name,
            video_path=Path(fifo_paths[roi.name]),
            original_roi=roi,
            scale_factor=float(downscale),
            crop_offset=(x0, y0),
        )
        sizes[roi.name] = scaled_size

    split = f"[0:v]split={len(rois)}" + "".join(f"[v{i}]" for i in range(len(rois)))
    filter_complex = ";".join([split, *filter_parts])
    cmd = [
        ffmpeg, "-y", "-nostats",
        *(hwaccel_flags or []),
        "-i", str(input_video),
        "-filter_complex", filter_complex,
        *output_args,
    ]
    return cmd, results, sizes


def extract_roi_video(
    input_video: Path,
    roi: CircleROI,
    output_path: Path,
    downscale: int = 2,
    use_hw_encode: bool = True,
    crf: int = 18,
) -> PreprocessResult:
    """
    Extract a single ROI sub-video using FFmpeg.

    Args:
        input_video: Path to source video
        roi: CircleROI to extract
        output_path: Path for output video
        downscale: Downscale factor (1, 2, or 4)
        use_hw_encode: Try hardware encoding (VideoToolbox on macOS)
        crf: Constant rate factor (quality, lower = better, 18 is visually lossless)

    Returns:
        PreprocessResult with extraction details
    """
    ffmpeg = get_ffmpeg_path()

    # Square crop around the ROI (even downscaled dims for libx264).
    crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)

    # Build filter chain
    filters = [f"crop={crop_size}:{crop_size}:{x0}:{y0}"]

    if downscale > 1:
        filters.append(f"scale={scaled_size}:{scaled_size}")

    filter_chain = ",".join(filters)

    # Choose encoder
    if use_hw_encode:
        # Try VideoToolbox on macOS
        encoder = "h264_videotoolbox"
        encoder_opts = ["-q:v", "50"]  # Quality 0-100
    else:
        encoder = "libx264"
        encoder_opts = ["-preset", "ultrafast", "-crf", str(crf)]

    # Build command
    cmd = [
        ffmpeg,
        "-y",  # Overwrite output
        "-i", str(input_video),
        "-vf", filter_chain,
        "-c:v", encoder,
        *encoder_opts,
        "-an",  # No audio
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            # If hardware encoder failed, fall back to software
            if use_hw_encode and "videotoolbox" in result.stderr.lower():
                return extract_roi_video(
                    input_video, roi, output_path,
                    downscale=downscale,
                    use_hw_encode=False,
                    crf=crf,
                )
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out during ROI extraction")

    return PreprocessResult(
        roi_name=roi.name,
        video_path=output_path,
        original_roi=roi,
        scale_factor=float(downscale),
        crop_offset=(x0, y0),
    )


def _extract_roi_videos_single_pass(
    input_video: Path,
    rois: List[CircleROI],
    output_dir: Path,
    downscale: int = 2,
    use_hw_encode: bool = True,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    total_frames: int = 0,
) -> Dict[str, PreprocessResult]:
    """
    Extract all ROI videos in a single FFmpeg pass.

    This decodes the input video once and writes all outputs simultaneously,
    which is faster than decoding multiple times.
    """
    ffmpeg = get_ffmpeg_path()

    # Build filter_complex for all ROIs
    filter_parts = []
    output_args = []
    results: Dict[str, PreprocessResult] = {}

    for i, roi in enumerate(rois):
        crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)

        # Filter for this ROI: [vN]crop=...,scale=...[outN]  (fed by split, below)
        filter_str = f"[v{i}]crop={crop_size}:{crop_size}:{x0}:{y0}"
        if downscale > 1:
            filter_str += f",scale={scaled_size}:{scaled_size}"
        filter_str += f"[out{i}]"
        filter_parts.append(filter_str)

        # Output args for this ROI
        output_path = output_dir / f"roi_{roi.name}.mp4"

        # Single-pass MUST use software encode: 4 simultaneous VideoToolbox
        # sessions serialize on the HW encoder (~1 fps). libx264 ultrafast across
        # cores is ~2x faster than the sequential-HW path on this hardware.
        encoder = "libx264"
        encoder_opts = ["-preset", "ultrafast", "-crf", "18"]

        output_args.extend([
            "-map", f"[out{i}]",
            "-c:v", encoder,
            *encoder_opts,
            "-an",
            str(output_path),
        ])

        results[roi.name] = PreprocessResult(
            roi_name=roi.name,
            video_path=output_path,
            original_roi=roi,
            scale_factor=float(downscale),
            crop_offset=(x0, y0),
        )

    # Decode the source ONCE, split it into N identical streams, crop each —
    # this replaces N separate full-video decode passes with a single decode.
    split = f"[0:v]split={len(rois)}" + "".join(f"[v{i}]" for i in range(len(rois)))
    filter_complex = ";".join([split, *filter_parts])

    # -progress pipe:1 streams `frame=N` to stdout so we can draw a decode bar;
    # -nostats silences the (now redundant) stderr progress stats.
    cmd = [
        ffmpeg,
        "-y",
        "-nostats",
        "-progress", "pipe:1",
        "-i", str(input_video),
        "-filter_complex", filter_complex,
        *output_args,
    ]

    returncode, stderr = _run_ffmpeg_with_progress(cmd, total_frames, progress_hook)
    if returncode != 0:
        # Single-pass graph failed (e.g. HW-encoder contention across 4
        # simultaneous outputs) or was killed by the watchdog. Fall back to the
        # proven sequential extractor, preserving the encoder choice — never
        # worse than before.
        return _extract_roi_videos_sequential(
            input_video, rois, output_dir,
            downscale=downscale,
            use_hw_encode=use_hw_encode,
            progress_hook=progress_hook,
        )

    return results


def _extract_roi_videos_sequential(
    input_video: Path,
    rois: List[CircleROI],
    output_dir: Path,
    downscale: int = 2,
    use_hw_encode: bool = True,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
) -> Dict[str, PreprocessResult]:
    """Sequential extraction fallback."""
    results: Dict[str, PreprocessResult] = {}

    for i, roi in enumerate(rois):
        if progress_hook:
            progress_hook("preprocessing", {
                "roi": roi.name,
                "index": i,
                "total": len(rois),
                "percent": float(i) / len(rois),
            })

        output_path = output_dir / f"roi_{roi.name}.mp4"

        result = extract_roi_video(
            input_video=input_video,
            roi=roi,
            output_path=output_path,
            downscale=downscale,
            use_hw_encode=use_hw_encode,
        )

        results[roi.name] = result

    return results


def extract_roi_videos(
    input_video: Path,
    rois: List[CircleROI],
    output_dir: Optional[Path] = None,
    downscale: int = 2,
    use_hw_encode: bool = True,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    total_frames: int = 0,
) -> Dict[str, PreprocessResult]:
    """
    Extract all ROI sub-videos using FFmpeg.

    Args:
        input_video: Path to source video
        rois: List of CircleROIs to extract
        output_dir: Directory for output videos (uses temp dir if None)
        downscale: Downscale factor (1, 2, or 4)
        use_hw_encode: Try hardware encoding
        progress_hook: Optional callback for progress updates

    Returns:
        Dictionary mapping roi_name -> PreprocessResult
    """
    if not rois:
        raise ValueError("No ROIs provided for extraction")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="anytrack_roi_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single-pass: decode the source ONCE and crop all ROIs (falls back to the
    # sequential per-ROI extractor if the single FFmpeg graph fails).
    if progress_hook:
        progress_hook("preprocessing",
                      {"roi": "single-pass", "index": 0, "total": 1, "percent": 0.0})
    results = _extract_roi_videos_single_pass(
        input_video=input_video,
        rois=rois,
        output_dir=output_dir,
        downscale=downscale,
        use_hw_encode=use_hw_encode,
        progress_hook=progress_hook,
        total_frames=total_frames,
    )

    if progress_hook:
        progress_hook("preprocessing", {
            "roi": "complete",
            "index": len(rois),
            "total": len(rois),
            "percent": 1.0,
        })

    return results


def cleanup_roi_videos(results: Dict[str, PreprocessResult]) -> None:
    """
    Remove extracted ROI video files.

    Args:
        results: Dictionary of PreprocessResults to clean up
    """
    for result in results.values():
        try:
            if result.video_path.exists():
                result.video_path.unlink()
        except OSError:
            pass  # Ignore cleanup errors

    # Try to remove parent directory if empty
    if results:
        first_result = next(iter(results.values()))
        parent = first_result.video_path.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def estimate_extraction_time(
    video_path: Path,
    n_rois: int,
    downscale: int = 2,
) -> float:
    """
    Estimate extraction time in seconds.

    This is a rough estimate based on typical FFmpeg performance.
    Actual time varies based on hardware and video codec.

    Args:
        video_path: Path to source video
        n_rois: Number of ROIs to extract
        downscale: Downscale factor

    Returns:
        Estimated time in seconds
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    duration_s = frame_count / fps

    # Rough estimate: 2x realtime per ROI with hardware encode
    # Slower with software encode or larger downscale
    base_factor = 0.5 if downscale >= 2 else 1.0

    return duration_s * n_rois * base_factor