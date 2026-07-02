"""
FFmpeg-based video preprocessing for fast tracking mode.

Extracts cropped, optionally downscaled ROI sub-videos for parallel processing.
"""
from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable
import tempfile

from .models import CircleROI


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

    # Calculate crop dimensions (square around ROI)
    crop_size = int(2 * roi.r)
    x0 = int(max(0, roi.cx - roi.r))
    y0 = int(max(0, roi.cy - roi.r))

    # Build filter chain
    filters = [f"crop={crop_size}:{crop_size}:{x0}:{y0}"]

    if downscale > 1:
        scaled_size = crop_size // downscale
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
        crop_size = int(2 * roi.r)
        x0 = int(max(0, roi.cx - roi.r))
        y0 = int(max(0, roi.cy - roi.r))
        scaled_size = crop_size // downscale if downscale > 1 else crop_size

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

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_video),
        "-filter_complex", filter_complex,
        *output_args,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            # Single-pass graph failed (e.g. HW-encoder contention across 4
            # simultaneous outputs). Fall back to the proven sequential
            # extractor, preserving the encoder choice — never worse than before.
            return _extract_roi_videos_sequential(
                input_video, rois, output_dir,
                downscale=downscale,
                use_hw_encode=use_hw_encode,
            )

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out during ROI extraction")

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