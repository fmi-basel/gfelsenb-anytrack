"""
Performance benchmark for anytrack tracking.

Runs tracking on a sample video, measures FPS over multiple runs,
and writes results to a TOML report file.

Usage:
    # Set environment variables
    export ANYTRACK_TEST_VIDEO=/path/to/test_video.avi
    export ANYTRACK_TEST_TIMING=/path/to/test_video.csv

    # Run benchmark with defaults (3 runs, 1000 frames)
    pytest tests/test_performance.py -v -s

    # Run with custom parameters
    pytest tests/test_performance.py -v -s --benchmark-frames=500 --benchmark-runs=5
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import pytest
import pandas as pd

from anytrack.config import AnyTrackConfig
from anytrack.models import VideoAsset
from anytrack.io import load_video_asset
from anytrack.roi import detect_circular_rois
from anytrack.background import build_background
from anytrack.tracking import track_video


def benchmark_tracking(
    video: VideoAsset,
    cfg: AnyTrackConfig,
    n_frames: int = 1000,
    n_runs: int = 3,
) -> dict:
    """
    Run tracking benchmark and return timing stats.

    Args:
        video: VideoAsset with ROIs already detected
        cfg: Tracking configuration
        n_frames: Number of frames to process per run
        n_runs: Number of runs to average

    Returns:
        Dictionary with benchmark results:
        {
            "n_frames": 1000,
            "n_runs": 3,
            "fps_per_run": [35.2, 37.1, 36.5],
            "fps_mean": 36.27,
            "fps_std": 0.95,
            "ms_per_frame_mean": 27.6,
            "total_time_s": 82.8,
        }
    """
    # Limit timing table to n_frames
    if video.timing is not None and len(video.timing) > n_frames:
        limited_timing = video.timing.head(n_frames).copy()
    else:
        limited_timing = video.timing
        n_frames = len(limited_timing) if limited_timing is not None else 0

    # Create a modified video asset with limited frames
    limited_video = VideoAsset(
        video_path=video.video_path,
        timing_csv_path=video.timing_csv_path,
        timing=limited_timing,
        fps_nominal=video.fps_nominal,
        frame_count=n_frames,
        width=video.width,
        height=video.height,
        rois=video.rois,
    )

    fps_per_run = []
    times_per_run = []

    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)

        start = time.perf_counter()
        _ = track_video(limited_video, cfg, progress_hook=None, cancel_event=None)
        elapsed = time.perf_counter() - start

        fps = n_frames / elapsed
        fps_per_run.append(fps)
        times_per_run.append(elapsed)

        print(f"{fps:.1f} fps ({elapsed:.2f}s)")

    fps_mean = mean(fps_per_run)
    fps_std_val = stdev(fps_per_run) if len(fps_per_run) > 1 else 0.0
    ms_per_frame = 1000.0 / fps_mean if fps_mean > 0 else 0.0
    total_time = sum(times_per_run)

    return {
        "n_frames": n_frames,
        "n_runs": n_runs,
        "fps_per_run": fps_per_run,
        "fps_mean": fps_mean,
        "fps_std": fps_std_val,
        "ms_per_frame_mean": ms_per_frame,
        "total_time_s": total_time,
    }


def write_benchmark_report(
    results: dict,
    output_path: Path,
    video: VideoAsset,
    cfg: AnyTrackConfig,
    metadata: Optional[dict] = None,
    git_commit: Optional[str] = None,
):
    """
    Write benchmark results to TOML file.

    Args:
        results: Benchmark results from benchmark_tracking()
        output_path: Path to write TOML report
        video: VideoAsset that was benchmarked
        cfg: Configuration used
        metadata: Machine info dictionary
        git_commit: Git commit hash if available
    """
    lines = []

    # Metadata section
    lines.append("[metadata]")
    lines.append(f'timestamp = "{datetime.now().isoformat()}"')
    if git_commit:
        lines.append(f'git_commit = "{git_commit}"')
    if metadata:
        if "cpu_brand" in metadata:
            lines.append(f'machine = "{metadata["cpu_brand"]}"')
        if "memory_gb" in metadata:
            lines.append(f"memory_gb = {metadata['memory_gb']}")
        lines.append(f'platform = "{metadata.get("platform", "unknown")}"')
        lines.append(f'python_version = "{metadata.get("python_version", "unknown")}"')
        lines.append(f'opencv_version = "{metadata.get("opencv_version", "unknown")}"')
    lines.append("")

    # Video section
    lines.append("[video]")
    lines.append(f'path = "{video.video_path.name}"')
    lines.append(f'resolution = "{video.width}x{video.height}"')
    lines.append(f"n_rois = {len(video.rois)}")
    lines.append(f"total_frames = {video.frame_count}")
    lines.append("")

    # Benchmark parameters section
    lines.append("[benchmark]")
    lines.append(f"n_frames = {results['n_frames']}")
    lines.append(f"n_runs = {results['n_runs']}")
    lines.append("")

    # Results section
    lines.append("[results]")
    fps_list = ", ".join(f"{x:.2f}" for x in results["fps_per_run"])
    lines.append(f"fps_per_run = [{fps_list}]")
    lines.append(f"fps_mean = {results['fps_mean']:.2f}")
    lines.append(f"fps_std = {results['fps_std']:.2f}")
    lines.append(f"ms_per_frame_mean = {results['ms_per_frame_mean']:.2f}")
    lines.append(f"total_time_s = {results['total_time_s']:.2f}")
    lines.append("")

    # Config section (relevant tracking parameters)
    lines.append("[config]")
    lines.append(f'bg_method = "{cfg.bg_method}"')
    lines.append(f'thr_method = "{cfg.thr_method}"')
    lines.append(f"thr_fixed = {cfg.thr_fixed}")
    lines.append(f"morph_open = {cfg.morph_open}")
    lines.append(f"morph_close = {cfg.morph_close}")
    lines.append(f"use_kalman = {str(cfg.use_kalman).lower()}")
    lines.append(f"max_jump_px = {cfg.max_jump_px}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def test_tracking_performance(
    test_video_path: Path,
    test_timing_path: Path,
    test_config: AnyTrackConfig,
    benchmark_frames: int,
    benchmark_runs: int,
    reports_dir: Path,
    machine_info: dict,
    git_commit: Optional[str],
):
    """
    Benchmark test that:
    1. Loads sample video
    2. Detects ROIs if needed
    3. Runs benchmark (default: 3 runs x 1000 frames)
    4. Writes TOML report to tests/reports/
    5. Prints summary to console
    """
    print("\n" + "=" * 60)
    print("ANYTRACK PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Load video
    print(f"\nLoading video: {test_video_path.name}")
    video = load_video_asset(test_video_path, test_timing_path)
    print(f"  Resolution: {video.width}x{video.height}")
    print(f"  Total frames: {video.frame_count}")

    # Detect ROIs if not present
    if not video.rois:
        print("\nDetecting ROIs...")
        bg = build_background(
            str(video.video_path),
            method=test_config.bg_method,
            k_min=max(10, test_config.bg_min_frames // 3),
            k_max=max(60, test_config.bg_min_frames),
        ).image
        video.rois = detect_circular_rois(
            bg,
            dp=test_config.roi_hough_dp,
            min_dist_ratio=test_config.roi_hough_min_dist_ratio,
            param1=test_config.roi_hough_param1,
            param2=test_config.roi_hough_param2,
            min_radius_ratio=test_config.min_radius_ratio,
            max_radius_ratio=test_config.max_radius_ratio,
        )
    print(f"  ROIs: {len(video.rois)}")

    if not video.rois:
        pytest.fail("No ROIs detected. Cannot run benchmark without ROIs.")

    # Run benchmark
    print(f"\nRunning benchmark: {benchmark_runs} runs x {benchmark_frames} frames")
    print("-" * 40)

    results = benchmark_tracking(
        video=video,
        cfg=test_config,
        n_frames=benchmark_frames,
        n_runs=benchmark_runs,
    )

    # Print summary
    print("-" * 40)
    print(f"\nRESULTS:")
    print(f"  Mean FPS: {results['fps_mean']:.2f} +/- {results['fps_std']:.2f}")
    print(f"  Mean ms/frame: {results['ms_per_frame_mean']:.2f}")
    print(f"  Total time: {results['total_time_s']:.2f}s")

    # Write report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"benchmark_{timestamp}.toml"

    write_benchmark_report(
        results=results,
        output_path=report_path,
        video=video,
        cfg=test_config,
        metadata=machine_info,
        git_commit=git_commit,
    )

    print(f"\nReport written to: {report_path}")
    print("=" * 60)

    # Assert something to make pytest happy
    assert results["fps_mean"] > 0, "FPS should be positive"