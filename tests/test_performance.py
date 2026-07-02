"""
Performance benchmark tests.

Thin wrapper over ``anytrack.benchmark`` (the benchmarking logic now lives in
the package so it is reusable from the CLI). Requires ANYTRACK_TEST_VIDEO /
ANYTRACK_TEST_TIMING (fixtures skip when unset).

Usage:
    export ANYTRACK_TEST_VIDEO=/path/to/test_video.avi
    export ANYTRACK_TEST_TIMING=/path/to/test_video.csv
    pytest tests/test_performance.py -v -s
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

from anytrack.config import AnyTrackConfig
from anytrack.io import load_video_asset
from anytrack.preprocess import check_ffmpeg_available
from anytrack.benchmark import benchmark_tracking, write_benchmark_report, ensure_rois


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
    """Legacy single-pass benchmark; writes a TOML report to tests/reports/."""
    video = load_video_asset(test_video_path, test_timing_path)
    ensure_rois(video, test_config)
    if not video.rois:
        pytest.fail("No ROIs detected. Cannot run benchmark without ROIs.")

    results = benchmark_tracking(
        video=video,
        cfg=test_config,
        n_frames=benchmark_frames,
        n_runs=benchmark_runs,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_benchmark_report(
        results=results,
        output_path=reports_dir / f"benchmark_{timestamp}.toml",
        video=video,
        cfg=test_config,
        metadata=machine_info,
        git_commit=git_commit,
    )

    assert results["fps_mean"] > 0, "FPS should be positive"


def test_tracking_performance_fast(
    test_video_path: Path,
    test_timing_path: Path,
    test_config: AnyTrackConfig,
    benchmark_frames: int,
    benchmark_runs: int,
    reports_dir: Path,
    machine_info: dict,
    git_commit: Optional[str],
):
    """Fast mode benchmark (FFmpeg preprocessing + parallel ROI tracking)."""
    if not check_ffmpeg_available():
        pytest.skip("FFmpeg not available. Install FFmpeg to run fast mode benchmark.")

    video = load_video_asset(test_video_path, test_timing_path)
    ensure_rois(video, test_config)
    if not video.rois:
        pytest.fail("No ROIs detected. Cannot run benchmark without ROIs.")

    test_config.fast_mode = True
    test_config.roi_downscale = 2
    test_config.n_tracking_workers = 4

    results = benchmark_tracking(
        video=video,
        cfg=test_config,
        n_frames=benchmark_frames,
        n_runs=benchmark_runs,
        use_fast_mode=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_benchmark_report(
        results=results,
        output_path=reports_dir / f"benchmark_fast_{timestamp}.toml",
        video=video,
        cfg=test_config,
        metadata=machine_info,
        git_commit=git_commit,
    )

    assert results["fps_mean"] > 0, "FPS should be positive"
