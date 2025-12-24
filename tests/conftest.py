"""
Pytest configuration and fixtures for anytrack tests.
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

import pytest
import cv2

from anytrack.config import load_config, AnyTrackConfig
from anytrack.io import load_video_asset
from anytrack.models import VideoAsset


def pytest_addoption(parser):
    """Add custom command-line options for benchmark tests."""
    parser.addoption(
        "--benchmark-frames",
        action="store",
        default=10000,
        type=int,
        help="Number of frames to process per benchmark run (default: 10000)",
    )
    parser.addoption(
        "--benchmark-runs",
        action="store",
        default=3,
        type=int,
        help="Number of benchmark runs to average (default: 3)",
    )


@pytest.fixture
def benchmark_frames(request) -> int:
    """Number of frames to process per benchmark run."""
    return request.config.getoption("--benchmark-frames")


@pytest.fixture
def benchmark_runs(request) -> int:
    """Number of benchmark runs to average."""
    return request.config.getoption("--benchmark-runs")


@pytest.fixture
def test_video_path() -> Path:
    """
    Get test video path from environment variable.

    Set ANYTRACK_TEST_VIDEO to the path of a test video file.
    """
    path_str = os.environ.get("ANYTRACK_TEST_VIDEO")
    if not path_str:
        pytest.skip(
            "ANYTRACK_TEST_VIDEO environment variable not set. "
            "Set it to the path of a test video file to run benchmark tests."
        )
    path = Path(path_str)
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    return path


@pytest.fixture
def test_timing_path() -> Path:
    """
    Get test timing CSV path from environment variable.

    Set ANYTRACK_TEST_TIMING to the path of the timing CSV file.
    If not set, defaults to video path with .csv extension.
    """
    path_str = os.environ.get("ANYTRACK_TEST_TIMING")
    if path_str:
        path = Path(path_str)
    else:
        video_str = os.environ.get("ANYTRACK_TEST_VIDEO")
        if not video_str:
            pytest.skip("ANYTRACK_TEST_VIDEO environment variable not set.")
        path = Path(video_str).with_suffix(".csv")

    if not path.exists():
        pytest.skip(f"Test timing CSV not found: {path}")
    return path


@pytest.fixture
def test_video_asset(test_video_path: Path, test_timing_path: Path) -> VideoAsset:
    """Load the test video as a VideoAsset."""
    return load_video_asset(test_video_path, test_timing_path)


@pytest.fixture
def test_config() -> AnyTrackConfig:
    """Load or create test configuration."""
    return load_config()


@pytest.fixture
def reports_dir() -> Path:
    """Get the reports directory, creating it if needed."""
    reports = Path(__file__).parent / "reports"
    reports.mkdir(exist_ok=True)
    return reports


@pytest.fixture
def machine_info() -> dict:
    """Collect machine information for benchmark reports."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
    }

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["cpu_brand"] = result.stdout.strip()
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                info["memory_gb"] = round(mem_bytes / (1024**3), 1)
        except Exception:
            pass

    return info


@pytest.fixture
def git_commit() -> Optional[str]:
    """Get current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None