"""
Accuracy validation for anytrack tracking.

Compares tracking results against a reference trajectory to ensure
optimizations don't degrade tracking quality.

Usage:
    # First, generate reference trajectory (run once with baseline code)
    pytest tests/test_accuracy.py::test_generate_reference -v -s

    # Then, validate current implementation against reference
    pytest tests/test_accuracy.py::test_validate_accuracy -v -s
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import VideoAsset
from anytrack.io import load_video_asset
from anytrack.roi import detect_circular_rois
from anytrack.background import build_background
from anytrack.tracking import track_video


REFERENCE_FILE = Path(__file__).parent / "reference_trajectory.parquet"
REFERENCE_META = Path(__file__).parent / "reference_trajectory.json"


def run_tracking_for_validation(
    video_path: Path,
    timing_path: Path,
    cfg: AnyTrackConfig,
    n_frames: int = 500,
) -> pd.DataFrame:
    """Run tracking and return results as DataFrame."""
    video = load_video_asset(video_path, timing_path)

    # Detect ROIs if not present
    if not video.rois:
        bg = build_background(
            str(video.video_path),
            method=cfg.bg_method,
            k_min=max(10, cfg.bg_min_frames // 3),
            k_max=max(60, cfg.bg_min_frames),
        ).image
        video.rois = detect_circular_rois(
            bg,
            dp=cfg.roi_hough_dp,
            min_dist_ratio=cfg.roi_hough_min_dist_ratio,
            param1=cfg.roi_hough_param1,
            param2=cfg.roi_hough_param2,
            min_radius_ratio=cfg.min_radius_ratio,
            max_radius_ratio=cfg.max_radius_ratio,
        )

    # Limit to n_frames
    if video.timing is not None and len(video.timing) > n_frames:
        video.timing = video.timing.head(n_frames).copy()

    result = track_video(video, cfg, progress_hook=None, cancel_event=None)
    return result.to_dataframe()


def compare_trajectories(
    ref_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pos_tolerance: float = 2.0,  # pixels
    angle_tolerance: float = 10.0,  # degrees
) -> dict:
    """
    Compare two tracking DataFrames and return comparison metrics.

    Args:
        ref_df: Reference trajectory DataFrame
        test_df: Test trajectory DataFrame
        pos_tolerance: Maximum allowed position difference (pixels)
        angle_tolerance: Maximum allowed angle difference (degrees)

    Returns:
        Dictionary with comparison metrics
    """
    results = {
        "total_ref_observations": len(ref_df),
        "total_test_observations": len(test_df),
        "matching_frames": 0,
        "position_errors": [],
        "angle_errors": [],
        "missing_in_test": 0,
        "extra_in_test": 0,
    }

    if ref_df.empty or test_df.empty:
        return results

    # Compare by (roi, frame)
    ref_indexed = ref_df.set_index(["roi", "frame"])
    test_indexed = test_df.set_index(["roi", "frame"])

    common_keys = ref_indexed.index.intersection(test_indexed.index)
    results["matching_frames"] = len(common_keys)
    results["missing_in_test"] = len(ref_indexed.index.difference(test_indexed.index))
    results["extra_in_test"] = len(test_indexed.index.difference(ref_indexed.index))

    for key in common_keys:
        ref_row = ref_indexed.loc[key]
        test_row = test_indexed.loc[key]

        # Position error (Euclidean distance)
        pos_err = np.sqrt(
            (ref_row["x"] - test_row["x"]) ** 2 +
            (ref_row["y"] - test_row["y"]) ** 2
        )
        results["position_errors"].append(float(pos_err))

        # Angle error (handle wraparound)
        angle_diff = abs(ref_row["angle_deg"] - test_row["angle_deg"])
        angle_err = min(angle_diff, 360 - angle_diff)
        results["angle_errors"].append(float(angle_err))

    if results["position_errors"]:
        results["position_error_mean"] = float(np.mean(results["position_errors"]))
        results["position_error_max"] = float(np.max(results["position_errors"]))
        results["position_error_std"] = float(np.std(results["position_errors"]))
        results["positions_within_tolerance"] = sum(
            1 for e in results["position_errors"] if e <= pos_tolerance
        )
        results["position_pass_rate"] = (
            results["positions_within_tolerance"] / len(results["position_errors"])
        )

    if results["angle_errors"]:
        results["angle_error_mean"] = float(np.mean(results["angle_errors"]))
        results["angle_error_max"] = float(np.max(results["angle_errors"]))
        results["angles_within_tolerance"] = sum(
            1 for e in results["angle_errors"] if e <= angle_tolerance
        )
        results["angle_pass_rate"] = (
            results["angles_within_tolerance"] / len(results["angle_errors"])
        )

    # Remove raw error lists for cleaner output
    del results["position_errors"]
    del results["angle_errors"]

    return results


def test_generate_reference(
    test_video_path: Path,
    test_timing_path: Path,
    test_config: AnyTrackConfig,
):
    """
    Generate reference trajectory for future validation.
    Run this once with the baseline (known-good) implementation.
    """
    print("\n" + "=" * 60)
    print("GENERATING REFERENCE TRAJECTORY")
    print("=" * 60)

    print(f"\nVideo: {test_video_path.name}")
    print(f"Frames: 500")

    df = run_tracking_for_validation(
        test_video_path,
        test_timing_path,
        test_config,
        n_frames=500,
    )

    print(f"\nTracking complete:")
    print(f"  Total observations: {len(df)}")
    print(f"  ROIs: {df['roi'].nunique() if not df.empty else 0}")

    # Save reference
    df.to_parquet(REFERENCE_FILE, index=False)

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "video": test_video_path.name,
        "n_frames": 500,
        "total_observations": len(df),
        "rois": list(df["roi"].unique()) if not df.empty else [],
    }
    REFERENCE_META.write_text(json.dumps(meta, indent=2))

    print(f"\nReference saved to:")
    print(f"  {REFERENCE_FILE}")
    print(f"  {REFERENCE_META}")
    print("=" * 60)


def test_validate_accuracy(
    test_video_path: Path,
    test_timing_path: Path,
    test_config: AnyTrackConfig,
):
    """
    Validate current tracking implementation against reference trajectory.
    """
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"Reference trajectory not found: {REFERENCE_FILE}\n"
            "Run test_generate_reference first to create it."
        )

    print("\n" + "=" * 60)
    print("VALIDATING TRACKING ACCURACY")
    print("=" * 60)

    # Load reference
    ref_df = pd.read_parquet(REFERENCE_FILE)
    meta = json.loads(REFERENCE_META.read_text()) if REFERENCE_META.exists() else {}
    print(f"\nReference: {meta.get('generated_at', 'unknown')}")
    print(f"  Observations: {len(ref_df)}")

    # Run current implementation
    print("\nRunning current implementation...")
    test_df = run_tracking_for_validation(
        test_video_path,
        test_timing_path,
        test_config,
        n_frames=500,
    )
    print(f"  Observations: {len(test_df)}")

    # Compare
    print("\nComparing trajectories...")
    comparison = compare_trajectories(
        ref_df,
        test_df,
        pos_tolerance=2.0,  # 2 pixel tolerance
        angle_tolerance=10.0,  # 10 degree tolerance
    )

    print("\n" + "-" * 40)
    print("COMPARISON RESULTS:")
    print("-" * 40)
    print(f"  Reference observations: {comparison['total_ref_observations']}")
    print(f"  Test observations:      {comparison['total_test_observations']}")
    print(f"  Matching frames:        {comparison['matching_frames']}")
    print(f"  Missing in test:        {comparison['missing_in_test']}")
    print(f"  Extra in test:          {comparison['extra_in_test']}")

    if "position_error_mean" in comparison:
        print(f"\n  Position error (mean): {comparison['position_error_mean']:.3f} px")
        print(f"  Position error (max):  {comparison['position_error_max']:.3f} px")
        print(f"  Position pass rate:    {comparison['position_pass_rate']*100:.1f}%")

    if "angle_error_mean" in comparison:
        print(f"\n  Angle error (mean):    {comparison['angle_error_mean']:.2f} deg")
        print(f"  Angle error (max):     {comparison['angle_error_max']:.2f} deg")
        print(f"  Angle pass rate:       {comparison['angle_pass_rate']*100:.1f}%")

    print("=" * 60)

    # Assertions
    # Allow 5% observation count difference
    obs_diff = abs(len(test_df) - len(ref_df)) / max(len(ref_df), 1)
    assert obs_diff < 0.05, f"Observation count differs by {obs_diff*100:.1f}%"

    # At least 95% of positions within tolerance
    if "position_pass_rate" in comparison:
        assert comparison["position_pass_rate"] >= 0.95, (
            f"Position pass rate {comparison['position_pass_rate']*100:.1f}% < 95%"
        )

    # At least 90% of angles within tolerance
    if "angle_pass_rate" in comparison:
        assert comparison["angle_pass_rate"] >= 0.90, (
            f"Angle pass rate {comparison['angle_pass_rate']*100:.1f}% < 90%"
        )

    print("\n✓ Accuracy validation PASSED")