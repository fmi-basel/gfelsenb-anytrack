"""Tests for per-stage benchmarking (anytrack.benchmark.benchmark_stages)."""
from __future__ import annotations

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import VideoAsset, CircleROI
from anytrack.preprocess import check_ffmpeg_available
from anytrack.benchmark import benchmark_stages, write_stage_report


def _write_video(path, n=40, h=200, w=200):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for i in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (50 + (i % 20), 50), 5, 20, -1)  # moving blob in r0
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


@pytest.mark.skipif(not check_ffmpeg_available(), reason="ffmpeg not available")
def test_benchmark_stages_reports_tracking_fps(tmp_path):
    vid = tmp_path / "in.avi"
    n = 40
    if not _write_video(vid, n=n):
        pytest.skip("no MJPG encoder in this OpenCV build")

    timing = pd.DataFrame({"frame": np.arange(n), "t_s": np.arange(n) / 30.0})
    video = VideoAsset(
        video_path=vid, timing_csv_path=vid.with_suffix(".csv"), timing=timing,
        width=200, height=200, frame_count=n,
        rois=[CircleROI(name="r0", cx=50, cy=50, r=40),
              CircleROI(name="r1", cx=150, cy=150, r=40)],
    )
    cfg = AnyTrackConfig(gmm_n_samples=15, roi_downscale=2, use_hw_encode=False)

    st = benchmark_stages(video, cfg, n_frames=n)

    for key in ("n_rois", "n_frames", "preprocess_s", "bg_build_s_max",
                "track_s_max", "tracking_fps_per_roi",
                "video_fps_excl_preprocess", "video_fps_incl_preprocess"):
        assert key in st, f"missing key {key!r}"
    assert st["n_rois"] == 2
    assert st["n_frames"] == n
    assert st["preprocess_s"] >= 0
    assert st["tracking_fps_per_roi"] > 0          # tracking-alone throughput measured
    assert st["video_fps_incl_preprocess"] > 0

    out = tmp_path / "stages.toml"
    write_stage_report(st, out, video, cfg, git_commit="deadbee")
    text = out.read_text()
    assert "[stages]" in text
    assert "tracking_fps_per_roi" in text
    assert 'git_commit = "deadbee"' in text
