"""Tests for anytrack.qc (Milestone A9 — post-hoc QC artifacts)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.qc import (
    compute_flags,
    missing_report,
    summarize,
    run_qc,
    main as qc_main,
    FLAG_COLUMNS,
)
from anytrack.models import CircleROI


def _video(w=400, h=400, n=5, path=None, rois=None):
    return SimpleNamespace(
        video_path=path, width=w, height=h, frame_count=n,
        fps_nominal=30.0, rois=rois or [],
    )


def test_compute_flags_each_kind():
    cfg = AnyTrackConfig(expected_fly_area_min=10, expected_fly_area_max=1500,
                         max_jump_px=40.0, crop_size=128)
    df = pd.DataFrame({
        "roi": ["r0"] * 5,
        "track_id": [0] * 5,
        "frame": [0, 1, 2, 3, 4],
        "x": [100.0, 100.0, 100.0, 300.0, 5.0],
        "y": [100.0, 100.0, 100.0, 100.0, 5.0],
        "area": [50.0, 5000.0, 50.0, 50.0, 50.0],
    })
    out = compute_flags(df, _video(), cfg)

    assert list(out["flag_area"]) == [False, True, False, False, False]
    # frame3: 100->300 jump=200>40 ; frame4: 300->5 jump~295>40
    assert list(out["flag_jump"]) == [False, False, False, True, True]
    # crop_size/2 = 64 ; only frame4 (x=5) is within 64px of an edge
    assert list(out["flag_crop_oob"]) == [False, False, False, False, True]
    # no n_candidates column -> multi flag always False
    assert not out["flag_multi"].any()


def test_compute_flags_multi_from_n_candidates():
    cfg = AnyTrackConfig()
    df = pd.DataFrame({
        "roi": ["r0", "r0"], "track_id": [0, 0], "frame": [0, 1],
        "x": [50.0, 50.0], "y": [50.0, 50.0], "area": [50.0, 50.0],
        "n_candidates": [1, 3],
    })
    out = compute_flags(df, _video(), cfg)
    assert list(out["flag_multi"]) == [False, True]


def test_missing_report_from_gaps():
    df = pd.DataFrame({
        "roi": ["r0"] * 4, "track_id": [0] * 4,
        "frame": [0, 1, 3, 4],  # frame 2 missing
        "x": [1.0] * 4, "y": [1.0] * 4, "area": [50.0] * 4,
    })
    rep = missing_report(df, _video(n=5))
    assert rep["r0"]["frames_present"] == 4
    assert rep["r0"]["frames_expected"] == 5
    assert rep["r0"]["frames_missing"] == 1
    assert abs(rep["r0"]["missing_fraction"] - 0.2) < 1e-9


def test_summarize_shape():
    cfg = AnyTrackConfig()
    df = pd.DataFrame({
        "roi": ["r0", "r0", "r1"], "track_id": [0, 0, 0],
        "frame": [0, 1, 0], "x": [10.0, 10.0, 10.0], "y": [10.0, 10.0, 10.0],
        "area": [50.0, 50.0, 50.0], "speed_mm_s": [0.0, 1.0, 0.5],
    })
    s = summarize(df, _video(n=2), cfg)
    assert s["n_rois"] == 2
    assert set(s["per_roi"]) == {"r0", "r1"}
    for stats in s["per_roi"].values():
        for c in FLAG_COLUMNS:
            assert c in stats
        assert "missing_fraction" in stats
        assert "speed_mm_s_mean" in stats


def _write_video(path, n=10, h=64, w=64):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for _ in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (32, 32), 5, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def _tracks_df(n=10):
    return pd.DataFrame({
        "roi": ["r0"] * n, "track_id": [0] * n, "frame": np.arange(n),
        "x": [32.0] * n, "y": [32.0] * n, "area": [50.0] * n,
        "speed_mm_s": np.linspace(0, 2, n),
    })


def test_run_qc_writes_artifacts(tmp_path):
    video_path = tmp_path / "vid.avi"
    if not _write_video(video_path):
        pytest.skip("No MJPG encoder available")

    video = _video(w=64, h=64, n=10, path=video_path,
                   rois=[CircleROI(name="r0", cx=32, cy=32, r=31)])
    cfg = AnyTrackConfig(crop_size=16)
    out_dir = tmp_path / "qc"

    res = run_qc(video, _tracks_df(10), cfg, out_dir, overlay=True, max_frames=5)

    assert (out_dir / "qc_summary.json").exists()
    assert (out_dir / "qc_flags.parquet").exists()
    assert (out_dir / "qc_diagnostics.png").exists()
    # overlay is best-effort (depends on an mp4 encoder being present)
    if res["overlay_path"] is not None:
        assert res["overlay_path"].exists()
        assert 0 < res["overlay_frames"] <= 5


def test_qc_cli(tmp_path):
    video_path = tmp_path / "vid.avi"
    if not _write_video(video_path):
        pytest.skip("No MJPG encoder available")
    tracks = tmp_path / "tracks.parquet"
    _tracks_df(8).to_parquet(tracks, index=False)
    out_dir = tmp_path / "cli_qc"

    rc = qc_main([
        "--video", str(video_path),
        "--tracks", str(tracks),
        "--out-dir", str(out_dir),
        "--no-overlay",
    ])
    assert rc == 0
    assert (out_dir / "qc_summary.json").exists()
    assert (out_dir / "qc_flags.parquet").exists()
    assert (out_dir / "qc_diagnostics.png").exists()
