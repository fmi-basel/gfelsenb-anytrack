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
    plot_timeseries,
    plot_coverage,
    plot_kinematics,
    plot_drift,
    plot_occupancy,
    render_flagged_montage,
    write_html_report,
    roi_color_map,
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


def test_compute_flags_low_contrast():
    cfg = AnyTrackConfig(qc_min_contrast=20.0)
    df = pd.DataFrame({
        "roi": ["r0"] * 3, "track_id": [0] * 3, "frame": [0, 1, 2],
        "x": [50.0] * 3, "y": [50.0] * 3, "area": [50.0] * 3,
        "contrast": [100.0, 5.0, float("nan")],
    })
    out = compute_flags(df, _video(), cfg)
    # only the mid frame (contrast 5 < 20); NaN contrast is not flagged
    assert list(out["flag_low_contrast"]) == [False, True, False]


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

    res = run_qc(video, _tracks_df(10), cfg, out_dir, overlay=True, max_frames=5,
                 show_progress=True)

    assert (out_dir / "qc_summary.json").exists()
    assert (out_dir / "qc_flags.parquet").exists()
    assert (out_dir / "qc_diagnostics.png").exists()
    assert (out_dir / "qc_timeseries.png").exists()
    assert (out_dir / "qc_coverage.png").exists()
    assert (out_dir / "qc_kinematics.png").exists()
    assert (out_dir / "qc_occupancy.png").exists()
    assert (out_dir / "qc_report.html").exists()
    assert res["report_path"].exists()
    # background PNG is built from the video and saved
    assert res["background_path"] is not None and res["background_path"].exists()
    # overlay is best-effort (depends on an mp4 encoder being present)
    if res["overlay_path"] is not None:
        assert res["overlay_path"].exists()
        assert 0 < res["overlay_frames"] <= 5


def test_plot_timeseries_all_ellipse_props(tmp_path):
    n = 12
    df = pd.DataFrame({
        "roi": ["r0"] * n + ["r1"] * n,
        "track_id": [0] * (2 * n),
        "frame": list(range(n)) * 2,
        "area": np.linspace(40, 60, 2 * n),
        "n_candidates": [1] * (2 * n),
        "major": np.linspace(10, 14, 2 * n),
        "minor": np.linspace(4, 6, 2 * n),
        "angle_deg": np.linspace(0, 180, 2 * n),
    })
    cfg = AnyTrackConfig()
    paths = plot_timeseries(df, cfg, tmp_path)
    assert paths and paths[0].exists()
    # coverage raster too
    cpaths = plot_coverage(df, _video(n=n), tmp_path)
    assert cpaths and cpaths[0].exists()


def test_plot_drift_present_and_absent(tmp_path):
    cfg = AnyTrackConfig()
    n = 10
    base = pd.DataFrame({"roi": ["r0"] * n, "track_id": [0] * n, "frame": np.arange(n),
                         "x": [10.0] * n, "y": [10.0] * n})
    assert plot_drift(base, cfg, tmp_path) == []                         # no column
    assert plot_drift(base.assign(bg_drift=[float("nan")] * n), cfg, tmp_path) == []  # all-NaN
    paths = plot_drift(base.assign(bg_drift=np.linspace(-5, 5, n)), cfg, tmp_path)
    assert paths and paths[0].exists()                                    # real drift -> figure


def test_roi_color_map_stable():
    m = roi_color_map(["b", "a", "a"])
    assert set(m) == {"a", "b"}
    assert m["a"] != m["b"]
    # deterministic (sorted): "a" gets palette[0]
    assert roi_color_map(["a", "b"]) == roi_color_map(["b", "a"])


def test_flagged_montage(tmp_path):
    video_path = tmp_path / "vid.avi"
    if not _write_video(video_path):
        pytest.skip("No MJPG encoder available")
    video = _video(w=64, h=64, n=10, path=video_path,
                   rois=[CircleROI(name="r0", cx=32, cy=32, r=31)])
    # area far out of range -> every frame flagged -> montage has tiles
    cfg = AnyTrackConfig(expected_fly_area_min=10, expected_fly_area_max=100, qc_montage_tile=48)
    df = pd.DataFrame({
        "roi": ["r0"] * 6, "track_id": [0] * 6, "frame": np.arange(6),
        "x": [32.0] * 6, "y": [32.0] * 6, "area": [9000.0] * 6,  # > max -> flag_area
    })
    out = tmp_path / "montage.png"
    path, n = render_flagged_montage(video, df, cfg, out)
    assert path is not None and path.exists()
    assert n > 0


def test_write_html_report(tmp_path):
    (tmp_path / "qc_diagnostics.png").write_bytes(b"x")  # dummy referenced file
    summary = {"n_rois": 1, "n_frames": 10,
               "per_roi": {"r0": {"frames_missing": 0, "flag_area": 2}}}
    report = write_html_report(tmp_path, summary,
                               {"diagnostics": tmp_path / "qc_diagnostics.png",
                                "overlay": None})
    assert report.exists()
    html = report.read_text()
    assert "anytrack QC report" in html
    assert "qc_diagnostics.png" in html and "r0" in html


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


def test_summarize_flags_edge_pinned_and_low_activity():
    """A ROI whose fly stays at the wall (r>0.85R) for ~all frames with ~zero
    speed is flagged edge_pinned + low_activity; an active interior ROI is not."""
    from anytrack.models import CircleROI
    cfg = AnyTrackConfig()
    n, R = 200, 100.0
    rows = []
    for f in range(n):
        rows.append(dict(roi="arena_01", track_id=1, frame=f, x=0.0, y=0.0,
                         r_center_px=0.95 * R, speed_mm_s=0.0))
        rows.append(dict(roi="arena_02", track_id=1, frame=f, x=0.0, y=0.0,
                         r_center_px=0.30 * R, speed_mm_s=2.0))
    df = pd.DataFrame(rows)
    video = SimpleNamespace(
        rois=[CircleROI(name="arena_01", cx=0, cy=0, r=R),
              CircleROI(name="arena_02", cx=0, cy=0, r=R)],
        frame_count=n, width=0, height=0)
    s = summarize(df, video, cfg)
    assert s["n_edge_pinned"] == 1 and s["n_low_activity"] == 1
    a1, a2 = s["per_roi"]["arena_01"], s["per_roi"]["arena_02"]
    assert a1["edge_pinned"] and a1["low_activity"] and a1["wall_fraction"] == 1.0
    assert not a2["edge_pinned"] and not a2["low_activity"]
