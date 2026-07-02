"""Tests for the headless anytrack-run CLI (full pipeline -> custom output)."""
from __future__ import annotations

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
import anytrack.run as run_mod


def _write_video(path, n=20, h=80, w=80):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for i in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (20 + i * 2, 40), 5, 20, -1)  # dark blob moving right
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def _write_timing(path, n=20, fps=30.0):
    pd.DataFrame({"frame": np.arange(n), "dt_s": [1.0 / fps] * n}).to_csv(path, index=False)


def _synthetic_cfg():
    # Mirrors the detector-test config so the synthetic dark blob is tracked
    # deterministically, independent of the user's saved config.
    return AnyTrackConfig(
        fast_mode=False,
        gmm_n_samples=15,
        arena_detection_enabled=False,
        bgdiff_type="dark",
        thr_method="fixed",
        thr_fixed=50,
        morph_open=3,
        morph_close=5,
        max_centroids_per_roi=1,
        expected_fly_area_min=30,
        expected_fly_area_max=4000,
        max_jump_px=40.0,
        miss_tolerance=15,
        use_kalman=True,
    )


def test_anytrack_run_writes_custom_output(tmp_path, monkeypatch):
    video = tmp_path / "vid.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    _write_timing(video.with_suffix(".csv"), n=20)

    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)

    def fake_ensure(v, cfg):  # inject a known ROI; detection is covered elsewhere
        v.rois = [CircleROI(name="r0", cx=40, cy=40, r=39)]
        return v
    monkeypatch.setattr(run_mod, "ensure_rois", fake_ensure)

    out = tmp_path / "custom" / "result.parquet"
    rc = run_mod.main(["--video", str(video), "--output", str(out), "--no-fast"])

    assert rc == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert not df.empty
    for col in ("roi", "frame", "x", "y", "x_roi", "y_roi"):
        assert col in df.columns
    assert df["x"].max() - df["x"].min() > 5  # captured the rightward motion


def test_anytrack_run_csv_output_and_missing_timing(tmp_path, monkeypatch):
    video = tmp_path / "vid.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")

    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)
    monkeypatch.setattr(run_mod, "ensure_rois",
                        lambda v, cfg: (setattr(v, "rois", [CircleROI(name="r0", cx=40, cy=40, r=39)]), v)[1])

    # No timing CSV next to the video -> argparse error (SystemExit).
    with pytest.raises(SystemExit):
        run_mod.main(["--video", str(video), "--output", str(tmp_path / "o.csv")])

    # Provide timing -> CSV output honored by suffix.
    _write_timing(video.with_suffix(".csv"), n=20)
    out = tmp_path / "o.csv"
    rc = run_mod.main(["--video", str(video), "--output", str(out), "--no-fast"])
    assert rc == 0
    assert out.exists()
    assert not pd.read_csv(out).empty
