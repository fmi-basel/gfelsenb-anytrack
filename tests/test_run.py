"""Tests for the headless anytrack-run CLI (full pipeline -> custom output)."""
from __future__ import annotations

import numpy as np
import cv2
import pandas as pd
import pytest

from types import SimpleNamespace

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.preprocess import roi_crop_geometry
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


def test_resolve_output_path(tmp_path):
    cfg = AnyTrackConfig()
    video = SimpleNamespace(video_path=tmp_path / "myclip.avi")

    # A directory (no file extension) -> auto-named <stem>_tracks.parquet inside it.
    d = tmp_path / "results"
    assert run_mod._resolve_output_path(d, cfg, video) == d / "myclip_tracks.parquet"
    # Trailing-separator directory behaves the same.
    d2 = tmp_path / "out"
    assert run_mod._resolve_output_path(str(d2) + "/", cfg, video) == d2 / "myclip_tracks.parquet"
    # A file path with a recognized suffix is used verbatim.
    f = tmp_path / "custom.parquet"
    assert run_mod._resolve_output_path(f, cfg, video) == f
    fc = tmp_path / "sub" / "custom.csv"
    assert run_mod._resolve_output_path(fc, cfg, video) == fc


def test_anytrack_run_dir_output_autonames(tmp_path, monkeypatch):
    video = tmp_path / "myclip.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    _write_timing(video.with_suffix(".csv"), n=20)

    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)
    monkeypatch.setattr(run_mod, "ensure_rois", _fake_ensure)

    out_dir = tmp_path / "results"
    rc = run_mod.main(["--video", str(video), "--output", str(out_dir), "--no-fast"])
    assert rc == 0
    assert (out_dir / "myclip_tracks.parquet").exists()


def _fake_ensure(v, cfg, background=None):
    v.rois = [CircleROI(name="r0", cx=40, cy=40, r=39)]
    return v


def test_anytrack_run_qc_full(tmp_path, monkeypatch):
    video = tmp_path / "vid.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    _write_timing(video.with_suffix(".csv"), n=20)

    captured = {}
    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)
    monkeypatch.setattr(run_mod, "ensure_rois", _fake_ensure)

    # Capture the cfg passed into run_qc to confirm --qc-full set full quality.
    import anytrack.qc as qcmod
    real_run_qc = qcmod.run_qc

    def spy(video, df, cfg, out_dir, **kw):
        captured["stride"] = cfg.qc_overlay_stride
        captured["downscale"] = cfg.qc_overlay_downscale
        captured["crf"] = cfg.qc_overlay_crf
        return real_run_qc(video, df, cfg, out_dir, **kw)
    monkeypatch.setattr(qcmod, "run_qc", spy)

    out = tmp_path / "o.parquet"
    rc = run_mod.main(["--video", str(video), "--output", str(out),
                       "--no-fast", "--qc-full", "--qc-max-frames", "5"])
    assert rc == 0
    # --qc-full implies QC ran, with full-quality overlay params.
    assert captured == {"stride": 1, "downscale": 1, "crf": 16}
    assert (out.parent / f"{out.stem}_qc" / "qc_summary.json").exists()


def test_anytrack_run_writes_custom_output(tmp_path, monkeypatch):
    video = tmp_path / "vid.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    _write_timing(video.with_suffix(".csv"), n=20)

    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)

    monkeypatch.setattr(run_mod, "ensure_rois", _fake_ensure)

    out = tmp_path / "custom" / "result.parquet"
    rc = run_mod.main(["--video", str(video), "--output", str(out), "--no-fast"])

    assert rc == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert not df.empty
    for col in ("roi", "frame", "x", "y", "x_roi", "y_roi"):
        assert col in df.columns
    assert df["x"].max() - df["x"].min() > 5  # captured the rightward motion


def test_print_run_config(capsys):
    """_print_run_config prints the file path, a summary, and flags non-defaults."""
    run_mod._print_run_config(AnyTrackConfig())          # all defaults
    out = capsys.readouterr().out
    assert "config" in out.lower() and "0 non-default" in out
    assert "thr_method" in out                            # full config is listed

    run_mod._print_run_config(AnyTrackConfig(thr_method="otsu", thr_fixed=35))
    out = capsys.readouterr().out
    assert "2 non-default" in out
    assert "thr_method" in out and "otsu" in out
    assert "non-default (default 'fixed')" in out         # the override is flagged


def _fake_ensure_two(v, cfg, background=None):
    v.rois = [CircleROI(name="r0", cx=25, cy=40, r=20),
              CircleROI(name="r1", cx=55, cy=40, r=20)]
    return v


def test_anytrack_run_roi_filter(tmp_path, monkeypatch):
    """--roi restricts the run to the named arena(s); an unknown name fails cleanly."""
    video = tmp_path / "vid.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    _write_timing(video.with_suffix(".csv"), n=20)
    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)
    monkeypatch.setattr(run_mod, "ensure_rois", _fake_ensure_two)

    # Restrict to r1: only that arena is tracked and written.
    out = tmp_path / "o.parquet"
    rc = run_mod.main(["--video", str(video), "--output", str(out), "--no-fast", "--roi", "r1"])
    assert rc == 0
    df = pd.read_parquet(out)
    assert set(df["roi"].unique()) == {"r1"}

    # Unknown arena → non-zero exit and no output file.
    out2 = tmp_path / "o2.parquet"
    rc2 = run_mod.main(["--video", str(video), "--output", str(out2),
                        "--no-fast", "--roi", "arena_99"])
    assert rc2 != 0
    assert not out2.exists()


def test_save_roi_backgrounds_writes_bg_and_flymask(tmp_path):
    """_save_roi_backgrounds writes <arena>.png and a red-tinted <arena>_flymask.png."""
    out = tmp_path / "clip_tracks.parquet"
    bg = np.full((40, 40), 200, np.uint8)
    mask = np.zeros((40, 40), bool)
    mask[15:25, 15:25] = True
    bg_dir = run_mod._save_roi_backgrounds(out, {"arena_01": bg}, {"arena_01": mask}, batch=False)

    assert bg_dir == out.parent / "clip_tracks_roi_bg"
    assert (bg_dir / "arena_01.png").exists()
    fm = bg_dir / "arena_01_flymask.png"
    assert fm.exists()
    ov = cv2.imread(str(fm))                     # BGR
    b, g, r = (int(c) for c in ov[20, 20])       # masked pixel -> red-tinted
    assert r > g and r > b
    b0, g0, r0 = (int(c) for c in ov[2, 2])      # unmasked pixel -> stays gray
    assert abs(r0 - g0) <= 2 and abs(g0 - b0) <= 2


def test_save_roi_backgrounds_none(tmp_path):
    """No backgrounds built -> nothing written, returns None."""
    out = tmp_path / "clip_tracks.parquet"
    assert run_mod._save_roi_backgrounds(out, None, None, batch=True) is None
    assert not (out.parent / "clip_tracks_roi_bg").exists()


def test_save_frame_diff(tmp_path):
    """_save_frame_diff writes |last − first| as a PNG: bright where the blob started
    and ended, near-black in between and in the static background."""
    video = tmp_path / "vid.avi"
    if not _write_video(video, n=20):           # dark blob moves 20px -> 58px at y=40
        pytest.skip("No MJPG encoder available")

    path = run_mod._save_frame_diff(video, tmp_path, "clip", batch=False)
    assert path == tmp_path / "clip_framediff.png"
    diff = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert diff is not None and diff.shape == (80, 80)
    assert diff[40, 20] > 100 and diff[40, 58] > 100   # blob start & end positions moved
    assert diff[10, 10] < 10                            # untouched background ~ 0


def _fake_roi_bgs(video_path, rois, cfg, fly_masks=None, flags=None, sample_stack=None):
    """Stand-in for build_roi_backgrounds_uniform: flat backgrounds + a small mask."""
    out = {}
    for r in rois:
        _, _, _, sc = roi_crop_geometry(r, cfg.roi_downscale)
        out[r.name] = np.full((sc, sc), 200, np.uint8)
        if fly_masks is not None:
            m = np.zeros((sc, sc), bool)
            m[sc // 2 - 2:sc // 2 + 2, sc // 2 - 2:sc // 2 + 2] = True
            fly_masks[r.name] = m
        if flags is not None:
            flags[r.name] = dict(tier=0, method="p90", flag="ok")
    return out


def test_anytrack_run_bg_only(tmp_path, monkeypatch):
    """--bg-only saves the backgrounds (+ fly-mask + full-frame) and writes no tracks."""
    video = tmp_path / "vid.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")
    _write_timing(video.with_suffix(".csv"), n=20)

    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)
    monkeypatch.setattr(run_mod, "ensure_rois", _fake_ensure)
    def _fake_bg_image(*a, return_samples=False, **k):
        img = np.full((80, 80), 200, np.uint8)
        return (img, np.full((3, 80, 80), 200, np.uint8)) if return_samples else img
    monkeypatch.setattr(run_mod, "build_background_image", _fake_bg_image)
    import anytrack.preprocess as pp
    monkeypatch.setattr(pp, "build_roi_backgrounds_uniform", _fake_roi_bgs)

    out = tmp_path / "o.parquet"
    rc = run_mod.main(["--video", str(video), "--output", str(out), "--no-fast", "--bg-only"])
    assert rc == 0
    bg_dir = out.parent / "o_roi_bg"
    assert (bg_dir / "r0.png").exists()
    assert (bg_dir / "r0_flymask.png").exists()
    assert (bg_dir / "o_full_frame_bg.png").exists()
    assert (bg_dir / "o_framediff.png").exists()  # bg-only includes the frame diff
    assert not out.exists()  # bg-only stops before tracking


def test_anytrack_run_csv_output_and_missing_timing(tmp_path, monkeypatch):
    video = tmp_path / "vid.avi"
    if not _write_video(video):
        pytest.skip("No MJPG encoder available")

    monkeypatch.setattr(run_mod, "load_config", _synthetic_cfg)
    monkeypatch.setattr(run_mod, "ensure_rois", _fake_ensure)

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
