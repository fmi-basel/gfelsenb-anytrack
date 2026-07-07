"""Tests for the QC overlay crop options (--crop-roi / --follow)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.qc import _resolve_roi_bbox, render_overlay


def test_resolve_roi_bbox_uses_geometry():
    video = SimpleNamespace(rois=[CircleROI("arena_01", 100, 100, 40)])
    df = pd.DataFrame({"roi": ["arena_01"], "x": [100.0], "y": [100.0]})
    bbox = _resolve_roi_bbox("arena_01", video, df)
    assert bbox == (60.0, 60.0, 140.0, 140.0)             # cx±r


def test_resolve_roi_bbox_fallback_square_min():
    # No geometry + a near-stationary fly -> a padded square window, not a sliver.
    video = SimpleNamespace(rois=[])
    df = pd.DataFrame({"roi": ["r0"] * 3, "x": [100.0, 101.0, 100.5], "y": [50.0, 50.5, 50.0]})
    x0, y0, x1, y1 = _resolve_roi_bbox("r0", video, df)
    assert abs((x1 - x0) - (y1 - y0)) < 1e-6              # square
    assert (x1 - x0) >= 300.0                             # floored, usable
    assert _resolve_roi_bbox("missing", video, df) is None


def _write_video(path, n=6, h=200, w=200):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for _ in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (100, 100), 5, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def _tracks(n=6):
    return pd.DataFrame({"roi": ["r0"] * n, "track_id": [0] * n, "frame": range(n),
                         "x": [100.0] * n, "y": [100.0] * n, "t_s": [i / 30 for i in range(n)],
                         "area": [50.0] * n})


def test_render_overlay_crop_roi_dims(tmp_path):
    vp = tmp_path / "v.avi"
    if not _write_video(vp):
        pytest.skip("No MJPG encoder available")
    cfg = AnyTrackConfig(); cfg.qc_overlay_downscale = 1; cfg.qc_overlay_stride = 1
    video = SimpleNamespace(video_path=vp, width=200, height=200, frame_count=6,
                            fps_nominal=30.0, rois=[CircleROI("r0", 100, 100, 40)])
    p, n = render_overlay(video, _tracks(), cfg, tmp_path / "o.mp4")   # uncropped baseline
    if p is None:
        pytest.skip("No overlay encoder available")
    cap = cv2.VideoCapture(str(p)); w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
    assert w0 == 200

    # Cropped overlays scale the source region UP to crop_output_size (square),
    # independent of the source window — so annotations stay crisp.
    p2, _ = render_overlay(video, _tracks(), cfg, tmp_path / "roi.mp4",
                           crop_roi="r0", crop_output_size=160)
    cap = cv2.VideoCapture(str(p2))
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    assert (cw, ch) == (160, 160)

    p3, _ = render_overlay(video, _tracks(), cfg, tmp_path / "foll.mp4",
                           follow="r0", follow_size=64, crop_output_size=160)
    cap = cv2.VideoCapture(str(p3))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    assert (fw, fh) == (160, 160)                        # output size, not the 64px window


def _last_frame(path):
    cap = cv2.VideoCapture(str(path)); fr = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr = f
    cap.release()
    return fr


def test_overlay_trace_length_and_frame_number(tmp_path):
    """Full trace by default draws the whole path so far; --trace caps it. The
    frame-number annotation sits in the bottom-right corner."""
    N, H, W = 24, 200, 200
    xs = np.linspace(30, 170, N); ys = np.linspace(30, 170, N)   # fly walks the diagonal
    vp = tmp_path / "mv.avi"
    wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H), isColor=True)
    if not wr.isOpened():
        wr.release(); pytest.skip("No MJPG encoder available")
    for i in range(N):
        g = np.full((H, W), 200, np.uint8)
        cv2.circle(g, (int(xs[i]), int(ys[i])), 4, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    df = pd.DataFrame({"roi": ["r0"] * N, "track_id": [0] * N, "frame": range(N),
                       "x": xs, "y": ys, "t_s": [i / 30 for i in range(N)], "area": [30.0] * N})
    video = SimpleNamespace(video_path=vp, width=W, height=H, frame_count=N,
                            fps_nominal=30.0, rois=[])   # no arena circle → clean probe
    cfg = AnyTrackConfig(); cfg.qc_overlay_downscale = 1; cfg.qc_overlay_stride = 1

    def early_has_trail(fr):   # an early path point (50,50), on the diagonal
        patch = fr[45:56, 45:56].astype(int)
        return int((np.abs(patch - 200) >= 20).any(axis=2).sum()) > 5

    pf, _ = render_overlay(video, df, cfg, tmp_path / "full.mp4", trail_len=-1)
    ps, _ = render_overlay(video, df, cfg, tmp_path / "short.mp4", trail_len=3)
    if pf is None or ps is None:
        pytest.skip("No overlay encoder available")
    ff = _last_frame(pf); fs = _last_frame(ps)
    assert early_has_trail(ff)          # full trace reaches the start of the path
    assert not early_has_trail(fs)      # a 3-frame trace does not

    white = np.all(ff >= 240, axis=2)   # frame-number text
    yw, xw = np.where(white)
    assert len(xw) and xw.mean() > W / 2 and yw.mean() > H / 2   # bottom-right quadrant


def test_flag_label():
    from anytrack.qc import flag_label
    assert flag_label({"flag_jump": True, "flag_multi": True}) == "jump,multi"   # FLAG_COLUMNS order
    assert flag_label({"flag_area": True, "flag_low_contrast": True}) == "area,lowC"
    assert flag_label({}) == "" and flag_label({"flag_jump": False}) == ""


def _count_red(path):
    cap = cv2.VideoCapture(str(path)); n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += int(((f[:, :, 2] > 150) & (f[:, :, 0] < 80) & (f[:, :, 1] < 80)).sum())
    cap.release()
    return n


def test_overlay_flag_ring_and_label(tmp_path):
    """Flagged frames get a red ring + flag label; the toggle removes the label."""
    N, H, W = 8, 200, 200
    vp = tmp_path / "v.avi"
    wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H), isColor=True)
    if not wr.isOpened():
        wr.release(); pytest.skip("No MJPG encoder available")
    for _ in range(N):
        wr.write(cv2.cvtColor(np.full((H, W), 200, np.uint8), cv2.COLOR_GRAY2BGR))
    wr.release()
    base = dict(roi=["r0"] * N, track_id=[0] * N, frame=list(range(N)),
                y=[100.0] * N, t_s=[i / 30 for i in range(N)], area=[50.0] * N,
                n_candidates=[1] * N)
    df_flag = pd.DataFrame({**base, "x": [100.0, 100, 100, 100, 170, 100, 100, 100]})  # jump at f4/f5
    df_clean = pd.DataFrame({**base, "x": [100.0] * N})
    video = SimpleNamespace(video_path=vp, width=W, height=H, frame_count=N,
                            fps_nominal=30.0, rois=[])
    cfg = AnyTrackConfig(); cfg.qc_overlay_downscale = 1; cfg.qc_overlay_stride = 1

    p_flag, _ = render_overlay(video, df_flag, cfg, tmp_path / "flag.mp4")
    p_clean, _ = render_overlay(video, df_clean, cfg, tmp_path / "clean.mp4")
    if p_flag is None or p_clean is None:
        pytest.skip("No overlay encoder available")
    red_flag, red_clean = _count_red(p_flag), _count_red(p_clean)
    assert red_flag > red_clean                       # ring + label only on flagged frames

    cfg.qc_overlay_flag_labels = False                # toggle off: ring stays, label goes
    p_nolabel, _ = render_overlay(video, df_flag, cfg, tmp_path / "nolabel.mp4")
    assert _count_red(p_nolabel) < red_flag


def test_overlay_trace_defaults_to_config(tmp_path):
    """trail_len=None resolves to cfg.qc_overlay_trace (default -1 = full)."""
    N, H, W = 16, 160, 160
    xs = np.linspace(20, 140, N); ys = np.linspace(20, 140, N)
    vp = tmp_path / "cfg.avi"
    wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H), isColor=True)
    if not wr.isOpened():
        wr.release(); pytest.skip("No MJPG encoder available")
    for i in range(N):
        g = np.full((H, W), 200, np.uint8)
        cv2.circle(g, (int(xs[i]), int(ys[i])), 3, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    df = pd.DataFrame({"roi": ["r0"] * N, "track_id": [0] * N, "frame": range(N),
                       "x": xs, "y": ys, "t_s": [i / 30 for i in range(N)], "area": [20.0] * N})
    video = SimpleNamespace(video_path=vp, width=W, height=H, frame_count=N,
                            fps_nominal=30.0, rois=[])
    cfg = AnyTrackConfig(); cfg.qc_overlay_downscale = 1; cfg.qc_overlay_stride = 1
    cfg.qc_overlay_trace = 2                              # config caps the trace
    p, _ = render_overlay(video, df, cfg, tmp_path / "o.mp4")   # trail_len=None → cfg
    if p is None:
        pytest.skip("No overlay encoder available")
    fr = _last_frame(p)
    patch = fr[35:46, 35:46].astype(int)                 # early path point (40,40)
    assert int((np.abs(patch - 200) >= 20).any(axis=2).sum()) <= 5   # not reached by a 2-frame trace
