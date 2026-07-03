"""Tests for Milestone B2: frame sampling, ellipse seeding, label store, export."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.pose.skeleton import DEFAULT_SKELETON
from anytrack.pose.labeling import (
    DEFAULT_NODE_COLORS, Keypoint, LabelStore, contrast_fg, crop_origin,
    crop_to_display, display_to_crop, extract_label_crops, parse_node_colors,
    resolve_node_colors, sample_frames, seed_from_ellipse,
)

_HAVE_SLEAP_IO = importlib.util.find_spec("sleap_io") is not None


def _tracks(n_per_roi=60, rois=("arena_01", "arena_02")):
    rows = []
    for r_i, roi in enumerate(rois):
        for f in range(n_per_roi):
            rows.append({
                "roi": roi, "track_id": r_i, "frame": f,
                "t_s": f / 30.0,
                "x": 100.0 + r_i * 200 + (f % 5),
                "y": 100.0 + (f % 7),
                "angle_deg": (f * 13) % 180,          # span the angle bins
                "major": 30.0 + (f % 4) * 10,
                "minor": 15.0,
                "area": 80.0 + (f % 3) * 100,          # span area bins
            })
    return pd.DataFrame(rows)


# ---- sampling ---------------------------------------------------------------

def test_sample_frames_count_and_determinism():
    df = _tracks()
    a = sample_frames(df, n=20, strategy="diversity", seed=7)
    b = sample_frames(df, n=20, strategy="diversity", seed=7)
    assert len(a) <= 20
    assert a[["roi", "frame"]].equals(b[["roi", "frame"]])       # deterministic


def test_sample_frames_per_roi_balance():
    df = _tracks(n_per_roi=60, rois=("arena_01", "arena_02"))
    s = sample_frames(df, n=20, strategy="uniform", per_roi=True, seed=0)
    counts = s["roi"].value_counts()
    assert set(counts.index) == {"arena_01", "arena_02"}         # both arenas present
    assert counts.min() >= 5                                     # roughly balanced


def test_sample_frames_diversity_spans_angles():
    df = _tracks()
    s = sample_frames(df, n=24, strategy="diversity", seed=1)
    n_angle_bins = np.unique((s["angle_deg"].to_numpy() / 30).astype(int)).size
    assert n_angle_bins >= 3                                     # spread across pose space


def test_sample_frames_empty():
    assert sample_frames(pd.DataFrame(), n=10).empty


# ---- seeding ----------------------------------------------------------------

def test_seed_from_ellipse_geometry():
    row = pd.Series({"x": 100.0, "y": 100.0, "angle_deg": 0.0, "major": 40.0, "minor": 20.0})
    seeds = seed_from_ellipse(row, DEFAULT_SKELETON, crop_size=96)

    assert set(seeds) == set(DEFAULT_SKELETON.nodes)
    x0, y0 = crop_origin(100.0, 100.0, 96)
    thorax = seeds["thorax"]
    assert abs(thorax.x - (100 - x0)) < 1e-6 and abs(thorax.y - (100 - y0)) < 1e-6
    # angle 0 -> head/abdomen split along +/- x, same y as thorax
    assert seeds["head"].x > thorax.x > seeds["abdomen_tip"].x
    assert abs(seeds["head"].y - thorax.y) < 1e-6
    for kp in seeds.values():                                    # all in-bounds
        assert 0.0 <= kp.x <= 96 and 0.0 <= kp.y <= 96


def test_seed_handles_missing_ellipse_fields():
    seeds = seed_from_ellipse(pd.Series({"x": 50.0, "y": 50.0}), DEFAULT_SKELETON, 96)
    assert set(seeds) == set(DEFAULT_SKELETON.nodes)             # no KeyError on NaN/missing


# ---- conversions ------------------------------------------------------------

def test_display_crop_roundtrip():
    xc, yc = display_to_crop(*crop_to_display(12.5, 30.0, 6.0), 6.0)
    assert abs(xc - 12.5) < 1e-9 and abs(yc - 30.0) < 1e-9


# ---- node colors ------------------------------------------------------------

def test_parse_node_colors():
    m = parse_node_colors("head:#ff0000, thorax:00ff00 ,bad, :#111, x:")
    assert m == {"head": "#ff0000", "thorax": "#00ff00"}       # tolerant of junk + missing #


def test_resolve_node_colors_precedence():
    nodes = ["head", "thorax", "mystery"]
    m = resolve_node_colors(nodes, "head:#123456")
    assert m["head"] == "#123456"                              # config override wins
    assert m["thorax"] == DEFAULT_NODE_COLORS["thorax"]        # built-in default
    assert m["mystery"].startswith("#")                        # fallback cycle, always set


def test_resolve_node_colors_default_spec_covers_fly5():
    from anytrack.config import AnyTrackConfig
    m = resolve_node_colors(list(DEFAULT_NODE_COLORS), AnyTrackConfig().pose_node_colors)
    assert m == DEFAULT_NODE_COLORS                            # config default == built-in


def test_contrast_fg():
    assert contrast_fg("#ffd740") == "#000000"                 # bright -> black text
    assert contrast_fg("#ff5252") == "#ffffff"                 # dark-ish red -> white text
    assert contrast_fg("garbage") == "#ffffff"                 # never raises


# ---- store ------------------------------------------------------------------

def test_store_roundtrip_and_manifest(tmp_path):
    df = _tracks()
    store = LabelStore.from_tracks("vid.avi", df, crop_size=96, n=8, seed=3)
    assert len(store) <= 8 and store.crop_size == 96

    fl = store.frames[0]
    fl.points["thorax"] = Keypoint(48.0, 48.0, visible=True, score=1.0)
    fl.labeled = True

    p = store.save(tmp_path / "labels.json")
    loaded = LabelStore.load(p)
    assert len(loaded) == len(store)
    assert loaded.skeleton.nodes == store.skeleton.nodes
    assert loaded.frames[0].labeled and "thorax" in loaded.frames[0].points

    man = loaded.to_manifest_df(only_labeled=True)
    assert not man.empty and (man["frame"] == fl.frame).all()
    x0, y0 = fl.origin(96)
    trow = man[man["keypoint"] == "thorax"].iloc[0]
    assert abs(trow["x_full"] - (48.0 + x0)) < 1e-6              # crop -> full mapping
    assert abs(trow["x_crop"] - 48.0) < 1e-6


def test_manifest_only_labeled_filters_unlabeled():
    df = _tracks()
    store = LabelStore.from_tracks("vid.avi", df, crop_size=96, n=6, seed=0)
    assert store.to_manifest_df(only_labeled=True).empty        # nothing confirmed yet
    assert not store.to_manifest_df(only_labeled=False).empty   # seeds still exported


# ---- crop extraction (needs an encoder) -------------------------------------

def _write_video(path, n=12, h=200, w=200):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for _ in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (100, 100), 6, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def test_extract_label_crops(tmp_path):
    vp = tmp_path / "vid.avi"
    if not _write_video(vp, n=12):
        pytest.skip("No MJPG encoder available")
    df = _tracks(n_per_roi=12, rois=("r0",))
    store = LabelStore.from_tracks(str(vp), df, crop_size=64, n=5, seed=0)
    crops = extract_label_crops(vp, store.frames, store.crop_size)   # context=0
    assert crops
    for i, fl in enumerate(store.frames):
        af = int(fl.frame)
        assert list(crops[i]) == [af]                # only the anchor frame
        assert crops[i][af].shape == (64, 64)


def test_extract_label_crops_context_window(tmp_path):
    vp = tmp_path / "vid.avi"
    if not _write_video(vp, n=12):
        pytest.skip("No MJPG encoder available")
    df = _tracks(n_per_roi=12, rois=("r0",))
    store = LabelStore.from_tracks(str(vp), df, crop_size=64, n=5, seed=0)
    crops = extract_label_crops(vp, store.frames, store.crop_size, context=2)
    for i, fl in enumerate(store.frames):
        af, win = int(fl.frame), crops[i]
        assert af in win                             # anchor always present
        assert all(0 <= k <= 11 for k in win)        # clamped to the 12-frame video
        assert all(abs(k - af) <= 2 for k in win)    # within +/-context
        assert 1 <= len(win) <= 5                    # up to 2*context+1, clamped at edges


# ---- .slp export ------------------------------------------------------------

@pytest.mark.skipif(_HAVE_SLEAP_IO, reason="sleap-io installed; missing-dep path not exercised")
def test_export_slp_without_dep_raises(tmp_path):
    store = LabelStore.from_tracks("vid.avi", _tracks(), crop_size=96, n=4, seed=0)
    with pytest.raises(ImportError, match="sleap-io"):
        store.export_slp(tmp_path / "labels.slp")


@pytest.mark.skipif(not _HAVE_SLEAP_IO, reason="sleap-io not installed")
def test_export_slp_roundtrip_preserves_points(tmp_path):
    # Regression: sio.Skeleton mutates its nodes list in place, which silently
    # turned every exported point into NaN. Export -> reload must keep real coords.
    import sleap_io as sio
    vp = tmp_path / "vid.avi"
    if not _write_video(vp, n=12):
        pytest.skip("No MJPG encoder available")
    df = _tracks(n_per_roi=12, rois=("r0",))
    store = LabelStore.from_tracks(str(vp), df, crop_size=64, n=3, seed=0)
    fl = store.frames[0]
    fl.points["thorax"] = Keypoint(30.0, 32.0, visible=True, score=1.0)
    fl.labeled = True

    p = store.export_slp(tmp_path / "labels.slp")
    labels = sio.load_slp(str(p))
    assert labels.skeletons[0].name == "fly5"
    lf = next(l for l in labels.labeled_frames if l.frame_idx == fl.frame)
    arr = lf.instances[0].numpy()
    x0, y0 = fl.origin(64)
    ti = store.skeleton.index("thorax")
    assert arr[ti][0] == pytest.approx(30.0 + x0)     # points survived, not NaN
    assert arr[ti][1] == pytest.approx(32.0 + y0)
