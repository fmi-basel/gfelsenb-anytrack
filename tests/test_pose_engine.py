"""Tests for Milestone B4: peak extraction + the engine factory."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.pose.skeleton import DEFAULT_SKELETON
from anytrack.pose.engine import MockPoseEngine, build_engine, heatmaps_to_keypoints

_HAVE_TORCH = importlib.util.find_spec("torch") is not None


# ---- heatmaps_to_keypoints --------------------------------------------------

def test_argmax_peak_location_and_score():
    # Single-peak heatmaps: one bright pixel per (crop, keypoint).
    n, k, h, w = 2, 3, 16, 16
    hm = np.zeros((n, k, h, w), np.float32)
    hm[0, 0, 4, 8] = 0.9        # (row=4, col=8)
    hm[0, 1, 0, 0] = 0.5
    hm[1, 2, 15, 15] = 1.0
    crop = 64                    # heatmap grid 16 -> crop 64, scale = 4
    kps, scores = heatmaps_to_keypoints(hm, crop, method="argmax")

    assert kps.shape == (2, 3, 2) and scores.shape == (2, 3)
    assert np.allclose(kps[0, 0], [(8 + 0.5) * 4, (4 + 0.5) * 4])   # x=col, y=row
    assert scores[0, 0] == pytest.approx(0.9)
    assert np.allclose(kps[1, 2], [(15 + 0.5) * 4, (15 + 0.5) * 4])
    assert scores[1, 2] == pytest.approx(1.0)


def test_centroid_matches_symmetric_peak():
    # A symmetric blob's soft-argmax centroid sits at the blob center.
    hm = np.zeros((1, 1, 16, 16), np.float32)
    hm[0, 0, 7:9, 7:9] = 1.0     # 2x2 block centered between 7 and 8 -> 7.5
    kps, _ = heatmaps_to_keypoints(hm, 16, method="centroid")   # scale 1.0
    assert np.allclose(kps[0, 0], [(7.5 + 0.5), (7.5 + 0.5)], atol=1e-4)


def test_heatmaps_shape_and_method_validation():
    with pytest.raises(ValueError):
        heatmaps_to_keypoints(np.zeros((3, 16, 16), np.float32), 64)   # not 4-D
    with pytest.raises(ValueError):
        heatmaps_to_keypoints(np.zeros((1, 1, 4, 4), np.float32), 64, method="nope")


# ---- build_engine factory ---------------------------------------------------

def test_build_engine_defaults_to_mock():
    eng = build_engine(AnyTrackConfig(), DEFAULT_SKELETON)          # no model path
    assert isinstance(eng, MockPoseEngine)
    assert eng.keypoint_names == DEFAULT_SKELETON.nodes


def test_build_engine_mock_backend():
    cfg = AnyTrackConfig(pose_backend="mock", sleap_model_path="/some/model")
    assert isinstance(build_engine(cfg, DEFAULT_SKELETON), MockPoseEngine)


def test_build_engine_unknown_backend_raises():
    cfg = AnyTrackConfig(pose_backend="bogus", sleap_model_path="/some/model")
    with pytest.raises(ValueError, match="unknown pose_backend"):
        build_engine(cfg, DEFAULT_SKELETON)


@pytest.mark.skipif(_HAVE_TORCH, reason="torch installed; missing-dep path not exercised")
def test_build_engine_sleap_without_torch_raises():
    cfg = AnyTrackConfig(pose_backend="sleap-nn", sleap_model_path="/nonexistent/model")
    with pytest.raises(ImportError, match="pose"):
        build_engine(cfg, DEFAULT_SKELETON)
