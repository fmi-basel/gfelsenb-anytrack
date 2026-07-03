"""Tests for Milestone B0+B1: skeleton, mock engine, crop batching, pose pass."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.pose.skeleton import Skeleton, DEFAULT_SKELETON, load_skeleton, get_skeleton
from anytrack.pose.engine import PoseEngine, MockPoseEngine
from anytrack.pose.pipeline import run_pose, POSE_COLUMNS
from anytrack.cropper import iter_crop_batches


# ---- skeleton ---------------------------------------------------------------

def test_default_skeleton():
    s = DEFAULT_SKELETON
    assert s.name == "fly5"
    assert s.nodes == ["head", "thorax", "abdomen_tip", "wingL", "wingR"]
    assert s.n_nodes == 5
    assert s.index("thorax") == 1
    assert ("wingL", "wingR") in s.symmetries
    assert (1, 0) or s.edge_indices()  # edges resolve to node indices
    assert (s.index("head"), s.index("thorax")) in s.edge_indices()


def test_skeleton_json_roundtrip(tmp_path):
    p = DEFAULT_SKELETON.to_json_file(tmp_path / "fly5.json")
    loaded = load_skeleton(p)
    assert loaded.to_dict() == DEFAULT_SKELETON.to_dict()


def test_skeleton_rejects_unknown_node():
    with pytest.raises(ValueError):
        Skeleton.from_dict({"name": "x", "nodes": ["a", "b"], "edges": [["a", "c"]]})


def test_get_skeleton(tmp_path):
    assert get_skeleton(AnyTrackConfig()).name == "fly5"           # default
    p = Skeleton("s2", ["p", "q"]).to_json_file(tmp_path / "s2.json")
    assert get_skeleton(AnyTrackConfig(pose_skeleton=str(p))).nodes == ["p", "q"]


# ---- engine -----------------------------------------------------------------

def test_mock_engine_shapes_and_protocol():
    eng = MockPoseEngine(DEFAULT_SKELETON)
    assert isinstance(eng, PoseEngine)              # satisfies the protocol
    crops = np.zeros((3, 96, 96), np.uint8)
    kps, scores = eng.infer_batch(crops)
    assert kps.shape == (3, 5, 2)
    assert scores.shape == (3, 5)
    assert np.allclose(scores, 1.0)
    assert (kps >= 0).all() and (kps <= 96).all()   # inside the crop


# ---- crop batching + pose pass ---------------------------------------------

def _write_video(path, n=12, h=200, w=200):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for _ in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (100, 100), 6, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def _video_and_df(tmp_path, n=12):
    vp = tmp_path / "vid.avi"
    ok = _write_video(vp, n=n)
    video = SimpleNamespace(video_path=vp, rois=[CircleROI(name="r0", cx=100, cy=100, r=90)])
    df = pd.DataFrame({
        "roi": ["r0"] * n, "track_id": [0] * n, "frame": np.arange(n),
        "x": [100.0] * n, "y": [100.0] * n, "t_s": np.arange(n) / 30.0,
    })
    return ok, video, df


def test_iter_crop_batches_shapes_and_meta(tmp_path):
    ok, video, df = _video_and_df(tmp_path, n=12)
    if not ok:
        pytest.skip("No MJPG encoder available")
    cfg = AnyTrackConfig(crop_size=64, pose_batch_size=5)
    total = 0
    for crops, metas in iter_crop_batches(video, df, cfg):
        assert crops.shape[1:] == (64, 64)
        assert crops.shape[0] == len(metas)
        for m in metas:
            assert m["roi"] == "r0" and "crop_x0" in m and "crop_y0" in m and "out_of_bounds" in m
        total += len(metas)
    assert total == 12


def test_iter_crop_batches_every_n(tmp_path):
    ok, video, df = _video_and_df(tmp_path, n=12)
    if not ok:
        pytest.skip("No MJPG encoder available")
    cfg = AnyTrackConfig(crop_size=64, pose_every_n=3)
    frames = [m["frame"] for crops, metas in iter_crop_batches(video, df, cfg) for m in metas]
    assert frames == [0, 3, 6, 9]


def test_run_pose_end_to_end_and_coords(tmp_path):
    ok, video, df = _video_and_df(tmp_path, n=10)
    if not ok:
        pytest.skip("No MJPG encoder available")
    cfg = AnyTrackConfig(crop_size=96)
    eng = MockPoseEngine(DEFAULT_SKELETON)
    pose = run_pose(video, df, cfg, eng)

    assert list(pose.columns) == POSE_COLUMNS
    assert len(pose) == 10 * 5                       # frames x keypoints (1 ROI)
    assert set(pose["keypoint"].unique()) == set(DEFAULT_SKELETON.nodes)

    # thorax (offset 0,0) -> full-frame == centroid (100,100); ROI-local subtracts
    # the ROI origin (cx-r = 10).
    thorax = pose[pose["keypoint"] == "thorax"].iloc[0]
    assert abs(thorax["x_full"] - 100) <= 1 and abs(thorax["y_full"] - 100) <= 1
    assert abs(thorax["x_roi"] - 90) <= 1
    # head sits above the centroid by ~0.28 * crop_size.
    head = pose[pose["keypoint"] == "head"].iloc[0]
    assert head["y_full"] < thorax["y_full"] - 20
    assert (pose["score"] == 1.0).all()
