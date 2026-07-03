"""Tests for Milestone B5: pose-stage dispatch + tracks->Labels->pose-df path."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.pose.skeleton import DEFAULT_SKELETON
from anytrack.pose.engine import MockPoseEngine
from anytrack.pose.pipeline import run_pose_stage, POSE_COLUMNS

_HAVE_SLEAP_IO = importlib.util.find_spec("sleap_io") is not None


def _write_video(path, n=6, h=200, w=520):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h), isColor=True)
    if not wr.isOpened():
        wr.release()
        return False
    for _ in range(n):
        g = np.full((h, w), 200, np.uint8)
        cv2.circle(g, (100, 100), 6, 20, -1)
        cv2.circle(g, (400, 100), 6, 20, -1)
        wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def _two_roi_tracks(n=4):
    rows = []
    for f in range(n):
        rows.append({"roi": "r0", "track_id": 0, "frame": f, "t_s": f / 30.0, "x": 100.0, "y": 100.0})
        rows.append({"roi": "r1", "track_id": 1, "frame": f, "t_s": f / 30.0, "x": 400.0, "y": 100.0})
    return pd.DataFrame(rows)


def _video(tmp_path, n=4):
    vp = tmp_path / "vid.avi"
    ok = _write_video(vp, n=n)
    rois = [CircleROI("r0", 100, 100, 90), CircleROI("r1", 400, 100, 90)]
    return ok, SimpleNamespace(video_path=vp, rois=rois)


# ---- dispatch: mock engine -> crop path ------------------------------------

def test_run_pose_stage_dispatch_mock(tmp_path):
    ok, video = _video(tmp_path, n=4)
    if not ok:
        pytest.skip("No MJPG encoder available")
    df = _two_roi_tracks(n=4)
    cfg = AnyTrackConfig(crop_size=64)
    pose = run_pose_stage(video, df, cfg, engine=MockPoseEngine(DEFAULT_SKELETON))
    assert list(pose.columns) == POSE_COLUMNS
    assert len(pose) == 4 * 2 * 5                       # frames x rois x keypoints
    assert set(pose["roi"]) == {"r0", "r1"}


def test_run_pose_stage_routes_to_predict_labels(tmp_path):
    # An engine exposing predict_labels must be routed to the sleap path, not run_pose.
    ok, video = _video(tmp_path, n=2)
    if not ok:
        pytest.skip("No MJPG encoder available")

    class _Sentinel(Exception):
        pass

    class FakePredictEngine:
        keypoint_names = list(DEFAULT_SKELETON.nodes)
        def predict_labels(self, labels, **kw):
            raise _Sentinel()

    with pytest.raises(_Sentinel):
        run_pose_stage(video, _two_roi_tracks(2), AnyTrackConfig(), engine=FakePredictEngine())


# ---- tracks -> Labels -> pose-df conversion (needs sleap-io) ----------------

@pytest.mark.skipif(not _HAVE_SLEAP_IO, reason="sleap-io not installed")
def test_tracks_to_labels_and_conversion(tmp_path):
    from anytrack.pose.infer import tracks_to_labels, labels_to_pose_df
    ok, video = _video(tmp_path, n=3)
    if not ok:
        pytest.skip("No MJPG encoder available")
    df = _two_roi_tracks(n=3)
    labels, index = tracks_to_labels(video, df, DEFAULT_SKELETON)
    assert len(labels.labeled_frames) == 3
    assert len(index) == 6                              # 3 frames x 2 rois

    # Treat the input labels (points anchored at centroids) as "predictions" and
    # verify the conversion + nearest-centroid ROI matching.
    node_names = list(DEFAULT_SKELETON.nodes)
    pose = labels_to_pose_df(labels, index, node_names, video.rois, crop_size=96)
    assert list(pose.columns) == POSE_COLUMNS
    assert len(pose) == 6 * 5

    r0 = pose[(pose["roi"] == "r0") & (pose["frame"] == 0)].iloc[0]
    assert abs(r0["x_full"] - 100.0) < 1e-3             # point == centroid
    assert abs(r0["x_roi"] - (100.0 - (100 - 90))) < 1e-3   # roi origin = cx - r
    r1 = pose[(pose["roi"] == "r1") & (pose["frame"] == 0)].iloc[0]
    assert abs(r1["x_full"] - 400.0) < 1e-3             # matched to the right arena
