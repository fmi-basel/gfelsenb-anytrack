"""Tests for Milestone B5 pose QC: flags + confidence plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anytrack.config import AnyTrackConfig
from anytrack.pose.skeleton import DEFAULT_SKELETON
from anytrack.pose.qc_pose import (
    POSE_FLAG_COLUMNS, compute_pose_flags, plot_pose_confidence, pose_flag_summary,
)

# Body along the image y-axis: head above, abdomen below, thorax between.
# Wings extended laterally: L to -x, R to +x (consistent, correct).
_NORMAL = {
    "head": (50.0, 30.0), "thorax": (50.0, 50.0), "abdomen_tip": (50.0, 70.0),
    "wingL": (30.0, 60.0), "wingR": (70.0, 60.0),
}


def _rows(frame, pts, scores=None, roi="r0"):
    scores = scores or {k: 0.9 for k in pts}
    out = []
    for kp, (x, y) in pts.items():
        out.append({"roi": roi, "track_id": 0, "frame": frame, "t_s": frame / 30.0,
                    "keypoint": kp, "x_full": x, "y_full": y,
                    "x_roi": x, "y_roi": y, "x_crop": x, "y_crop": y,
                    "score": scores[kp], "out_of_bounds": False})
    return out


def _df(frame_pts):
    rows = []
    for f, pts in frame_pts:
        rows.extend(_rows(f, pts))
    return pd.DataFrame(rows)


def test_normal_track_no_swap_no_flip():
    df = _df([(f, _NORMAL) for f in range(8)])
    flags = compute_pose_flags(df, DEFAULT_SKELETON, AnyTrackConfig())
    assert len(flags) == 8
    assert not flags["flag_wing_swap"].any()
    assert not flags["flag_headtail"].any()
    assert not flags["flag_low_conf"].any()
    assert not flags["flag_missing"].any()


def test_wing_swap_flagged():
    frames = [(f, _NORMAL) for f in range(7)]
    swapped = dict(_NORMAL, wingL=(70.0, 60.0), wingR=(30.0, 60.0))   # L/R exchanged
    frames.append((7, swapped))
    flags = compute_pose_flags(_df(frames), DEFAULT_SKELETON, AnyTrackConfig())
    fl = flags.set_index("frame")["flag_wing_swap"]
    assert fl.loc[7] and not fl.loc[0:6].any()


def test_headtail_flip_flagged():
    frames = [(f, _NORMAL) for f in range(5)]
    flipped = dict(_NORMAL, head=(50.0, 70.0), abdomen_tip=(50.0, 30.0))  # body reversed
    frames.append((5, flipped))
    flags = compute_pose_flags(_df(frames), DEFAULT_SKELETON, AnyTrackConfig())
    assert flags.set_index("frame")["flag_headtail"].loc[5]


def test_low_conf_and_missing():
    low = _df([(0, _NORMAL)])
    low.loc[low["keypoint"] == "head", "score"] = 0.05                # below conf_min 0.2
    fl = compute_pose_flags(low, DEFAULT_SKELETON, AnyTrackConfig())
    assert bool(fl["flag_low_conf"].iloc[0]) and not bool(fl["flag_missing"].iloc[0])

    miss = _df([(0, _NORMAL)])
    miss.loc[miss["keypoint"] == "wingL", ["x_full", "y_full", "score"]] = np.nan
    fm = compute_pose_flags(miss, DEFAULT_SKELETON, AnyTrackConfig())
    assert bool(fm["flag_missing"].iloc[0])


def test_pose_flag_summary_and_empty():
    df = _df([(f, _NORMAL) for f in range(4)])
    s = pose_flag_summary(compute_pose_flags(df, DEFAULT_SKELETON, AnyTrackConfig()))
    assert s["n_instances"] == 4 and set(POSE_FLAG_COLUMNS) <= set(s)
    empty = compute_pose_flags(pd.DataFrame(), DEFAULT_SKELETON, AnyTrackConfig())
    assert empty.empty and list(empty.columns)[:4] == ["roi", "track_id", "frame", "t_s"]


def test_plot_pose_confidence(tmp_path):
    df = _df([(f, _NORMAL) for f in range(5)])
    paths = plot_pose_confidence(df, AnyTrackConfig(), tmp_path)
    assert paths and Path(paths[0]).exists()
