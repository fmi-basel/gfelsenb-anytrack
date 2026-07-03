"""sleap-nn pose inference on tracked centroids (Milestone B5).

Turns the classical tracker's centroid dataframe into a long-format keypoint
table by running a trained **centered-instance** model. The tracker already
supplies the crop anchors, so we build a ``sleap_io.Labels`` with one instance
per ``(frame, ROI, track)`` placed at the centroid, run ``Predictor.predict``
(which crops around each anchor, runs the model, and refines peaks), then map
the predicted full-frame keypoints back to the pose schema.

This is the real-backend counterpart to the crop-based
:func:`anytrack.pose.pipeline.run_pose` (the mock/test path); both emit the same
:data:`~anytrack.pose.pipeline.POSE_COLUMNS`. Heavy deps (sleap-io/sleap-nn) are
imported lazily inside the functions that need them.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..coordinates import full_to_roi
from .labeling import crop_origin
from .pipeline import POSE_COLUMNS
from .skeleton import get_skeleton


def tracks_to_labels(video, df: pd.DataFrame, skeleton, every_n: int = 1,
                     x_col: str = "x", y_col: str = "y"):
    """Build an input ``sleap_io.Labels`` with one instance per centroid.

    Returns ``(labels, index)`` where ``index`` is the flat list of per-instance
    metadata (frame/roi/track_id/t_s/cx/cy) in creation order, used to map
    predictions back to tracks.
    """
    import sleap_io as sio

    node_names = [str(n) for n in skeleton.nodes]
    k = len(node_names)
    # sio.Skeleton mutates its nodes list in place -> hand it a throwaway copy.
    skel = sio.Skeleton(
        nodes=list(node_names),
        edges=[(a, b) for a, b in skeleton.edges],
        symmetries=[set(s) for s in skeleton.symmetries],
        name=skeleton.name,
    )
    vid = sio.Video.from_filename(str(video.video_path))

    valid = df.dropna(subset=[x_col, y_col])
    index: List[Dict[str, Any]] = []
    lfs = []
    for f, g in valid.groupby("frame"):
        f = int(f)
        if every_n > 1 and (f % every_n != 0):
            continue
        insts = []
        for _, r in g.iterrows():
            cx, cy = float(r[x_col]), float(r[y_col])
            pts = np.tile([cx, cy], (k, 1)).astype("float64")   # anchor = centroid
            insts.append(sio.Instance.from_numpy(pts, skeleton=skel))
            index.append({
                "frame": f, "roi": str(r.get("roi", "")),
                "track_id": int(r["track_id"]) if "track_id" in r and pd.notna(r["track_id"]) else 0,
                "t_s": float(r["t_s"]) if "t_s" in r and pd.notna(r["t_s"]) else float("nan"),
                "cx": cx, "cy": cy,
            })
        if insts:
            lfs.append(sio.LabeledFrame(video=vid, frame_idx=f, instances=insts))

    labels = sio.Labels(labeled_frames=lfs, videos=[vid], skeletons=[skel])
    return labels, index


def _point_scores(inst, k: int) -> np.ndarray:
    """Per-keypoint confidence from a sleap-io PredictedInstance (robust to API)."""
    try:
        out = inst.numpy(scores=True)
        if isinstance(out, tuple) and len(out) == 2:
            return np.asarray(out[1], dtype=float).reshape(-1)[:k]
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, 2][:k]
    except (TypeError, ValueError):
        pass
    pts = getattr(inst, "points", None)
    if pts is not None:
        try:
            return np.asarray(pts["score"], dtype=float).reshape(-1)[:k]
        except Exception:  # noqa: BLE001 - fall through to a safe default
            pass
    return np.ones(k, dtype=float)


def labels_to_pose_df(pred_labels, index: List[Dict[str, Any]], node_names: List[str],
                      rois, crop_size: int) -> pd.DataFrame:
    """Map a predicted ``Labels`` back to the long pose table.

    Predicted instances are matched to input tracks by **nearest centroid within
    the frame** (arenas are far apart, so this is unambiguous and robust to any
    instance reordering by the predictor).
    """
    inputs_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in index:
        inputs_by_frame[rec["frame"]].append(rec)
    roi_map = {r.name: r for r in (rois or [])}
    k = len(node_names)

    rows: List[Dict[str, Any]] = []
    for lf in pred_labels.labeled_frames:
        f = int(lf.frame_idx)
        candidates = inputs_by_frame.get(f, [])
        used: set = set()
        for inst in lf.instances:
            pts = np.asarray(inst.numpy(), dtype=float)          # (K, 2) full-frame
            finite = pts[np.isfinite(pts).all(axis=1)]
            anchor = finite.mean(axis=0) if len(finite) else np.array([np.nan, np.nan])
            # nearest unused input centroid in this frame
            best, bd = None, np.inf
            for i, rec in enumerate(candidates):
                if i in used:
                    continue
                d = (rec["cx"] - anchor[0]) ** 2 + (rec["cy"] - anchor[1]) ** 2
                if d < bd:
                    bd, best = d, i
            if best is None:
                continue
            used.add(best)
            rec = candidates[best]
            roi = roi_map.get(rec["roi"])
            x0, y0 = crop_origin(rec["cx"], rec["cy"], crop_size)
            scores = _point_scores(inst, k)
            for j, node in enumerate(node_names):
                xf, yf = float(pts[j, 0]), float(pts[j, 1])
                if roi is not None:
                    xr, yr = full_to_roi(xf, yf, roi)
                else:
                    xr = yr = float("nan")
                rows.append({
                    "roi": rec["roi"], "track_id": rec["track_id"], "frame": f,
                    "t_s": rec["t_s"], "keypoint": node,
                    "x_full": xf, "y_full": yf, "x_roi": xr, "y_roi": yr,
                    "x_crop": xf - x0, "y_crop": yf - y0,
                    "score": float(scores[j]) if j < len(scores) else float("nan"),
                    "out_of_bounds": False,
                })
    return pd.DataFrame.from_records(rows, columns=POSE_COLUMNS)


def run_pose_sleap(video, df: pd.DataFrame, cfg, engine, show_progress: bool = False) -> pd.DataFrame:
    """Full sleap-nn pose pass: tracks df -> Labels -> predict -> long pose df."""
    skeleton = get_skeleton(cfg)
    every_n = int(getattr(cfg, "pose_every_n", 1))
    labels, index = tracks_to_labels(video, df, skeleton, every_n=every_n)
    if not index:
        return pd.DataFrame(columns=POSE_COLUMNS)
    kwargs = {}
    if show_progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(labels.labeled_frames), desc="  pose",
                       bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                       leave=True)
            kwargs["progress_callback"] = lambda done, total: (bar.update(done - bar.n),
                                                               bar.refresh())[0]
        except Exception:
            bar = None
    else:
        bar = None
    pred = engine.predict_labels(labels, **kwargs)
    if bar is not None:
        bar.close()
    node_names = [str(n) for n in skeleton.nodes]
    return labels_to_pose_df(pred, index, node_names, getattr(video, "rois", []),
                             int(getattr(cfg, "crop_size", 96)))
