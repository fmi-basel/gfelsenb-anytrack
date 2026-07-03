"""Pose inference pass: centroid crops -> engine -> long-format keypoint table.

Backend-agnostic: takes any :class:`~anytrack.pose.engine.PoseEngine` (the mock
for tests, the sleap-nn engine in production), so the crop→infer→merge→write
plumbing is fully testable without heavy dependencies.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..cropper import iter_crop_batches
from ..coordinates import crop_to_full, full_to_roi

# One row per (frame, roi, keypoint). Coordinates in crop-local, full-frame and
# ROI-local frames so downstream (QC, kinematics, export) can use whichever.
POSE_COLUMNS = [
    "roi", "track_id", "frame", "t_s", "keypoint",
    "x_full", "y_full", "x_roi", "y_roi", "x_crop", "y_crop",
    "score", "out_of_bounds",
]


def run_pose(video, df: pd.DataFrame, cfg, engine, show_progress: bool = False) -> pd.DataFrame:
    """Run ``engine`` over centroid crops from ``df`` and return a long pose table.

    ``df`` is the centroid dataframe (needs roi/frame/x/y, optionally track_id/t_s).
    Keypoints come back crop-local from the engine and are mapped to full-frame
    (``crop_to_full``) and ROI-local (``full_to_roi``).
    """
    rois = {str(r.name): r for r in (getattr(video, "rois", []) or [])}
    names = list(engine.keypoint_names)
    records: List[Dict] = []

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(desc="  pose", unit="crop",
                        bar_format="  {desc} {n_fmt} crops [{elapsed}]", leave=True)
        except Exception:
            pbar = None

    for crops, metas in iter_crop_batches(video, df, cfg):
        kps, scores = engine.infer_batch(crops)
        for i, meta in enumerate(metas):
            x0, y0 = meta["crop_x0"], meta["crop_y0"]
            roi = rois.get(meta["roi"])
            for k, kp in enumerate(names):
                xc = float(kps[i, k, 0])
                yc = float(kps[i, k, 1])
                xf, yf = crop_to_full(xc, yc, x0, y0)
                if roi is not None:
                    xr, yr = full_to_roi(xf, yf, roi)
                else:
                    xr = yr = float("nan")
                records.append({
                    "roi": meta["roi"], "track_id": meta["track_id"], "frame": meta["frame"],
                    "t_s": meta["t_s"], "keypoint": kp,
                    "x_full": xf, "y_full": yf, "x_roi": xr, "y_roi": yr,
                    "x_crop": xc, "y_crop": yc,
                    "score": float(scores[i, k]), "out_of_bounds": meta["out_of_bounds"],
                })
        if pbar is not None:
            pbar.update(len(metas))

    if pbar is not None:
        pbar.close()

    return pd.DataFrame.from_records(records, columns=POSE_COLUMNS)
