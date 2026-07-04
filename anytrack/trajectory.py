from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

def interpolate_missing(df: pd.DataFrame, cols=("x", "y", "angle_deg", "major", "minor")) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("frame").reset_index(drop=True).copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float).interpolate(limit_direction="both")
    return out

def enforce_single_per_frame(df: pd.DataFrame) -> pd.DataFrame:
    # If multiple detections per frame exist, keep the first (placeholder for more advanced logic).
    if df.empty:
        return df
    return df.sort_values(["frame"]).drop_duplicates(subset=["roi", "track_id", "frame"], keep="first")


def resample_to_frames(df: pd.DataFrame, frames, t_s_by_frame=None,
                       cols=("x", "y", "angle_deg", "major", "minor")) -> pd.DataFrame:
    """Reindex each (roi, track_id) to the full ``frames`` set and interpolate ``cols``.

    Fills rows for frames absent from ``df`` — e.g. those skipped by
    ``track_stride`` — so downstream stages see a per-frame trajectory. ``t_s`` is
    taken from ``t_s_by_frame`` (the timing table) when given, else interpolated.
    """
    if df.empty:
        return df
    frames = np.asarray(sorted({int(f) for f in frames}), dtype=np.int64)
    parts = []
    for (roi, tid), g in df.sort_values("frame").groupby(["roi", "track_id"], sort=False):
        g = g.drop_duplicates("frame").set_index("frame").reindex(frames)
        g["roi"] = roi
        g["track_id"] = tid
        for c in cols:
            if c in g.columns:
                g[c] = g[c].astype(float).interpolate(limit_direction="both")
        if t_s_by_frame is not None:
            g["t_s"] = [t_s_by_frame.get(int(f), float("nan")) for f in frames]
        elif "t_s" in g.columns:
            g["t_s"] = g["t_s"].astype(float).interpolate(limit_direction="both")
        parts.append(g.reset_index().rename(columns={"index": "frame"}))
    return pd.concat(parts, ignore_index=True)
