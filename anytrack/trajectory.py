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
