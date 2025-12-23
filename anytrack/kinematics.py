from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

def add_kinematics(
    df: pd.DataFrame,
    roi_centers: dict[str, Tuple[float, float]],
    px_per_mm_by_roi: dict[str, float],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values(["roi", "track_id", "frame"]).reset_index(drop=True).copy()

    # per-ROI center distance
    cx = out["roi"].map(lambda r: roi_centers[r][0]).astype(float)
    cy = out["roi"].map(lambda r: roi_centers[r][1]).astype(float)
    out["dx_center_px"] = out["x"].astype(float) - cx
    out["dy_center_px"] = out["y"].astype(float) - cy
    out["r_center_px"] = np.hypot(out["dx_center_px"], out["dy_center_px"])

    # convert to mm
    scale = out["roi"].map(lambda r: px_per_mm_by_roi[r]).astype(float)
    out["r_center_mm"] = out["r_center_px"] / scale

    # velocities
    out["x_mm"] = out["x"].astype(float) / scale
    out["y_mm"] = out["y"].astype(float) / scale

    out["dt_s"] = out.groupby(["roi", "track_id"])["t_s"].diff().fillna(0.0)
    out["dx_mm"] = out.groupby(["roi", "track_id"])["x_mm"].diff().fillna(0.0)
    out["dy_mm"] = out.groupby(["roi", "track_id"])["y_mm"].diff().fillna(0.0)
    out["speed_mm_s"] = np.where(out["dt_s"] > 0, np.hypot(out["dx_mm"], out["dy_mm"]) / out["dt_s"], 0.0)

    # unwrap angles for angular speed
    ang = np.deg2rad(out["angle_deg"].astype(float).fillna(0.0).to_numpy())
    # group-wise unwrap (simple: global unwrap, OK if single track; refine later)
    ang_un = np.unwrap(ang)
    out["angle_unwrapped_deg"] = np.rad2deg(ang_un)
    out["dangle_deg"] = out.groupby(["roi", "track_id"])["angle_unwrapped_deg"].diff().fillna(0.0)
    out["ang_speed_deg_s"] = np.where(out["dt_s"] > 0, out["dangle_deg"] / out["dt_s"], 0.0)

    return out
