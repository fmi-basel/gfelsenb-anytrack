"""
Coordinate-system conversions for the tracking pipeline.

Three coordinate systems appear in the pipeline:

* **full-frame**   — pixels in the original video frame (canonical output; the
  ``x``/``y`` columns everything downstream reads).
* **ROI-local**    — pixels relative to a ROI's bounding-box top-left, i.e.
  ``CircleROI.bbox()[:2]``.
* **scaled-local** — ROI-local pixels of a *downscaled* ROI sub-video (fast
  path); related to full-frame by ``full = scaled * scale + crop_offset``.
* **crop-local**   — pixels relative to a dynamic object-centered crop's
  top-left (used by the pose crops in ``cropper.py``).

All functions accept scalars or numpy arrays (they are pure arithmetic), so
they vectorize over dataframe columns.
"""
from __future__ import annotations

from typing import Tuple, Sequence, Dict

import pandas as pd

from .models import CircleROI


# --- scaled-local <-> full-frame (downscaled ROI sub-videos, fast path) ------

def scaled_to_full(x, y, scale: float, x0: float, y0: float):
    """scaled-local -> full-frame (matches the fast tracker's inline math)."""
    return x * scale + x0, y * scale + y0


def full_to_scaled(x, y, scale: float, x0: float, y0: float):
    """full-frame -> scaled-local."""
    return (x - x0) / scale, (y - y0) / scale


# --- ROI-local <-> full-frame ------------------------------------------------

def roi_origin(roi: CircleROI) -> Tuple[int, int]:
    """The full-frame pixel that is ROI-local (0, 0): the bbox top-left."""
    x0, y0, _, _ = roi.bbox()
    return x0, y0


def roi_to_full(x_roi, y_roi, roi: CircleROI):
    x0, y0 = roi_origin(roi)
    return x_roi + x0, y_roi + y0


def full_to_roi(x_full, y_full, roi: CircleROI):
    x0, y0 = roi_origin(roi)
    return x_full - x0, y_full - y0


# --- crop-local <-> full-frame (dynamic centroid crops, pose) ----------------

def crop_to_full(x_crop, y_crop, crop_x0: float, crop_y0: float):
    return x_crop + crop_x0, y_crop + crop_y0


def full_to_crop(x_full, y_full, crop_x0: float, crop_y0: float):
    return x_full - crop_x0, y_full - crop_y0


# --- dataframe helper --------------------------------------------------------

def add_roi_local_columns(
    df: pd.DataFrame,
    rois: Sequence[CircleROI],
    x_col: str = "x",
    y_col: str = "y",
    roi_col: str = "roi",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``x_roi``/``y_roi`` columns added.

    Full-frame ``x``/``y`` stay canonical and untouched; this is purely
    additive so existing consumers are unaffected. ROIs not present in
    ``rois`` fall back to a (0, 0) origin.
    """
    if df.empty or x_col not in df.columns or roi_col not in df.columns:
        return df
    origins: Dict[str, Tuple[int, int]] = {r.name: roi_origin(r) for r in rois}
    x0 = df[roi_col].map(lambda n: origins.get(n, (0, 0))[0])
    y0 = df[roi_col].map(lambda n: origins.get(n, (0, 0))[1])
    df = df.copy()
    df["x_roi"] = df[x_col] - x0
    df["y_roi"] = df[y_col] - y0
    return df
