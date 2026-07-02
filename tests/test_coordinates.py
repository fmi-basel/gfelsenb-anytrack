"""Tests for anytrack.coordinates (Milestone A3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anytrack.coordinates import (
    scaled_to_full,
    full_to_scaled,
    roi_origin,
    roi_to_full,
    full_to_roi,
    crop_to_full,
    full_to_crop,
    add_roi_local_columns,
)
from anytrack.models import CircleROI


def test_scaled_full_roundtrip_scalar():
    x, y = 12.5, 30.0
    scale, x0, y0 = 2.0, 100.0, 200.0
    xf, yf = scaled_to_full(x, y, scale, x0, y0)
    assert (xf, yf) == (125.0, 260.0)
    xs, ys = full_to_scaled(xf, yf, scale, x0, y0)
    assert xs == pytest.approx(x) and ys == pytest.approx(y)


def test_scaled_full_roundtrip_array():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([5.0, 15.0, 25.0])
    xf, yf = scaled_to_full(x, y, 2.0, 100.0, 200.0)
    xs, ys = full_to_scaled(xf, yf, 2.0, 100.0, 200.0)
    np.testing.assert_allclose(xs, x)
    np.testing.assert_allclose(ys, y)


def test_roi_origin_uses_bbox_topleft():
    roi = CircleROI(name="r", cx=100, cy=120, r=30)
    assert roi_origin(roi) == (70, 90)


def test_roi_full_roundtrip():
    roi = CircleROI(name="r", cx=100, cy=120, r=30)
    x_full, y_full = 105.0, 118.0
    xr, yr = full_to_roi(x_full, y_full, roi)
    assert (xr, yr) == (35.0, 28.0)
    xf, yf = roi_to_full(xr, yr, roi)
    assert (xf, yf) == (x_full, y_full)


def test_roi_origin_clamps_at_frame_edge():
    # cx - r < 0 -> bbox clamps origin to 0.
    roi = CircleROI(name="edge", cx=10, cy=10, r=30)
    assert roi_origin(roi) == (0, 0)
    xf, yf = roi_to_full(*full_to_roi(5.0, 5.0, roi), roi)
    assert (xf, yf) == (5.0, 5.0)


def test_crop_full_roundtrip():
    xf, yf = crop_to_full(10.0, 12.0, 300.0, 400.0)
    assert (xf, yf) == (310.0, 412.0)
    xc, yc = full_to_crop(xf, yf, 300.0, 400.0)
    assert (xc, yc) == (10.0, 12.0)


def test_add_roi_local_columns():
    rois = [
        CircleROI(name="a", cx=100, cy=120, r=30),  # origin (70, 90)
        CircleROI(name="b", cx=300, cy=300, r=50),  # origin (250, 250)
    ]
    df = pd.DataFrame({
        "roi": ["a", "b", "a"],
        "x": [75.0, 260.0, 100.0],
        "y": [95.0, 255.0, 120.0],
    })
    out = add_roi_local_columns(df, rois)
    # Original columns preserved (canonical full-frame).
    assert list(df["x"]) == [75.0, 260.0, 100.0]
    assert out["x_roi"].tolist() == [5.0, 10.0, 30.0]
    assert out["y_roi"].tolist() == [5.0, 5.0, 30.0]


def test_add_roi_local_columns_empty_df_is_noop():
    out = add_roi_local_columns(pd.DataFrame(), [])
    assert out.empty
