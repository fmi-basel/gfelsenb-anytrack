"""Tests for odor-port detection (anytrack.odor_port) — classical backend + plumbing."""
from __future__ import annotations

import numpy as np
import cv2
import pandas as pd

from anytrack.config import AnyTrackConfig
from anytrack.models import CircleROI
from anytrack.odor_port import (build_segment_engine, ClassicalPortDetector, detect_odor_port,
                                detect_ports, port_distance, add_port_distance)


def _arena_frame(port_xy=(150, 150), port_r=8, wall_blob=None):
    f = np.zeros((300, 300), np.uint8)
    cv2.circle(f, (150, 150), 100, 200, -1)          # bright arena
    if port_xy is not None:
        cv2.circle(f, port_xy, port_r, 40, -1)       # dark central port
    if wall_blob is not None:
        cv2.circle(f, wall_blob, 7, 40, -1)          # off-centre dark blob (a fly)
    return f


def test_default_backend_is_classical():
    assert build_segment_engine(AnyTrackConfig()).name == "classical"


def test_classical_finds_central_port():
    roi = CircleROI(name="arena_01", cx=150, cy=150, r=100)
    p = detect_odor_port(_arena_frame(), roi, AnyTrackConfig())
    assert p is not None and p.backend == "classical"
    assert abs(p.cx - 150) <= 2 and abs(p.cy - 150) <= 2
    assert p.offset_frac < 0.05 and 4 <= p.r <= 20 and p.conf > 0


def _faint_disk_frame(cx=150, cy=150, port_r=11):
    """Arena with a *large, low-contrast* circular port — the real fixture shape.

    A darkness-blob detector latches onto interior sub-spots and mis-sizes such a
    port; the edge (radial-gradient) detector should localize its rim and size it."""
    f = np.zeros((300, 300), np.uint8)
    cv2.circle(f, (150, 150), 100, 200, -1)          # bright arena floor
    cv2.circle(f, (cx, cy), port_r, 165, -1)         # faint darker disk (Δ~35)
    return cv2.GaussianBlur(f, (0, 0), 2.0)          # low-contrast, soft-edged


def test_edge_detector_sizes_low_contrast_port():
    """Regression: the port is sized at its true (larger) radius, not ~half."""
    roi = CircleROI(name="arena_01", cx=150, cy=150, r=100)
    p = detect_odor_port(_faint_disk_frame(port_r=11), roi, AnyTrackConfig())
    assert p is not None and p.backend == "classical"
    assert abs(p.cx - 150) <= 3 and abs(p.cy - 150) <= 3
    assert abs(p.r - 11) <= 3 and p.conf > 0


def test_offcentre_blob_not_taken_as_port():
    """A fly at the wall must not be mistaken for the (central) port."""
    roi = CircleROI(name="arena_01", cx=150, cy=150, r=100)
    frame = _arena_frame(port_xy=None, wall_blob=(230, 150))
    assert detect_odor_port(frame, roi, AnyTrackConfig()) is None


def test_no_port_returns_none():
    roi = CircleROI(name="arena_01", cx=150, cy=150, r=100)
    assert detect_odor_port(_arena_frame(port_xy=None), roi, AnyTrackConfig()) is None


def test_port_distance_and_columns():
    roi = CircleROI(name="arena_01", cx=150, cy=150, r=100)
    cfg = AnyTrackConfig(); cfg.arena_diameter_mm = 100.0     # 200 px diameter → 0.5 mm/px
    ports = detect_ports(_arena_frame(), [roi], cfg)
    assert set(ports) == {"arena_01"}
    p = ports["arena_01"]
    assert round(port_distance(150, 190, p)) == 40           # 40 px straight down

    df = pd.DataFrame({"roi": ["arena_01", "arena_01"], "x": [150.0, 150.0], "y": [150.0, 190.0]})
    add_port_distance(df, ports, [roi], cfg)
    assert {"dist_to_port_px", "dist_to_port_mm"} <= set(df.columns)
    assert abs(df["dist_to_port_px"].iloc[1] - 40) < 1.5
    assert abs(df["dist_to_port_mm"].iloc[1] - 20) < 1.0      # 40 px × 0.5 mm/px


def test_shared_radius_gives_constant_size_across_arenas():
    """Ports of different pixel sizes get a single shared (median) radius."""
    frame = np.zeros((300, 600), np.uint8)
    cv2.circle(frame, (150, 150), 100, 200, -1); cv2.circle(frame, (150, 150), 8, 40, -1)
    cv2.circle(frame, (450, 150), 100, 200, -1); cv2.circle(frame, (450, 150), 13, 40, -1)
    rois = [CircleROI(name="a1", cx=150, cy=150, r=100),
            CircleROI(name="a2", cx=450, cy=150, r=100)]
    ports = detect_ports(frame, rois, AnyTrackConfig())
    assert set(ports) == {"a1", "a2"}
    assert abs(ports["a1"].r - ports["a2"].r) < 1e-6      # constant size across arenas
    # both centres localized correctly despite the shared radius
    assert abs(ports["a1"].cx - 150) <= 2 and abs(ports["a2"].cx - 450) <= 2


def test_sam_backend_falls_back_without_weights(capsys):
    """port_backend='sam' with no checkpoint / package falls back to classical."""
    cfg = AnyTrackConfig(); cfg.port_backend = "sam"; cfg.sam_model_path = ""
    eng = build_segment_engine(cfg)
    assert eng.name == "classical"                            # graceful fallback
