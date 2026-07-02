"""Tests for anytrack.background.BackgroundState (A4 drift correction + protection)."""
from __future__ import annotations

import numpy as np
import cv2

from anytrack.config import AnyTrackConfig
from anytrack.background import BackgroundState


def test_drift_estimate_and_clean_diff():
    # Uniformly darkened frame + a dark object: the estimated drift should ~match
    # the applied offset, so the corrected background subtraction is clean on the
    # background yet still detects the object.
    cfg = AnyTrackConfig()
    bg = np.full((100, 100), 180, np.uint8)
    st = BackgroundState(bg, cfg, protect_radius=12, use_arena_mask=False)

    gray = np.full((100, 100), 150, np.uint8)   # uniform -30
    cv2.circle(gray, (70, 70), 5, 20, -1)        # dark object (protected)

    bg_corr = st.corrected(gray, center=(70, 70))
    assert abs(float(np.median(bg_corr)) - 150) <= 2      # drift ~ -30 applied
    diff = cv2.subtract(bg_corr, gray)                     # 'dark' background diff
    assert int(diff[10, 10]) <= 3                          # background clean
    assert int(diff[70, 70]) >= 100                        # object still detected


def test_protect_excludes_object_from_drift():
    # No illumination change, but a big dark blob at the anchor. If it weren't
    # protected, the drift median would be biased downward; with protection the
    # background is left unchanged.
    cfg = AnyTrackConfig()
    bg = np.full((80, 80), 200, np.uint8)
    st = BackgroundState(bg, cfg, protect_radius=15, use_arena_mask=False)
    gray = bg.copy()
    cv2.circle(gray, (40, 40), 8, 10, -1)
    bg_corr = st.corrected(gray, center=(40, 40))
    assert abs(float(np.median(bg_corr)) - 200) <= 1


def test_asym_update_adapts_base_on_safe_pixels():
    cfg = AnyTrackConfig(bg_asym_update=True, bg_step_up=1.0, bg_step_down=0.1)
    bg = np.full((60, 60), 150, np.uint8)
    st = BackgroundState(bg, cfg, protect_radius=5, use_arena_mask=False)
    gray = np.full((60, 60), 170, np.uint8)   # brighter -> step_up=1.0 adapts fully
    st.corrected(gray, center=(30, 30))
    assert float(np.median(st.bg)) > 160       # base plate moved toward 170


def test_no_drift_when_static():
    cfg = AnyTrackConfig()
    bg = np.full((50, 50), 128, np.uint8)
    st = BackgroundState(bg, cfg, protect_radius=8, use_arena_mask=False)
    bg_corr = st.corrected(bg.copy(), center=(25, 25))
    assert abs(float(np.median(bg_corr)) - 128) <= 1
