"""
Foreground detection primitives shared by the tracking pipelines.

This module owns the single canonical implementation of ellipse-candidate
detection (background subtraction -> threshold -> morphology -> contours ->
``cv2.fitEllipse``). Both the legacy single-pass tracker (``tracking.py``) and
the fast parallel tracker (``tracking_fast.py``) call into here, so there is
exactly one place where detection behavior lives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2

from .config import AnyTrackConfig


@dataclass
class EllipseCandidate:
    """A fitted-ellipse detection.

    Coordinates are in whatever frame the caller passed to
    :func:`extract_ellipses` (ROI-local, scaled-local, or full-frame); the
    detector does not convert coordinates.
    """
    x: float
    y: float
    angle_deg: float
    cv_angle_deg: float
    major: float
    minor: float
    area: float
    contour: Optional[np.ndarray] = None


@dataclass
class DebugFrame:
    """Intermediate detection artifacts for visualization/QC."""
    diff: np.ndarray
    mask: np.ndarray
    candidates: List[EllipseCandidate]


def build_morph_kernels(cfg: AnyTrackConfig):
    """Pre-allocate morphology structuring elements (avoid per-frame creation)."""
    kernel_open = None
    kernel_close = None
    if cfg.morph_open > 0:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_open, cfg.morph_open)
        )
    if cfg.morph_close > 0:
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close)
        )
    return kernel_open, kernel_close


def compute_diff(gray: np.ndarray, bg_gray: np.ndarray, cfg: AnyTrackConfig) -> np.ndarray:
    """Directional background difference per ``cfg.bgdiff_type``."""
    bgdiff_type = getattr(cfg, "bgdiff_type", "dark")
    if bgdiff_type == "dark":
        # Only pixels darker than background (dark object on bright background).
        return cv2.subtract(bg_gray, gray)
    elif bgdiff_type == "bright":
        # Only pixels brighter than background.
        return cv2.subtract(gray, bg_gray)
    else:  # "absolute"
        return cv2.absdiff(gray, bg_gray)


def threshold_fg(
    diff: np.ndarray,
    cfg: AnyTrackConfig,
    kernel_open: Optional[np.ndarray] = None,
    kernel_close: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Binarize the diff image and clean it up with open/close morphology."""
    if cfg.thr_method == "fixed":
        _, bw = cv2.threshold(diff, int(cfg.thr_fixed), 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if kernel_open is not None:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open)
    if kernel_close is not None:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close)
    return bw


def _candidates_from_mask(bw: np.ndarray, cfg: AnyTrackConfig) -> List[EllipseCandidate]:
    """Fit ellipses to the foreground blobs of a binary mask, filtered by area."""
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand: List[EllipseCandidate] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < cfg.expected_fly_area_min or area > cfg.expected_fly_area_max:
            continue
        if len(c) < 5:
            continue
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        major = float(max(MA, ma))
        minor = float(min(MA, ma))
        # OpenCV's ellipse angle is measured with a flipped y-axis; the
        # geometric orientation used downstream is angle - 90 (degrees).
        geometric_angle = angle - 90
        cand.append(EllipseCandidate(
            x=float(x),
            y=float(y),
            angle_deg=geometric_angle,
            cv_angle_deg=angle,
            major=major,
            minor=minor,
            area=area,
            contour=c,
        ))
    # Sort largest-area first, then keep at most max_centroids_per_roi.
    cand.sort(key=lambda e: e.area, reverse=True)
    return cand[: max(1, cfg.max_centroids_per_roi)]


def extract_ellipses(
    gray: np.ndarray,
    bg_gray: np.ndarray,
    cfg: AnyTrackConfig,
    mask: Optional[np.ndarray] = None,
    kernel_open: Optional[np.ndarray] = None,
    kernel_close: Optional[np.ndarray] = None,
) -> List[EllipseCandidate]:
    """Detect ellipse candidates in ``gray`` relative to ``bg_gray``.

    If ``mask`` is given, the diff is restricted to it (used by the legacy
    per-ROI path to ignore pixels outside the circular arena). The fast path
    passes ``mask=None`` and operates on the whole ROI crop.
    """
    diff = compute_diff(gray, bg_gray, cfg)
    if mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=mask)
    bw = threshold_fg(diff, cfg, kernel_open, kernel_close)
    return _candidates_from_mask(bw, cfg)


def debug_frame(
    gray: np.ndarray,
    bg_gray: np.ndarray,
    cfg: AnyTrackConfig,
    mask: Optional[np.ndarray] = None,
) -> DebugFrame:
    """Run the exact detection path and return its intermediate artifacts.

    This is the single code path the GUI tracking-debug window should render
    from, so the debug view never drifts from what the tracker actually does.
    """
    kernel_open, kernel_close = build_morph_kernels(cfg)
    diff = compute_diff(gray, bg_gray, cfg)
    if mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=mask)
    bw = threshold_fg(diff, cfg, kernel_open, kernel_close)
    candidates = _candidates_from_mask(bw, cfg)
    return DebugFrame(diff=diff, mask=bw, candidates=candidates)
