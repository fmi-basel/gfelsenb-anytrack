from __future__ import annotations

from typing import List, Tuple

import numpy as np
import cv2
import math

from .models import CircleROI

def _reading_order(circles):
    """Order ``(cx, cy, r)`` circles in reading order.

    Rows top→bottom, and left→right within each row. Circles whose centers are
    within ~one median radius in ``y`` are treated as the same row, so slight
    vertical jitter between arenas in a row doesn't scramble their left/right
    order (a plain ``(cy, cx)`` sort does exactly that).
    """
    circles = list(circles)
    if not circles:
        return []
    med_r = float(np.median([c[2] for c in circles]))
    row_tol = max(1.0, med_r)  # same row if centers are within ~one radius in y
    by_y = sorted(circles, key=lambda c: c[1])
    rows, cur = [], [by_y[0]]
    for c in by_y[1:]:
        row_cy = float(np.mean([x[1] for x in cur]))
        if abs(c[1] - row_cy) <= row_tol:
            cur.append(c)
        else:
            rows.append(cur)
            cur = [c]
    rows.append(cur)
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda c: c[0]))  # left -> right within a row
    return ordered


def detect_circular_rois( img,
                            expected_n: int = 4,
                            blur_size: int = 9,
                            dp: float = 1.2,
                            min_dist_ratio: int = 3,
                            name_prefix: str = "arena",
                            param1: int = 120,
                            param2: int = 35,
                            min_radius_ratio: float = 0.18,
                            max_radius_ratio: float = 0.30) -> List[CircleROI]:
    
    #gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = img.copy()
    # Smooth noise but keep edges
    gray_blur = cv2.medianBlur(gray, 9)
    h, w = gray.shape
    min_dim = min(h, w)
    # Rough radius bounds for "4 big circles in a square"
    minR = int(min_radius_ratio * min_dim)
    maxR = int(max_radius_ratio * min_dim)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,                              # accumulator resolution
        minDist=min_dim // min_dist_ratio,  # circles are far apart
        param1=param1,                      # Canny high threshold (internal)
        param2=param2,                      # vote threshold (lower -> more detections)
        minRadius=minR,
        maxRadius=maxR
    )
    if circles is None:
        return []
    #I want to keep values float for more precise circle drawing
    #circles = np.round(circles[0]).astype(int)  # (x, y, r)
    circles = circles[0]  # (x, y, r)
    # keep biggest/strongest ones if extras appear
    circles = sorted(circles, key=lambda c: c[2], reverse=True)[:expected_n]
    # Number in reading order: rows top->bottom, left->right within each row.
    circles = _reading_order(circles)
    circles_rois = [CircleROI(name=f"{name_prefix}_{i+1:02d}", cx=cx, cy=cy, r=r) for i, (cx, cy, r) in enumerate(circles)]
    return circles_rois  # list of CircleROIs


# NOTE: If arenas are not detected reliably, tune `block_size`, `c_sub`, and the
# circularity/fill thresholds inside `detect_circular_rois`.
def detect_circular_rois_old(
    bg_gray: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 80,
    param1: int = 120,
    param2: int = 35,
    min_radius: int = 30,
    max_radius: int = 250,
    name_prefix: str = "arena",
    binary_thresh: int = 50,
    block_size: int = 51,
    c_sub: int = 5,
    circ_min: float = 0.65,
    fill_min: float = 0.55,
    flatten_illumination: bool = True,
    flatten_sigma: float = 40.0,
    flatten_offset: int = 128,
) -> List[CircleROI]:
    """Detect bright circular arenas on a dark background.

    Notes:
      - `dp`, `param1`, `param2` are kept for backward compatibility (legacy HoughCircles API)
      - `block_size` and `c_sub` control adaptive thresholding
      - `circ_min` and `fill_min` filter candidates by circularity and fill ratio
      - `binary_thresh` is a fixed threshold (0–255) to segment bright arenas from dark background
      - `flatten_illumination` subtracts a large blur to reduce vignetting and improves threshold stability
      - `flatten_sigma` controls how large the blur is (in pixels); larger values remove slower gradients
    """

    if bg_gray is None or bg_gray.size == 0:
        return []

    g = bg_gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    # Denoise and improve local contrast a bit
    g = cv2.GaussianBlur(g, (0, 0), 2)
    try:
        # CLAHE is often more stable than equalizeHist when illumination is uneven
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
    except Exception:
        g = cv2.equalizeHist(g)

    # Optional: flatten illumination/vignetting by subtracting a large blur and renormalizing.
    # This makes thresholding much more stable across the field of view.
    if flatten_illumination:
        sig = float(flatten_sigma)
        if sig > 0:
            bg = cv2.GaussianBlur(g, (0, 0), sig)
            # work in float, add offset to keep mid-gray, then normalize back to uint8
            gf = g.astype(np.float32) - bg.astype(np.float32) + float(flatten_offset)
            g = cv2.normalize(gf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Primary segmentation: fixed binary threshold to eliminate dark background.
    # Assumption: arenas are bright (high pixel values) on a dark background.
    bt = int(binary_thresh)
    bt = max(0, min(255, bt))
    _, th = cv2.threshold(g, bt, 255, cv2.THRESH_BINARY)

    # Clean up mask: remove small specks, fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    # If binary threshold produced a degenerate mask, fall back to adaptive threshold.
    nnz = int(np.count_nonzero(th))
    if th is None or th.size == 0 or nnz == 0 or nnz == th.size:
        # Adaptive threshold fallback (useful when illumination varies strongly)
        # blockSize must be odd and >= 3
        bs = int(block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        cs = int(c_sub)

        th = cv2.adaptiveThreshold(
            g,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            cs,
        )
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    # If thresholding still failed, fall back to Otsu.
    if th is None or th.size == 0 or int(np.count_nonzero(th)) == 0:
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    # Find candidate contours
    cnts, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    out: List[CircleROI] = []
    if not cnts:
        return out

    min_area = math.pi * float(min_radius) * float(min_radius)
    max_area = math.pi * float(max_radius) * float(max_radius)

    candidates: List[Tuple[float, float, float, float]] = []  # (cy, cx, r, score)

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue

        peri = float(cv2.arcLength(c, True))
        if peri <= 1e-6:
            continue

        # Circularity: 1.0 for perfect circle
        circ = 4.0 * math.pi * area / (peri * peri)
        if circ < float(circ_min):
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        r = float(r)
        if r < float(min_radius) or r > float(max_radius):
            continue

        # Extra sanity: contour area should be reasonably close to circle area
        circle_area = math.pi * r * r
        fill = area / max(circle_area, 1e-6)
        if fill < float(fill_min):
            continue

        # Score combines circularity and fill
        score = circ * fill
        candidates.append((float(y), float(x), r, float(score)))

    if not candidates:
        return out

    # Non-maximum suppression by center distance (min_dist) to avoid duplicates
    candidates.sort(key=lambda t: t[3], reverse=True)  # best score first
    kept: List[Tuple[float, float, float, float]] = []
    for cy, cx, r, score in candidates:
        ok = True
        for cy2, cx2, r2, _ in kept:
            if (cx - cx2) * (cx - cx2) + (cy - cy2) * (cy - cy2) < float(min_dist) * float(min_dist):
                ok = False
                break
        if ok:
            kept.append((cy, cx, r, score))

    # Sort top-to-bottom then left-to-right for stable naming
    kept_sorted = sorted(kept, key=lambda t: (t[0], t[1]))
    for i, (cy, cx, r, _score) in enumerate(kept_sorted):
        out.append(CircleROI(name=f"{name_prefix}_{i+1:02d}", cx=float(cx), cy=float(cy), r=float(r)))

    return out


def crop_roi(frame: np.ndarray, roi: CircleROI) -> Tuple[np.ndarray, Tuple[int, int]]:
    # Return ROI crop and offset (x0, y0)
    x0, y0, x1, y1 = roi.bbox()
    crop = frame[y0:y1, x0:x1].copy()
    return crop, (x0, y0)


def roi_mask(shape_hw: Tuple[int, int], roi: CircleROI, offset_xy: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    x0, y0 = offset_xy
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(round(roi.cx - x0))
    cy = int(round(roi.cy - y0))
    r = int(round(roi.r))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask
