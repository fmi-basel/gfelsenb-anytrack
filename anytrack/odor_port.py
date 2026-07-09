"""Odor-port detection — locate the central odor-delivery port in each arena.

The port is a small, static fixture at (near) the arena centre. Localizing it turns
what the background model treats as a nuisance dark blob into a first-class feature:
the odor *source*, so tracking can report fly-to-port distance / approach kinematics
(the readout of a local-search / odor-navigation assay), and the background stage can
tell a stationary fly from the port.

Detection is behind a small backend-agnostic :class:`SegmentEngine`, mirroring the
pose :class:`~anytrack.pose.engine.PoseEngine` pattern:

- :class:`ClassicalPortDetector` (default, zero-dependency) — the port is a dark
  circular fixture near the arena centre; it is found by a circular-edge matched
  filter (strongest outward radial gradient) within a central window and a
  constant-size radius band, not by interior darkness.
- :class:`SamPortDetector` (opt-in, ``port_backend="sam"``) — prompts a Segment
  Anything model at the arena centre and takes the returned mask. Lazily imports
  torch/SAM; falls back to the classical detector if unavailable, so the core install
  stays dependency-light.

Detection runs once per arena on the (static) model background — not per frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .models import CircleROI


@dataclass
class PortDetection:
    """A detected odor port, in full-frame pixels."""
    roi: str
    cx: float                       # full-frame px
    cy: float
    r: float                        # equivalent radius px (sqrt(area/pi))
    conf: float                     # [0,1] rough detection confidence
    backend: str                    # "classical" | "sam"
    offset_frac: float = 0.0        # centroid distance from arena centre, as frac of R

    def as_dict(self) -> Dict[str, float]:
        return dict(roi=self.roi, cx=round(self.cx, 2), cy=round(self.cy, 2),
                    r=round(self.r, 2), conf=round(self.conf, 3), backend=self.backend,
                    offset_frac=round(self.offset_frac, 3))


@runtime_checkable
class SegmentEngine(Protocol):
    """Backend-agnostic port segmenter.

    ``detect_port`` takes a grayscale ROI crop of the model background and the arena
    centre + radius in crop-local pixels, and returns ``(cx, cy, r, conf, mask)`` in
    crop-local pixels (mask is ``(H,W)`` bool), or ``None`` if no port is found.
    """
    name: str

    def detect_port(self, roi_gray: np.ndarray, center: Tuple[float, float], radius: float
                    ) -> Optional[Tuple[float, float, float, float, np.ndarray]]: ...


def _radial_edge_score(gx, gy, cxs, cys, radii, K):
    """Best ``(response, cx, cy, r)`` of a circular-edge matched filter.

    For each candidate centre in the ``cys × cxs`` grid and each radius, scores the
    **mean outward radial image gradient** sampled at ``K`` points around the circle
    (``gx·cosθ + gy·sinθ``). A dark disk on a brighter floor has intensity rising
    outward across its rim, so its boundary gives a strong positive response at the
    true centre/radius — the signature of the odor-port fixture. Returns the argmax
    over the whole grid (``response`` in gray-levels per pixel), or ``None``.
    """
    h, w = gx.shape
    th = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    CX, CY = np.meshgrid(np.asarray(cxs, np.float32), np.asarray(cys, np.float32))
    best = None
    for r in radii:
        acc = np.zeros(CX.shape, np.float32)
        for k in range(K):
            sx = np.clip((CX + r * ct[k]).round().astype(np.intp), 0, w - 1)
            sy = np.clip((CY + r * st[k]).round().astype(np.intp), 0, h - 1)
            acc += gx[sy, sx] * ct[k] + gy[sy, sx] * st[k]
        acc /= K
        i = int(np.argmax(acc))
        resp = float(acc.flat[i])
        if best is None or resp > best[0]:
            best = (resp, float(CX.flat[i]), float(CY.flat[i]), float(r))
    return best


class ClassicalPortDetector:
    """The port = a dark circular fixture near the arena centre, of ~constant size.

    The port is a low-contrast *ring/well* (its interior texture varies), so a
    darkness-blob detector latches onto internal sub-spots and mis-sizes it. This
    detector keys on the fixture's **circular edge** instead, using all three priors:
    - *central*: only centres within ``port_max_center_frac``·R of the arena centre
      are searched, so an off-centre fly can never be taken as the port.
    - *circular*: a matched filter scores each candidate ``(centre, radius)`` by the
      mean outward radial image gradient around that circle (:func:`_radial_edge_score`)
      — robust to fill and to low contrast, and it localizes the rim, not a dark lobe.
    - *constant, small size*: the radius is searched only in ``port_radius_frac_min..
      max``·R; a coarse pass on a 2× downscaled crop localizes the circle and a
      full-resolution local pass refines it. A port is accepted only if the response
      clears ``port_edge_min`` (mean gray-levels/px — an interpretable edge-contrast
      check). (The single shared size across arenas is applied in :func:`detect_ports`.)
    """

    name = "classical"
    K_COARSE = 36
    K_FINE = 64

    def __init__(self, cfg):
        self.edge_min = float(getattr(cfg, "port_edge_min", 4.0))
        self.search_frac = float(getattr(cfg, "port_max_center_frac", 0.30))
        self.rmin_f = float(getattr(cfg, "port_radius_frac_min", 0.04))
        self.rmax_f = float(getattr(cfg, "port_radius_frac_max", 0.14))

    @staticmethod
    def _gradients(img, sigma):
        import cv2
        gs = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
        gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
        return gx, gy

    def detect_port(self, roi_gray, center, radius):
        import cv2
        cx0, cy0 = float(center[0]), float(center[1])
        R = float(radius)
        if roi_gray.size == 0 or R < 8.0:
            return None
        win = self.search_frac * R
        rmin = max(2.0, self.rmin_f * R)
        rmax = max(rmin + 1.0, self.rmax_f * R)
        ds = max(1, int(round(R / 240.0)))          # keep the search grid bounded on big arenas

        # coarse pass on a downscaled crop — localizes the circle cheaply
        g = roi_gray if ds == 1 else cv2.resize(
            roi_gray, (0, 0), fx=1.0 / ds, fy=1.0 / ds, interpolation=cv2.INTER_AREA)
        gx, gy = self._gradients(g, 1.2)
        cxs = np.arange(int((cx0 - win) / ds), int((cx0 + win) / ds) + 1)
        cys = np.arange(int((cy0 - win) / ds), int((cy0 + win) / ds) + 1)
        radii = np.arange(rmin / ds, rmax / ds + 1e-6, 1.0)
        if cxs.size == 0 or cys.size == 0 or radii.size == 0:
            return None
        coarse = _radial_edge_score(gx, gy, cxs, cys, radii, self.K_COARSE)
        if coarse is None:
            return None
        bx, by, br = coarse[1] * ds, coarse[2] * ds, coarse[3] * ds

        # full-resolution local refine around the coarse hit
        gxf, gyf = self._gradients(roi_gray, 2.0)
        w = ds + 1
        cxs = np.arange(int(bx - w), int(bx + w) + 1)
        cys = np.arange(int(by - w), int(by + w) + 1)
        radii = np.arange(max(rmin, br - w), min(rmax, br + w) + 1e-6, 1.0)
        fine = _radial_edge_score(gxf, gyf, cxs, cys, radii, self.K_FINE)
        resp, px, py, r_sel = fine if fine is not None else (coarse[0], bx, by, br)

        if resp < self.edge_min:                    # no clear circular fixture here
            return None
        if np.hypot(px - cx0, py - cy0) > win:      # centre must stay central
            return None
        conf = float(np.clip(resp / (4.0 * self.edge_min), 0.0, 1.0))
        h, w = roi_gray.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        disk = (xx - px) ** 2 + (yy - py) ** 2 <= r_sel * r_sel
        return px, py, float(r_sel), conf, disk


class SamPortDetector:
    """Segment-Anything port segmenter (opt-in). Lazily imports torch + a SAM package.

    Prompts the model with a point at the arena centre and returns the highest-score
    mask. Constructing it raises ImportError with a clear message if the extra isn't
    installed, so :func:`build_segment_engine` can fall back to the classical detector.
    """

    name = "sam"

    def __init__(self, model_path: str, device: str = "auto", variant: str = "vit_b"):
        if not model_path:
            raise ImportError("port_backend='sam' needs sam_model_path (a SAM checkpoint).")
        self._predictor = self._load(model_path, device, variant)

    @staticmethod
    def _resolve_device(device: str):
        import torch
        if device and device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self, model_path, device, variant):
        try:                                            # full SAM, then MobileSAM
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            try:
                from mobile_sam import sam_model_registry, SamPredictor
            except ImportError as e:
                raise ImportError(
                    "port_backend='sam' needs the 'segment-anything' or 'mobile-sam' "
                    "package (install the optional extra: pip install 'anytrack[sam]')."
                ) from e
        dev = self._resolve_device(device)
        sam = sam_model_registry[variant](checkpoint=model_path)
        sam.to(dev)
        return SamPredictor(sam)

    def detect_port(self, roi_gray, center, radius):
        import cv2
        import numpy as np
        rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        self._predictor.set_image(rgb)
        pts = np.array([[center[0], center[1]]], dtype=np.float32)
        masks, scores, _ = self._predictor.predict(
            point_coords=pts, point_labels=np.array([1]), multimask_output=True)
        # pick the highest-scoring mask that is port-plausible (compact, near centre)
        best, best_score = None, -1.0
        for m, s in zip(masks, scores):
            area = int(m.sum())
            if area == 0 or area > 0.25 * np.pi * radius * radius:
                continue
            ys, xs = np.nonzero(m)
            d = float(np.hypot(xs.mean() - center[0], ys.mean() - center[1]))
            if d <= 0.30 * radius and s > best_score:
                best, best_score = m, float(s)
        if best is None:
            return None
        ys, xs = np.nonzero(best)
        r = float(np.sqrt(best.sum() / np.pi))
        return float(xs.mean()), float(ys.mean()), r, float(np.clip(best_score, 0, 1)), best

    def segment_fly(self, roi_gray, seed):
        """Refine a fly mask around a seed point (crop-local px) — used by the two-pass
        stuck-fly repair to get a contrast-independent mask where the classical detector
        can't see a low-contrast wall fly. Returns a bool mask or None."""
        import cv2
        import numpy as np
        self._predictor.set_image(cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB))
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[float(seed[0]), float(seed[1])]], dtype=np.float32),
            point_labels=np.array([1]), multimask_output=True)
        # smallest plausible (fly-sized) mask containing the seed
        best = None
        for m in masks[np.argsort(scores)[::-1]]:
            a = int(m.sum())
            if 20 <= a <= 4000 and bool(m[int(seed[1]), int(seed[0])]):
                best = m.astype(bool)
                break
        return best


def build_segment_engine(cfg) -> SegmentEngine:
    """Resolve the configured port segmenter, falling back to the classical detector
    if the SAM extra isn't installed (so the pipeline always has a working engine)."""
    backend = str(getattr(cfg, "port_backend", "classical") or "classical").lower()
    if backend in ("", "classical"):
        return ClassicalPortDetector(cfg)
    if backend == "sam":
        try:
            return SamPortDetector(
                str(getattr(cfg, "sam_model_path", "") or ""),
                device=str(getattr(cfg, "sam_device", "auto") or "auto"),
                variant=str(getattr(cfg, "sam_variant", "vit_b") or "vit_b"))
        except ImportError as e:
            print(f"[odor_port] SAM backend unavailable ({e}); using classical detector.")
            return ClassicalPortDetector(cfg)
    raise ValueError(f"unknown port_backend {backend!r} (use 'classical' or 'sam')")


def _roi_crop(bg_full: np.ndarray, roi: CircleROI):
    """Grayscale ROI crop of the full-frame background + the arena centre within it."""
    import cv2
    gray = bg_full if bg_full.ndim == 2 else cv2.cvtColor(bg_full, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    x0 = int(max(0, round(roi.cx - roi.r)))
    y0 = int(max(0, round(roi.cy - roi.r)))
    x1 = int(min(W, round(roi.cx + roi.r)))
    y1 = int(min(H, round(roi.cy + roi.r)))
    return np.ascontiguousarray(gray[y0:y1, x0:x1]), x0, y0


def detect_odor_port(bg_full: np.ndarray, roi: CircleROI, cfg, engine: Optional[SegmentEngine] = None
                     ) -> Optional[PortDetection]:
    """Detect the odor port for one ROI on the (static) full-frame model background.

    Returns a :class:`PortDetection` in full-frame pixels, or ``None`` if no central
    port is found."""
    engine = engine or build_segment_engine(cfg)
    crop, x0, y0 = _roi_crop(bg_full, roi)
    if crop.size == 0:
        return None
    center = (roi.cx - x0, roi.cy - y0)
    res = engine.detect_port(crop, center, float(roi.r))
    if res is None:
        return None
    px, py, r, conf, _ = res
    offset = float(np.hypot(px - center[0], py - center[1]) / max(1.0, roi.r))
    return PortDetection(roi=roi.name, cx=x0 + px, cy=y0 + py, r=r, conf=conf,
                         backend=engine.name, offset_frac=offset)


def detect_ports(bg_full: np.ndarray, rois: List[CircleROI], cfg) -> Dict[str, PortDetection]:
    """Detect ports for all ROIs; returns ``{roi_name: PortDetection}`` (found only).

    Applies the *constant size* prior: when ``port_shared_radius`` is set (default),
    every detected port's radius is replaced by the median across arenas, so the
    reported size is consistent instead of varying with each arena's contrast.
    """
    import dataclasses
    engine = build_segment_engine(cfg)
    out = {}
    for roi in rois:
        p = detect_odor_port(bg_full, roi, cfg, engine=engine)
        if p is not None:
            out[roi.name] = p
    if out and getattr(cfg, "port_shared_radius", True):
        r_shared = float(np.median([p.r for p in out.values()]))
        out = {n: dataclasses.replace(p, r=r_shared) for n, p in out.items()}
    return out


def port_distance(x: float, y: float, port: PortDetection) -> float:
    """Euclidean full-frame-pixel distance from a point to the port centre."""
    return float(np.hypot(x - port.cx, y - port.cy))


def add_port_distance(df, ports: Dict[str, PortDetection], rois: List[CircleROI], cfg):
    """Add ``dist_to_port_px`` / ``dist_to_port_mm`` columns to a tracks DataFrame.

    Uses each ROI's px→mm scale (``arena_diameter_mm`` / arena-diameter-px). Rows whose
    ROI has no detected port get NaN. Returns the same df (modified in place)."""
    import numpy as np
    if df.empty or "roi" not in df.columns:
        return df
    rmap = {r.name: r for r in rois}
    dpx = np.full(len(df), np.nan, dtype=float)
    dmm = np.full(len(df), np.nan, dtype=float)
    xr, yr = df["x"].to_numpy(), df["y"].to_numpy()
    rois_col = df["roi"].to_numpy()
    for name, port in ports.items():
        m = rois_col == name
        if not m.any():
            continue
        dpx[m] = np.hypot(xr[m] - port.cx, yr[m] - port.cy)
        roi = rmap.get(name)
        if roi is not None and roi.r > 0:
            mm_per_px = float(getattr(cfg, "arena_diameter_mm", 75.0)) / (2.0 * roi.r)
            dmm[m] = dpx[m] * mm_per_px
    df["dist_to_port_px"] = dpx
    df["dist_to_port_mm"] = dmm
    return df
