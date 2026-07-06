"""anytrack-debug — interactive per-frame detection/tracking inspector.

Opens on a single video, builds each arena's background, tracks it, and lets you
scrub frame-by-frame while inspecting the detection pipeline in two synced,
zoomable panes (pick a stage per pane):

    raw · background · subtracted · threshold · mask · candidates · final(track)

Live sliders (threshold / morphology / area / max-jump) re-render the current
frame instantly; **Re-track** re-runs linking for the selected arena so you can
see how a parameter changes the tracker's decisions.

The core (background, stages, tracking) is plain functions that mirror the
production detector/tracker exactly (``detector.extract_ellipses`` +
``tracker.CentroidTracker``), so what you see is what the tracker does. Decode is
cv2 (random-access scrubbing); the production streaming path decodes via FFmpeg,
which differs only in sub-pixel rim scaling.
"""
from __future__ import annotations

import argparse
import dataclasses
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
import pandas as pd

from ..config import AnyTrackConfig, load_config
from ..models import CircleROI
from ..background import (build_background_image, sample_frames_uniformly,
                          fit_gmm_background)
from ..roi import detect_circular_rois
from ..preprocess import roi_crop_geometry, deghost, _arena_mask  # deghost: canonical impl (also production)
from ..detector import (compute_diff, threshold_fg, build_morph_kernels,
                        _candidates_from_mask, extract_ellipses, centroid_contrast)
from ..tracker import CentroidTracker
from ..coordinates import scaled_to_full
from ..tracking_fast import _rescale_detect_params

STAGES = ["raw", "background", "subtracted", "threshold", "mask", "candidates", "final"]

# ----------------------------------------------------------------------------
# Core (no Tk) — testable
# ----------------------------------------------------------------------------


def crop_roi_gray(full_gray: np.ndarray, roi: CircleROI, downscale: int) -> np.ndarray:
    """Crop ``full_gray`` to ``roi`` and downscale exactly as the tracker's ROI frame."""
    crop_size, x0, y0, scaled = roi_crop_geometry(roi, downscale)
    H, W = full_gray.shape[:2]
    crop = full_gray[y0:min(y0 + crop_size, H), x0:min(x0 + crop_size, W)]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        pad = np.zeros((crop_size, crop_size), np.uint8)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    if scaled != crop_size:
        crop = cv2.resize(crop, (scaled, scaled), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(crop)


BG_METHODS = ["gmm", "percentile", "fg_excluded"]


def roi_uniform_samples(video_path, roi: CircleROI, cfg: AnyTrackConfig):
    """(stack, frame_indices): uniform cv2 samples cropped+downscaled to the ROI.
    ``frame_indices[k]`` is the source frame of ``stack[k]`` (for fg-exclusion)."""
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    samples = sample_frames_uniformly(str(video_path), cfg.gmm_n_samples)  # (N,H,W)
    idxs = np.linspace(0, max(0, n_total - 1), cfg.gmm_n_samples, dtype=int)[:len(samples)]
    stack = np.stack([crop_roi_gray(g, roi, cfg.roi_downscale) for g in samples])
    return stack, idxs


def _bg_from_stack(stack: np.ndarray, cfg: AnyTrackConfig, method: str, percentile: float):
    if method == "percentile":
        return np.ascontiguousarray(np.percentile(stack, percentile, axis=0).astype(np.uint8))
    gmm, _ = fit_gmm_background(stack, bic_improvement=cfg.gmm_bic_improvement,
                                lowp=cfg.gmm_lowp, min_std=cfg.gmm_min_std,
                                reg_covar=cfg.gmm_reg_covar)
    return np.ascontiguousarray(gmm)


def build_roi_background(video_path, roi: CircleROI, cfg: AnyTrackConfig,
                         method: str = "gmm", percentile: float = 90, tracks=None,
                         apply_deghost: bool = False, deghost_percentile: float = 98,
                         deghost_margin: int = 15) -> Optional[np.ndarray]:
    """Per-ROI background from uniform cv2 samples (self-consistent with the
    cv2-decoded frames this GUI displays). Returns scaled-local uint8 or None.

    method: ``gmm`` (brighter-component GMM, current default) · ``percentile``
    (per-pixel high percentile — robust to dwell, can't overshoot brighter than
    an observed arena pixel) · ``fg_excluded`` (per-pixel percentile computed
    only from frames where the tracked fly is far from that pixel — removes
    dwell ghosts; needs ``tracks``, else falls back to plain percentile).

    ``apply_deghost`` runs the fixture-safe :func:`deghost` post-pass to lift any
    residual baked-in fly blob to the arena brightness (for very-long-dwell flies
    the other methods can't exclude)."""
    try:
        stack, idxs = roi_uniform_samples(video_path, roi, cfg)
    except Exception:
        return None
    try:
        if method == "fg_excluded":
            bg = _bg_fg_excluded(stack, idxs, cfg, tracks, percentile)
        else:
            bg = _bg_from_stack(stack, cfg, method, percentile)
        if bg is not None and apply_deghost:
            bg = deghost(bg, stack, deghost_percentile, deghost_margin)
        return bg
    except Exception:
        return None


def _bg_fg_excluded(stack, idxs, cfg, tracks, percentile, radius=None):
    """Per-pixel percentile with the tracked fly masked out of each sample."""
    if tracks is None or not len(tracks):
        return _bg_from_stack(stack, cfg, "percentile", percentile)
    scale = float(cfg.roi_downscale)
    sc = stack.shape[1]
    if radius is None:
        radius = int(max(6, getattr(cfg, "bg_protect_radius_px", 24) / scale))
    tf = tracks.set_index("frame")
    fstack = stack.astype(np.float32)
    yy, xx = np.mgrid[0:sc, 0:sc]
    for k, fidx in enumerate(idxs):
        if int(fidx) in tf.index:
            r = tf.loc[int(fidx)]
            m = (xx - float(r["xs"])) ** 2 + (yy - float(r["ys"])) ** 2 <= radius * radius
            fstack[k][m] = np.nan
    bg = np.nanpercentile(fstack, percentile, axis=0)
    nan = np.isnan(bg)
    if nan.any():                                   # pixel always covered → plain percentile
        bg[nan] = np.percentile(stack, percentile, axis=0)[nan]
    return np.ascontiguousarray(bg.astype(np.uint8))


def _pick_candidates(cands, contrasts, pred, max_jump_scaled, contrast_gate, darkness_weight):
    """Gate low-contrast candidates and, if darkness_weight>0, reduce to the single
    best-scoring one; return the list to hand to CentroidTracker.step (which then
    picks nearest-to-prediction among survivors). gate=weight=0 → unchanged."""
    kept = [(c, ct) for c, ct in zip(cands, contrasts)
            if not (contrast_gate > 0 and (ct != ct or ct < contrast_gate))]
    if not kept or darkness_weight <= 0 or pred is None:
        return [c for c, _ in kept]
    cmax = max((ct for _, ct in kept if ct == ct), default=1.0) or 1.0

    def score(c, ct):
        d = np.hypot(c.x - pred[0], c.y - pred[1]) / max(1e-6, max_jump_scaled)
        return d - darkness_weight * ((ct / cmax) if ct == ct else 0.0)
    return [min(kept, key=lambda t: score(*t))[0]]


def detect_stages(roi_gray: np.ndarray, bg: np.ndarray, cfg: AnyTrackConfig) -> dict:
    """All detector artifacts for one ROI frame: diff, pre-morph threshold,
    post-morph mask, and the area-filtered candidates (mirrors detector.py)."""
    kopen, kclose = build_morph_kernels(cfg)
    diff = compute_diff(roi_gray, bg, cfg)
    thr = threshold_fg(diff, cfg)                       # pre-morph
    mask = threshold_fg(diff, cfg, kopen, kclose)       # post-morph
    candidates = _candidates_from_mask(mask, cfg)
    return {"diff": diff, "threshold": thr, "mask": mask, "candidates": candidates}


def track_roi(video_path, roi: CircleROI, bg: np.ndarray, cfg: AnyTrackConfig,
              progress=None, contrast_gate: float = 0.0,
              darkness_weight: float = 0.0) -> pd.DataFrame:
    """Detect + link one arena over the whole video (cv2 frames), mirroring
    tracking_fast._track_from_frames. Returns per-frame rows with scaled-local
    (xs, ys) for overlay, full-frame (x, y), n_candidates and jump distance.

    ``contrast_gate`` drops candidates whose background-diff contrast is below it
    (kills faint dwell ghosts); ``darkness_weight`` biases the pick toward the
    darkest/highest-contrast candidate. Both 0 → production behaviour."""
    scale = float(cfg.roi_downscale)
    _, x0, y0, _ = roi_crop_geometry(roi, cfg.roi_downscale)
    rcfg = _rescale_detect_params(cfg, scale)
    kopen, kclose = build_morph_kernels(rcfg)
    max_jump_scaled = rcfg.max_jump_px / scale
    tracker = CentroidTracker(center_xy=(roi.r / scale, roi.r / scale),
                              max_jump=max_jump_scaled,
                              miss_tolerance=rcfg.miss_tolerance,
                              use_kalman=rcfg.use_kalman)
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    rows, prev = [], None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        roi_gray = crop_roi_gray(gray, roi, cfg.roi_downscale)
        cands = extract_ellipses(roi_gray, bg, rcfg, kernel_open=kopen, kernel_close=kclose)
        contrasts = [centroid_contrast(roi_gray, bg, c.x, c.y, rcfg) for c in cands]
        passed = _pick_candidates(cands, contrasts, tracker.last, max_jump_scaled,
                                  contrast_gate, darkness_weight)
        chosen, xf, yf = tracker.step(passed)
        if chosen is not None:
            jump = float(np.hypot(xf - prev[0], yf - prev[1])) if prev else 0.0
            prev = (xf, yf)
            xfull, yfull = scaled_to_full(xf, yf, scale, x0, y0)
            rows.append(dict(frame=i, t_s=i / fps, xs=float(xf), ys=float(yf),
                             x=float(xfull), y=float(yfull), track_id=1,
                             n_candidates=len(cands), area=float(chosen.area * scale * scale),
                             jump_px=jump * scale))
        if progress is not None and i % 500 == 0:
            progress(i, n)
        i += 1
    cap.release()
    if progress is not None:
        progress(n, n)
    return pd.DataFrame(rows)


def track_all_rois(video_path, rois: List[CircleROI], bgs: Dict[str, np.ndarray],
                   cfg: AnyTrackConfig, progress=None, contrast_gate: float = 0.0,
                   darkness_weight: float = 0.0) -> Dict[str, pd.DataFrame]:
    """Detect + link every arena in ONE decode pass (vs a full decode per arena).

    Each frame is decoded once, cropped to all ROIs, and fed to that ROI's
    tracker. Same detection/linking as :func:`track_roi` (incl. contrast_gate /
    darkness_weight). ROIs without a background are skipped."""
    scale = float(cfg.roi_downscale)
    rcfg = _rescale_detect_params(cfg, scale)
    kopen, kclose = build_morph_kernels(rcfg)
    max_jump_scaled = rcfg.max_jump_px / scale
    state = {}
    for roi in rois:
        if bgs.get(roi.name) is None:
            continue
        _, x0, y0, _ = roi_crop_geometry(roi, cfg.roi_downscale)
        state[roi.name] = dict(
            roi=roi, x0=x0, y0=y0, bg=bgs[roi.name], prev=None, rows=[],
            tracker=CentroidTracker(center_xy=(roi.r / scale, roi.r / scale),
                                    max_jump=max_jump_scaled,
                                    miss_tolerance=rcfg.miss_tolerance,
                                    use_kalman=rcfg.use_kalman))
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        for s in state.values():
            roi_gray = crop_roi_gray(gray, s["roi"], cfg.roi_downscale)
            cands = extract_ellipses(roi_gray, s["bg"], rcfg, kernel_open=kopen, kernel_close=kclose)
            contrasts = [centroid_contrast(roi_gray, s["bg"], c.x, c.y, rcfg) for c in cands]
            passed = _pick_candidates(cands, contrasts, s["tracker"].last, max_jump_scaled,
                                      contrast_gate, darkness_weight)
            chosen, xf, yf = s["tracker"].step(passed)
            if chosen is None:
                continue
            jump = float(np.hypot(xf - s["prev"][0], yf - s["prev"][1])) if s["prev"] else 0.0
            s["prev"] = (xf, yf)
            xfull, yfull = scaled_to_full(xf, yf, scale, s["x0"], s["y0"])
            s["rows"].append(dict(frame=i, t_s=i / fps, xs=float(xf), ys=float(yf),
                                  x=float(xfull), y=float(yfull), track_id=1,
                                  n_candidates=len(cands), area=float(chosen.area * scale * scale),
                                  jump_px=jump * scale))
        if progress is not None and i % 500 == 0:
            progress(i, n)
        i += 1
    cap.release()
    if progress is not None:
        progress(n, n)
    return {name: pd.DataFrame(s["rows"]) for name, s in state.items()}


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def render_stage(stage: str, roi_gray: np.ndarray, bg: np.ndarray, cfg: AnyTrackConfig,
                 tracks: Optional[pd.DataFrame], frame_idx: int, trail: int = 40,
                 contrast_gate: float = 0.0) -> np.ndarray:
    """Return a BGR image for ``stage`` (scaled-ROI resolution) with overlays."""
    if stage == "raw":
        return _to_bgr(roi_gray)
    if stage == "background":
        return _to_bgr(bg)
    st = detect_stages(roi_gray, bg, cfg)
    if stage == "subtracted":
        return _to_bgr(st["diff"])
    if stage == "threshold":
        return _to_bgr(st["threshold"])
    if stage == "mask":
        return _to_bgr(st["mask"])
    if stage == "candidates":
        # Each area-valid blob is annotated with its contrast; green = passes the
        # contrast gate (a real fly), orange = gated out (a faint dwell ghost),
        # red = rejected by area. Lets you dial the gate against real ghosts.
        vis = _to_bgr(roi_gray)
        cnts, _ = cv2.findContours(st["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (cfg.expected_fly_area_min <= area <= cfg.expected_fly_area_max):
                cv2.drawContours(vis, [c], -1, (0, 0, 230), 1, cv2.LINE_AA)   # rejected by area
                continue
            if len(c) < 5:
                continue
            (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
            ct = centroid_contrast(roi_gray, bg, ex, ey, cfg)
            gated_out = contrast_gate > 0 and (ct != ct or ct < contrast_gate)
            col = (0, 140, 255) if gated_out else (0, 220, 0)
            cv2.ellipse(vis, ((ex, ey), (MA, ma), ang), col, 1, cv2.LINE_AA)
            cv2.circle(vis, (int(ex), int(ey)), 2, col, -1, cv2.LINE_AA)
            cv2.putText(vis, f"{ct:.0f}", (int(ex) + 4, int(ey) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        return vis
    # final: raw + tracked centroid + trail + jump marker
    vis = _to_bgr(roi_gray)
    rad = max(7, roi_gray.shape[0] // 45)   # marker size scales with ROI size
    if tracks is None or not len(tracks):
        cv2.putText(vis, "no track (still computing?)", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        return vis
    past = tracks[(tracks["frame"] <= frame_idx) & (tracks["frame"] > frame_idx - trail)]
    pts = past[["xs", "ys"]].to_numpy().astype(np.int32)
    for j in range(1, len(pts)):
        cv2.line(vis, tuple(pts[j - 1]), tuple(pts[j]), (255, 180, 0), 2, cv2.LINE_AA)
    cur = tracks[tracks["frame"] == frame_idx]
    if len(cur):                                   # detected on this frame
        r = cur.iloc[0]
        c = (int(r["xs"]), int(r["ys"]))           # "xs" is a reserved pandas method — index with []
        jumped = bool(r["jump_px"] > cfg.max_jump_px)
        cv2.circle(vis, c, rad, (0, 0, 255) if jumped else (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, c, (255, 255, 255), cv2.MARKER_CROSS, rad + 4, 1)
    elif len(past):                                # missed here → dim last-known position
        c = tuple(int(v) for v in past[["xs", "ys"]].to_numpy()[-1])
        cv2.circle(vis, c, rad, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "no detection this frame", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    return vis


# ----------------------------------------------------------------------------
# Tk shell
# ----------------------------------------------------------------------------


def _build_app(video_path: Path, cfg: AnyTrackConfig):
    """Construct the DebugApp (imported lazily so the core stays Tk-free/testable)."""
    import tkinter as tk
    import ttkbootstrap as tb
    from ttkbootstrap.constants import HORIZONTAL
    from .app import ZoomableImageViewer

    class DebugApp(tb.Window):
        def __init__(self):
            super().__init__(themename="darkly")
            self.title(f"anytrack-debug · {video_path.name}")
            self.geometry("1400x900")
            self.cfg0 = cfg
            self.work = dataclasses.replace(cfg)   # live-editable working config
            # Experiment knobs for the dwell-contamination A/B (background method +
            # candidate gating). Defaults reproduce the shipped pipeline.
            self.opts = dict(bg_method="gmm", percentile=90, contrast_gate=0.0,
                             darkness_weight=0.0, deghost=False, deghost_margin=15)
            self.cap = cv2.VideoCapture(str(video_path))
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
            self.frame_idx = 0
            self._frame_cache: Dict[int, np.ndarray] = {}
            self.rois: List[CircleROI] = []
            self.roi_bg: Dict[str, np.ndarray] = {}
            self.tracks: Dict[str, pd.DataFrame] = {}
            self.cur_roi: Optional[str] = None
            self._playing = False
            self._syncing = False
            self._build_ui()
            self.after(50, self._start_worker)

        # ---- UI ----
        def _build_ui(self):
            top = tb.Frame(self); top.pack(side="top", fill="x", padx=8, pady=6)
            tb.Label(top, text="Arena:").pack(side="left")
            self.arena_var = tk.StringVar()
            self.arena_cb = tb.Combobox(top, textvariable=self.arena_var, width=12, state="readonly")
            self.arena_cb.pack(side="left", padx=(4, 16))
            self.arena_cb.bind("<<ComboboxSelected>>", lambda e: self._on_arena())
            self.stage_vars = [tk.StringVar(value="subtracted"), tk.StringVar(value="final")]
            self.stage_cbs = []
            for i, sv in enumerate(self.stage_vars):
                tb.Label(top, text=f"Pane {i+1}:").pack(side="left")
                cb = tb.Combobox(top, textvariable=sv, values=STAGES, width=12, state="readonly")
                cb.pack(side="left", padx=(4, 12))
                cb.bind("<<ComboboxSelected>>", lambda e: self._render())
                self.stage_cbs.append(cb)
            self.retrack_btn = tb.Button(top, text="Re-track", bootstyle="warning", command=self._retrack)
            self.retrack_btn.pack(side="right")

            body = tb.Frame(self); body.pack(side="top", fill="both", expand=True, padx=8)
            panes = tb.Frame(body); panes.pack(side="left", fill="both", expand=True)
            self.viewers = []
            for i in range(2):
                f = tb.Frame(panes); f.pack(side="left", fill="both", expand=True, padx=(0, 4))
                v = ZoomableImageViewer(f, on_view_change=lambda z, px, py, i=i: self._sync(i, z, px, py))
                v.canvas.pack(fill="both", expand=True)
                # Re-render on resize: the canvas starts at 1x1, so a render fired
                # before layout draws nothing; this repaints once it has a real size.
                v.canvas.bind("<Configure>", lambda e, vv=v: vv.render())
                self.viewers.append(v)

            side = tb.Labelframe(body, text="Detection params", padding=8)
            side.pack(side="right", fill="y", padx=(8, 0))
            self._sliders = {}
            self._add_slider(side, "thr_fixed", 0, 100, self.work.thr_fixed)
            self._add_slider(side, "morph_open", 0, 15, self.work.morph_open)
            self._add_slider(side, "morph_close", 0, 15, self.work.morph_close)
            self._add_slider(side, "expected_fly_area_min", 0, 800, self.work.expected_fly_area_min)
            self._add_slider(side, "expected_fly_area_max", 100, 5000, self.work.expected_fly_area_max)
            self._add_slider(side, "max_jump_px", 5, 200, int(self.work.max_jump_px))
            tb.Label(side, text="(sliders re-render this frame;\nRe-track applies to linking)",
                     font=("", 9), bootstyle="secondary").pack(pady=(8, 0))

            # ---- dwell-contamination experiments ----
            exp = tb.Labelframe(body, text="Experiments (dwell)", padding=8)
            exp.pack(side="right", fill="y", padx=(8, 0))
            tb.Label(exp, text="background", font=("", 9), anchor="w").pack(fill="x")
            self.bg_method_var = tk.StringVar(value=self.opts["bg_method"])
            bm = tb.Combobox(exp, textvariable=self.bg_method_var, values=BG_METHODS,
                             width=13, state="readonly")
            bm.pack(fill="x", pady=(0, 6))
            bm.bind("<<ComboboxSelected>>", lambda e: self._on_bg_method())
            self._add_opt_slider(exp, "percentile", 50, 99, self.opts["percentile"])
            self.deghost_var = tk.BooleanVar(value=self.opts["deghost"])
            tb.Checkbutton(exp, text="de-ghost (lift baked-in fly)", variable=self.deghost_var,
                           bootstyle="round-toggle", command=self._on_deghost).pack(fill="x", pady=(6, 0))
            self._add_opt_slider(exp, "deghost_margin", 5, 60, self.opts["deghost_margin"])
            self._add_opt_slider(exp, "contrast_gate", 0, 60, self.opts["contrast_gate"])
            self._add_opt_slider(exp, "darkness_weight", 0, 300, self.opts["darkness_weight"], div=100.0)
            tb.Label(exp, text="(bg/percentile/de-ghost rebuild+retrack;\ncontrast_gate previews on candidates;\ngate/weight apply on Re-track)",
                     font=("", 8), bootstyle="secondary").pack(pady=(8, 0))

            bot = tb.Frame(self); bot.pack(side="bottom", fill="x", padx=8, pady=6)
            self.play_btn = tb.Button(bot, text="▶", width=3, command=self._toggle_play)
            self.play_btn.pack(side="left")
            tb.Button(bot, text="◀", width=3, command=lambda: self._seek(self.frame_idx - 1)).pack(side="left", padx=2)
            tb.Button(bot, text="▶|", width=3, command=lambda: self._seek(self.frame_idx + 1)).pack(side="left", padx=2)
            self.slider = tb.Scale(bot, from_=0, to=max(0, self.n - 1), orient=HORIZONTAL,
                                   command=lambda v: self._seek(int(float(v)), from_slider=True))
            self.slider.pack(side="left", fill="x", expand=True, padx=8)
            self.status = tb.Label(bot, text="loading…", width=52, anchor="w")
            self.status.pack(side="right")

        def _add_slider(self, parent, name, lo, hi, val):
            row = tb.Frame(parent); row.pack(fill="x", pady=3)
            tb.Label(row, text=name, width=20, anchor="w", font=("", 9)).pack(side="left")
            var = tk.IntVar(value=int(val))
            tb.Label(row, textvariable=var, width=5).pack(side="right")
            s = tb.Scale(parent, from_=lo, to=hi, value=int(val), orient=HORIZONTAL,
                         command=lambda v, n=name, vr=var: self._on_param(n, float(v), vr))
            s.pack(fill="x")
            self._sliders[name] = var

        def _add_opt_slider(self, parent, name, lo, hi, val, div=1.0):
            """Slider bound to self.opts[name] (div>1 → fractional value)."""
            row = tb.Frame(parent); row.pack(fill="x", pady=3)
            tb.Label(row, text=name, width=16, anchor="w", font=("", 9)).pack(side="left")
            var = tk.StringVar(value=f"{val:g}")
            tb.Label(row, textvariable=var, width=6).pack(side="right")
            tb.Scale(parent, from_=lo, to=hi, value=float(val) * div, orient=HORIZONTAL,
                     command=lambda v, n=name, vr=var, d=div: self._on_opt(n, float(v) / d, vr)).pack(fill="x")

        # ---- worker: background + ROIs + initial tracking ----
        def _start_worker(self):
            threading.Thread(target=self._worker, daemon=True).start()

        def _worker(self):
            # Progressive: raw view appears after ROIs (~seconds); stages after the
            # per-ROI backgrounds; the track overlay after one shared decode pass.
            try:
                self._set_status("building background + detecting arenas…")
                bg_img = build_background_image(str(video_path), self.cfg0)
                rois = detect_circular_rois(
                    bg_img, dp=self.cfg0.roi_hough_dp, min_dist_ratio=self.cfg0.roi_hough_min_dist_ratio,
                    param1=self.cfg0.roi_hough_param1, param2=self.cfg0.roi_hough_param2,
                    min_radius_ratio=self.cfg0.min_radius_ratio, max_radius_ratio=self.cfg0.max_radius_ratio)
                rois = sorted(rois, key=lambda r: r.name)
                if not rois:
                    self._set_status("no arenas detected — check ROI/background settings")
                    return
                self.rois = rois
                self.after(0, self._on_rois)            # show raw immediately
                for roi in rois:
                    self._set_status(f"building background: {roi.name}…")
                    self.roi_bg[roi.name] = build_roi_background(video_path, roi, self.cfg0)
                self.after(0, self._render)             # subtracted/threshold/mask/candidates now work
                self._set_status("tracking all arenas (one decode pass)…")
                self.tracks = track_all_rois(
                    video_path, rois, self.roi_bg, self.cfg0,
                    progress=lambda i, n: self._set_status(f"tracking: frame {i}/{n}"))
                self.after(0, self._on_ready)           # final track overlay now works
            except Exception as e:  # never leave the window blank/silent
                import traceback; traceback.print_exc()
                self._set_status(f"error: {type(e).__name__}: {e}")

        def _on_rois(self):
            names = [r.name for r in self.rois]
            self.arena_cb.configure(values=names)
            if names and not self.cur_roi:
                self.arena_var.set(names[0]); self.cur_roi = names[0]
            self._render()

        def _on_ready(self):
            self._set_status("ready")
            self._render()

        # ---- interaction ----
        def _current_roi(self) -> Optional[CircleROI]:
            return next((r for r in self.rois if r.name == self.cur_roi), None)

        def _full_gray(self, idx: int) -> Optional[np.ndarray]:
            idx = max(0, min(self.n - 1, idx))
            if idx in self._frame_cache:
                return self._frame_cache[idx]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok:
                return None
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if len(self._frame_cache) > 8:
                self._frame_cache.clear()
            self._frame_cache[idx] = g
            return g

        def _render(self):
            roi = self._current_roi()
            if roi is None:
                return
            g = self._full_gray(self.frame_idx)
            if g is None:
                return
            roi_gray = crop_roi_gray(g, roi, self.work.roi_downscale)
            bg = self.roi_bg.get(roi.name)
            trk = self.tracks.get(roi.name)
            for v, sv in zip(self.viewers, self.stage_vars):
                # Background not built yet → only "raw" is renderable; show it so the
                # window is never blank while backgrounds/tracking are still computing.
                stage = sv.get() if bg is not None else "raw"
                img = render_stage(stage, roi_gray, bg, self.work, trk, self.frame_idx,
                                   contrast_gate=self.opts["contrast_gate"])
                v.set_image(img)
            if bg is not None:
                self._update_status(roi_gray, bg, trk)

        def _update_status(self, roi_gray, bg, trk):
            st = detect_stages(roi_gray, bg, self.work)
            nc = len(st["candidates"])
            row = trk[trk.frame == self.frame_idx] if trk is not None else None
            extra = ""
            if row is not None and len(row):
                r = row.iloc[0]
                extra = f" · track(area={r.area:.0f}, jump={r.jump_px:.1f}px)"
            self._set_status(f"frame {self.frame_idx}/{self.n-1} · {self.cur_roi} · {nc} candidate(s){extra}")

        def _seek(self, idx, from_slider=False):
            self.frame_idx = max(0, min(self.n - 1, int(idx)))
            if not from_slider:
                self.slider.set(self.frame_idx)
            self._render()

        def _on_arena(self):
            self.cur_roi = self.arena_var.get()
            self._render()

        def _on_param(self, name, value, var):
            iv = int(round(value))
            var.set(iv)
            if name == "max_jump_px":
                self.work = dataclasses.replace(self.work, max_jump_px=float(iv))
            else:
                self.work = dataclasses.replace(self.work, **{name: iv})
            self._render()

        def _on_opt(self, name, value, var):
            var.set(f"{value:g}")
            self.opts[name] = value
            if name in ("percentile", "deghost_margin"):
                self._rebuild_bg()          # bg depends on these → rebuild + retrack
            else:                            # contrast_gate previews live; both apply on Re-track
                self._render()

        def _on_bg_method(self):
            self.opts["bg_method"] = self.bg_method_var.get()
            self._rebuild_bg()

        def _on_deghost(self):
            self.opts["deghost"] = bool(self.deghost_var.get())
            self._rebuild_bg()

        def _rebuild_bg(self):
            """Rebuild the current arena's background with the selected method, then
            re-track it (bg change ⇒ track change). Threaded."""
            roi = self._current_roi()
            if roi is None:
                return
            self.retrack_btn.configure(state="disabled")

            def job():
                m, p = self.opts["bg_method"], self.opts["percentile"]
                dg = self.opts["deghost"]
                self._set_status(f"rebuilding {roi.name} bg ({m}{', de-ghost' if dg else ''})…")
                bg = build_roi_background(video_path, roi, self.cfg0, method=m, percentile=p,
                                          tracks=self.tracks.get(roi.name),
                                          apply_deghost=dg, deghost_margin=self.opts["deghost_margin"])
                if bg is not None:
                    self.roi_bg[roi.name] = bg
                    self._set_status(f"re-tracking {roi.name}…")
                    self.tracks[roi.name] = track_roi(
                        video_path, roi, bg, self.work,
                        contrast_gate=self.opts["contrast_gate"],
                        darkness_weight=self.opts["darkness_weight"])
                self.after(0, lambda: (self.retrack_btn.configure(state="normal"),
                                       self._set_status(f"ready · {roi.name} · bg={m}{'+dg' if dg else ''}"),
                                       self._render()))
            threading.Thread(target=job, daemon=True).start()

        def _retrack(self):
            roi = self._current_roi()
            if roi is None:
                return
            self.retrack_btn.configure(state="disabled")

            def job():
                df = track_roi(video_path, roi, self.roi_bg[roi.name], self.work,
                               contrast_gate=self.opts["contrast_gate"],
                               darkness_weight=self.opts["darkness_weight"],
                               progress=lambda i, n: self._set_status(f"re-tracking {roi.name}: {i}/{n}"))
                self.tracks[roi.name] = df
                self.after(0, lambda: (self.retrack_btn.configure(state="normal"), self._render()))
            threading.Thread(target=job, daemon=True).start()

        def _toggle_play(self):
            self._playing = not self._playing
            self.play_btn.configure(text="⏸" if self._playing else "▶")
            if self._playing:
                self._advance()

        def _advance(self):
            if not self._playing:
                return
            if self.frame_idx >= self.n - 1:
                self._playing = False; self.play_btn.configure(text="▶"); return
            self._seek(self.frame_idx + 1)
            self.after(int(1000 / self.fps), self._advance)

        def _sync(self, src, zoom, px, py):
            if self._syncing:
                return
            self._syncing = True
            self.viewers[1 - src].set_view_from_sync(zoom, px, py)
            self._syncing = False

        def _set_status(self, msg):
            try:
                self.after(0, lambda: self.status.configure(text=msg))
            except Exception:
                pass

    return DebugApp()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="anytrack-debug",
                                 description="Interactive per-frame detection/tracking inspector.")
    ap.add_argument("video", type=Path, help="Path to a video file")
    args = ap.parse_args(argv)
    if not args.video.exists():
        ap.error(f"video not found: {args.video}")
    cfg = load_config()   # reads the standard anytrack config
    app = _build_app(args.video, cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
