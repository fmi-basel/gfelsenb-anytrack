"""anytrack-arena-debug — a quick single-arena tracking inspector.

A stripped-down companion to ``anytrack-debug`` for eyeballing ONE arena:

  * a dropdown to pick the image-processing **stage** shown in the canvas
    (raw / background / subtracted / threshold / mask), and
  * independently **toggleable, colour-customizable overlays** drawn on top of
    that stage — candidate blobs, the tracked centroid, the Kalman prediction,
    its acceptance-gate circle, and the trail.

No parameter sliders: it runs the production detection + linking (via
``CentroidTracker``) once over the chosen arena and lets you scrub frames and
flip overlays. The core below (stage/overlay rendering + the debug tracking
pass) is plain functions with no Tk, so it is unit-tested headless; the Tk shell
is imported lazily in :func:`main`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from ..config import AnyTrackConfig, load_config
from ..models import CircleROI
from ..background import build_background_image
from ..roi import detect_circular_rois
from ..preprocess import roi_crop_geometry
from ..detector import build_morph_kernels, extract_ellipses, inscribed_disk_mask
from ..tracker import CentroidTracker
from ..tracking_fast import _rescale_detect_params
from .debug_app import crop_roi_gray, build_roi_background, detect_stages, _to_bgr

# Image-processing steps selectable in the stage dropdown. Overlays (below) are
# drawn on top of whichever of these is shown.
STAGES = ["raw", "background", "subtracted", "threshold", "mask"]

# (key, label, default BGR colour). Each is an independent toggle + colour.
OVERLAYS: List[Tuple[str, str, Tuple[int, int, int]]] = [
    ("candidates", "Candidate blobs",   (0, 220, 0)),
    ("centroid",   "Tracked centroid",  (0, 0, 255)),
    ("prediction", "Kalman prediction", (255, 180, 0)),
    ("gate",       "Acceptance gate",   (0, 180, 255)),
    ("trail",      "Trail",             (255, 120, 0)),
]

DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {k: c for k, _, c in OVERLAYS}
DEFAULT_ON: Dict[str, bool] = {k: True for k, _, _ in OVERLAYS}


# ----------------------------------------------------------------------------
# Core (no Tk) — testable
# ----------------------------------------------------------------------------


def track_arena_debug(video_path, roi: CircleROI, bg: np.ndarray, cfg: AnyTrackConfig,
                      progress=None) -> List[dict]:
    """Detect + link one arena over the whole video, recording per-frame the
    exact state the debug overlays need. Mirrors the production detection/linking
    (``tracking_fast._track_from_frames`` / ``debug_app.track_roi``) and reads the
    tracker's recorded prediction + gate.

    Returns a list indexed by decoded frame; each entry is::

        {frame, chosen: (x, y) | None, pred: (x, y) | None, gate: float,
         missed: bool, cands: [(x, y, major, minor, angle_deg), ...]}

    Coordinates are scaled-local (the frame the canvas shows)."""
    scale = float(cfg.roi_downscale)
    _, _, _, scaled = roi_crop_geometry(roi, cfg.roi_downscale)
    rcfg = _rescale_detect_params(cfg, scale)
    kopen, kclose = build_morph_kernels(rcfg)
    arena_mask = (inscribed_disk_mask((scaled, scaled))
                  if getattr(cfg, "detect_arena_mask", True) else None)
    tracker = CentroidTracker(center_xy=(roi.r / scale, roi.r / scale),
                              max_jump=rcfg.max_jump_px / scale,
                              miss_tolerance=rcfg.miss_tolerance,
                              use_kalman=rcfg.use_kalman)
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: List[dict] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        roi_gray = crop_roi_gray(gray, roi, cfg.roi_downscale)
        cands = extract_ellipses(roi_gray, bg, rcfg, mask=arena_mask,
                                 kernel_open=kopen, kernel_close=kclose)
        chosen, xf, yf = tracker.step(cands)
        out.append(dict(
            frame=i,
            chosen=((float(xf), float(yf)) if chosen is not None else None),
            pred=(tuple(float(v) for v in tracker.last_pred) if tracker.last_pred else None),
            gate=float(tracker.last_gate),
            missed=chosen is None,
            cands=[(float(c.x), float(c.y), float(c.major), float(c.minor),
                    float(c.cv_angle_deg)) for c in cands],
        ))
        if progress is not None and i % 500 == 0:
            progress(i, n)
        i += 1
    cap.release()
    if progress is not None:
        progress(len(out), len(out))
    return out


def _base_stage(stage: str, roi_gray: np.ndarray, bg: np.ndarray,
                cfg: AnyTrackConfig) -> np.ndarray:
    """The selected image-processing step as a BGR image (no overlays)."""
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
    return _to_bgr(roi_gray)


def render_arena_frame(roi_gray: np.ndarray, bg: np.ndarray, cfg: AnyTrackConfig,
                       dbg_frame: Optional[dict], trail_pts, *, stage: str,
                       overlays_on: Dict[str, bool],
                       colors: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """Compose the ``stage`` image with the enabled overlays in their colours.

    ``dbg_frame`` is one entry from :func:`track_arena_debug` (or None);
    ``trail_pts`` is a list of scaled-local ``(x, y)`` for the trail. ``colors``
    are BGR. Returns a BGR image at scaled-ROI resolution."""
    vis = _base_stage(stage, roi_gray, bg, cfg)
    sc = roi_gray.shape[0]
    rad = max(5, sc // 45)          # marker size scales with the ROI

    def col(key):
        return tuple(int(v) for v in colors.get(key, DEFAULT_COLORS[key]))

    # trail first, so markers sit on top of it
    if overlays_on.get("trail") and trail_pts is not None and len(trail_pts) >= 2:
        pts = np.asarray(trail_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, col("trail"), 2, cv2.LINE_AA)

    if dbg_frame is None:
        return vis

    if overlays_on.get("candidates"):
        c = col("candidates")
        for (cx, cy, MA, ma, ang) in dbg_frame.get("cands", []):
            cv2.ellipse(vis, ((cx, cy), (MA, ma), ang), c, 1, cv2.LINE_AA)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 2, c, -1, cv2.LINE_AA)

    pred = dbg_frame.get("pred")
    if overlays_on.get("gate") and pred is not None and dbg_frame.get("gate", 0) > 0:
        cv2.circle(vis, (int(round(pred[0])), int(round(pred[1]))),
                   int(round(dbg_frame["gate"])), col("gate"), 1, cv2.LINE_AA)
    if overlays_on.get("prediction") and pred is not None:
        p = (int(round(pred[0])), int(round(pred[1])))
        cv2.drawMarker(vis, p, col("prediction"), cv2.MARKER_TILTED_CROSS, rad + 2, 1, cv2.LINE_AA)

    chosen = dbg_frame.get("chosen")
    if overlays_on.get("centroid"):
        c = col("centroid")
        if chosen is not None:
            pc = (int(round(chosen[0])), int(round(chosen[1])))
            cv2.circle(vis, pc, rad, c, 2, cv2.LINE_AA)
            cv2.drawMarker(vis, pc, c, cv2.MARKER_CROSS, rad + 4, 1, cv2.LINE_AA)
        elif trail_pts is not None and len(trail_pts):     # missed → dim last-known
            last = trail_pts[-1]
            cv2.circle(vis, (int(round(last[0])), int(round(last[1]))), rad, c, 1, cv2.LINE_AA)
    return vis


def trail_up_to(dbg: List[dict], frame_idx: int, trail_len: int) -> List[Tuple[float, float]]:
    """Chosen-centroid positions for frames ``(frame_idx-trail_len, frame_idx]``
    (``trail_len < 0`` = full trace), skipping frames with no detection."""
    lo = 0 if trail_len < 0 else max(0, frame_idx - trail_len)
    pts = []
    for j in range(lo, min(frame_idx + 1, len(dbg))):
        ch = dbg[j].get("chosen")
        if ch is not None:
            pts.append(ch)
    return pts


def _hex_to_bgr(hx: str) -> Tuple[int, int, int]:
    hx = hx.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return (b, g, r)


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


# ----------------------------------------------------------------------------
# Tk shell (imported lazily so the core stays Tk-free / testable)
# ----------------------------------------------------------------------------


def _build_app(video_path: Path, cfg: AnyTrackConfig, arena: Optional[str]):
    import threading
    import tkinter as tk
    import ttkbootstrap as tb
    from tkinter import colorchooser
    from PIL import Image, ImageTk

    class ArenaDebugApp(tb.Window):
        def __init__(self):
            super().__init__(themename="darkly")
            self.title(f"anytrack-arena-debug · {video_path.name}")
            self.geometry("1100x850")
            self.cfg = cfg
            self.cap = cv2.VideoCapture(str(video_path))
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_idx = 0
            self._frame_cache: Dict[int, np.ndarray] = {}
            self.rois: List[CircleROI] = []
            self.roi: Optional[CircleROI] = None
            self.bg: Optional[np.ndarray] = None
            self.dbg: List[dict] = []
            self.ready = False
            self._photo = None

            self.stage_var = tk.StringVar(value="subtracted")
            self.arena_var = tk.StringVar(value=arena or "")
            self.trail_var = tk.IntVar(value=120)
            self.on_vars = {k: tk.BooleanVar(value=DEFAULT_ON[k]) for k, _, _ in OVERLAYS}
            self.colors = dict(DEFAULT_COLORS)
            self.swatches: Dict[str, tk.Widget] = {}

            self._build_ui()
            self.after(50, lambda: threading.Thread(target=self._prepare, daemon=True).start())

        # ---- UI ----
        def _build_ui(self):
            top = tb.Frame(self, padding=8)
            top.pack(side="top", fill="x")
            tb.Label(top, text="arena").pack(side="left")
            self.arena_cb = tb.Combobox(top, textvariable=self.arena_var, width=12, state="readonly")
            self.arena_cb.pack(side="left", padx=(4, 14))
            self.arena_cb.bind("<<ComboboxSelected>>",
                               lambda e: threading.Thread(target=self._prepare, daemon=True).start())
            tb.Label(top, text="stage").pack(side="left")
            cb = tb.Combobox(top, textvariable=self.stage_var, values=STAGES, width=12, state="readonly")
            cb.pack(side="left", padx=(4, 14))
            cb.bind("<<ComboboxSelected>>", lambda e: self._render())
            self.status = tb.Label(top, text="preparing…")
            self.status.pack(side="left", padx=8)

            # Pack the bottom nav BEFORE the expanding body: an expand=True sibling
            # packed first eats the whole cavity and squeezes a later side="bottom"
            # strip to zero height (that was the vanishing frame buttons).
            nav = tb.Frame(self, padding=8)
            nav.pack(side="bottom", fill="x")
            tb.Button(nav, text="◀", width=3, command=lambda: self._step(-1)).pack(side="left")
            tb.Button(nav, text="▶", width=3, command=lambda: self._step(1)).pack(side="left", padx=(2, 8))
            # takefocus=False so the slider never grabs the arrow keys away from the
            # window-level ←/→ frame-step bindings.
            self.scale = tb.Scale(nav, from_=0, to=max(0, self.n - 1), orient="horizontal",
                                  takefocus=False, command=self._on_scrub)
            self.scale.pack(side="left", fill="x", expand=True)
            self.frame_lbl = tb.Label(nav, text="0", width=14)
            self.frame_lbl.pack(side="left", padx=8)

            body = tb.Frame(self, padding=(8, 0))
            body.pack(side="top", fill="both", expand=True)
            self.canvas = tk.Label(body, background="#111")
            self.canvas.pack(side="left", fill="both", expand=True)

            panel = tb.Frame(body, padding=8)
            panel.pack(side="right", fill="y")
            tb.Label(panel, text="Overlays", font="-weight bold").pack(anchor="w")
            for key, label, _ in OVERLAYS:
                row = tb.Frame(panel); row.pack(anchor="w", fill="x", pady=2)
                tb.Checkbutton(row, text=label, variable=self.on_vars[key],
                               command=self._render).pack(side="left")
                sw = tk.Button(row, width=3, background=_bgr_to_hex(self.colors[key]),
                               relief="ridge", command=lambda k=key: self._pick_color(k))
                sw.pack(side="right")
                self.swatches[key] = sw
            tb.Separator(panel).pack(fill="x", pady=8)
            tb.Label(panel, text="trail length (frames, -1 = full)").pack(anchor="w")
            tb.Entry(panel, textvariable=self.trail_var, width=8).pack(anchor="w")
            tb.Button(panel, text="apply trail", command=self._render).pack(anchor="w", pady=(2, 0))

            # ←/→ step one frame back / forward, bound on the window so they work
            # whatever has focus (the readonly comboboxes + focus-less slider don't
            # swallow them). Keeps working even if the nav buttons are off-screen.
            self.bind("<Left>", lambda e: self._step(-1))
            self.bind("<Right>", lambda e: self._step(1))
            self.focus_set()

        def _pick_color(self, key):
            init = _bgr_to_hex(self.colors[key])
            rgb, hx = colorchooser.askcolor(color=init, title=f"colour · {key}")
            if hx:
                self.colors[key] = _hex_to_bgr(hx)
                self.swatches[key].configure(background=hx)
                self._render()

        # ---- data prep (worker thread) ----
        def _prepare(self):
            self.ready = False
            self._set_status("detecting ROIs…")
            bgimg = build_background_image(str(video_path), self.cfg)
            self.rois = sorted(detect_circular_rois(
                bgimg, dp=self.cfg.roi_hough_dp, min_dist_ratio=self.cfg.roi_hough_min_dist_ratio,
                param1=self.cfg.roi_hough_param1, param2=self.cfg.roi_hough_param2,
                min_radius_ratio=self.cfg.min_radius_ratio,
                max_radius_ratio=self.cfg.max_radius_ratio), key=lambda r: r.name)
            names = [r.name for r in self.rois]
            self.arena_cb.configure(values=names)
            want = self.arena_var.get() or (arena if arena in names else (names[0] if names else ""))
            if want not in names:
                self._set_status(f"arena {want!r} not found; detected: {', '.join(names)}")
                return
            self.arena_var.set(want)
            self.roi = next(r for r in self.rois if r.name == want)
            self._set_status(f"modelling background for {want}…")
            self.bg = build_roi_background(str(video_path), self.roi, self.cfg, method="adaptive")
            self._set_status(f"tracking {want}…")
            self.dbg = track_arena_debug(str(video_path), self.roi, self.bg, self.cfg,
                                         progress=lambda i, t: self._set_status(f"tracking {want}… {i}/{t}"))
            self.ready = True
            self._set_status(f"{want}: {len(self.dbg)} frames · ready")
            self.after(0, self._render)

        def _set_status(self, msg):
            self.after(0, lambda: self.status.configure(text=msg))

        # ---- navigation ----
        def _on_scrub(self, v):
            self.frame_idx = int(float(v))
            self._render()

        def _step(self, d):
            self.frame_idx = max(0, min(self.n - 1, self.frame_idx + d))
            self.scale.set(self.frame_idx)
            self._render()

        def _read_gray(self, idx):
            if idx in self._frame_cache:
                return self._frame_cache[idx]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = self.cap.read()
            if not ok:
                return None
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if len(self._frame_cache) > 64:
                self._frame_cache.clear()
            self._frame_cache[idx] = g
            return g

        def _render(self):
            if self.roi is None or self.bg is None:
                return
            gray = self._read_gray(self.frame_idx)
            if gray is None:
                return
            roi_gray = crop_roi_gray(gray, self.roi, self.cfg.roi_downscale)
            dbg_frame = self.dbg[self.frame_idx] if (self.ready and self.frame_idx < len(self.dbg)) else None
            try:
                trail_len = int(self.trail_var.get())
            except Exception:
                trail_len = 120
            trail = trail_up_to(self.dbg, self.frame_idx, trail_len) if self.ready else []
            on = {k: bool(v.get()) for k, v in self.on_vars.items()}
            vis = render_arena_frame(roi_gray, self.bg, self.cfg, dbg_frame, trail,
                                     stage=self.stage_var.get(), overlays_on=on, colors=self.colors)
            self.frame_lbl.configure(text=f"{self.frame_idx}/{self.n - 1}")
            self._show(vis)

        def _show(self, bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            cw = max(200, self.canvas.winfo_width()); ch = max(200, self.canvas.winfo_height())
            s = min(cw / img.width, ch / img.height)
            if s > 0 and s != 1.0:
                img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                                 Image.NEAREST if s > 1 else Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.configure(image=self._photo)

    return ArenaDebugApp()


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="anytrack-arena-debug",
        description="Quick single-arena tracking inspector: stage dropdown + toggleable, "
                    "colour-customizable overlays (candidates, centroid, Kalman prediction/gate, trail).")
    ap.add_argument("--video", required=True, type=Path, help="Source video.")
    ap.add_argument("--roi", default=None, metavar="NAME",
                    help="Arena to inspect (e.g. arena_04). Omit to pick from a dropdown.")
    args = ap.parse_args(argv)
    cfg = load_config()
    app = _build_app(args.video, cfg, args.roi)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
