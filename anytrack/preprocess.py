"""
FFmpeg-based video preprocessing for fast tracking mode.

Extracts cropped, optionally downscaled ROI sub-videos for parallel processing.
"""
from __future__ import annotations

import subprocess
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import tempfile

from .models import CircleROI


def _run_ffmpeg_with_progress(
    cmd: List[str],
    total_frames: int,
    progress_hook: Optional[Callable[[str, dict], None]],
    timeout: float = 600.0,
) -> Tuple[int, str]:
    """Run an FFmpeg command, streaming ``frame=`` progress as frame events.

    The command must include ``-progress pipe:1`` so FFmpeg writes ``frame=N``
    lines to stdout as it decodes; each is surfaced as
    ``("frames", {"stage": "preprocess", "n", "total"})``. stderr is drained on a
    side thread (so a full pipe never deadlocks) and returned for error
    reporting. A watchdog kills the process after ``timeout`` seconds, letting
    the caller fall back exactly as the old blocking path did on a nonzero code.

    Returns ``(returncode, stderr_text)``.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_chunks: List[str] = []

    def _drain() -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                stderr_chunks.append(line)
        except Exception:
            pass

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.daemon = True
    watchdog.start()

    total = int(total_frames) if total_frames else 0
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line.startswith("frame=") and progress_hook is not None:
                try:
                    n = int(line.split("=", 1)[1].strip())
                    progress_hook("frames", {"stage": "preprocess", "n": n, "total": total})
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        proc.wait()
        watchdog.cancel()
        drainer.join(timeout=1.0)

    return proc.returncode, "".join(stderr_chunks)


@dataclass
class PreprocessResult:
    """Result of ROI video extraction."""
    roi_name: str
    video_path: Path
    original_roi: CircleROI
    scale_factor: float  # To map coordinates back (1.0 = no scaling)
    crop_offset: tuple[int, int]  # (x0, y0) in original frame


def get_ffmpeg_path() -> str:
    """Get FFmpeg executable path."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg to use fast mode:\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )
    return ffmpeg


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def get_ffprobe_path() -> Optional[str]:
    """Path to ffprobe, or None if not installed."""
    return shutil.which("ffprobe")


def _probe_codec(video_path) -> Optional[str]:
    """Video stream codec name via ffprobe (e.g. 'h264', 'mpeg4'), or None."""
    ffprobe = get_ffprobe_path()
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=np=1:nq=1",
             str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        return (out.stdout or "").strip() or None
    except Exception:
        return None


# Cache the hardware-decode decision per (backend, codec): whether -hwaccel
# actually initializes on this machine for this codec. Probed once (cheap).
_HW_DECODE_CACHE: Dict[tuple, list] = {}


def _hwaccel_probe_ok(video_path, flags) -> bool:
    """Decode one frame with ``flags`` to confirm the hwaccel initializes."""
    try:
        r = subprocess.run(
            [get_ffmpeg_path(), "-v", "error", *flags, "-i", str(video_path),
             "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def hw_decode_flags(video_path, cfg) -> list:
    """FFmpeg input flags for hardware decode of this source, or ``[]``.

    Honors ``cfg.use_hw_decode`` / ``cfg.hw_decode_backend`` (auto|videotoolbox|
    none). The decision is probed once per (backend, codec) by decoding a single
    frame and cached — if the hwaccel can't initialize (e.g. VideoToolbox has no
    mpeg4 decoder), we return ``[]`` and FFmpeg decodes in software as before.
    Passing ``-hwaccel`` is harmless when unsupported (FFmpeg silently uses SW),
    but probing avoids relying on that and lets callers know the true state.
    """
    if not getattr(cfg, "use_hw_decode", True):
        return []
    backend = getattr(cfg, "hw_decode_backend", "auto") or "auto"
    if backend == "none":
        return []
    if backend == "auto":
        backend = "videotoolbox"
    key = (backend, _probe_codec(video_path))
    if key not in _HW_DECODE_CACHE:
        flags = ["-hwaccel", backend]
        _HW_DECODE_CACHE[key] = flags if _hwaccel_probe_ok(video_path, flags) else []
    return list(_HW_DECODE_CACHE[key])


def roi_crop_geometry(roi: CircleROI, downscale: int) -> Tuple[int, int, int, int]:
    """Square crop + downscaled size for one ROI — shared by every extract path.

    Returns ``(crop_size, x0, y0, scaled_size)``. ``crop_size`` is rounded down to
    a multiple of ``2*downscale`` so the downscaled output has EVEN dimensions
    (libx264/yuv420p rejects odd width/height, which would fail the encode).
    ``scaled_size == crop_size`` when ``downscale == 1``. Cropping the full-frame
    background with the same geometry (Phase 5) reproduces a worker's ROI
    background exactly.
    """
    crop_size = int(2 * roi.r)
    crop_size -= crop_size % (2 * downscale)
    x0 = int(max(0, roi.cx - roi.r))
    y0 = int(max(0, roi.cy - roi.r))
    scaled_size = crop_size // downscale if downscale > 1 else crop_size
    return crop_size, x0, y0, scaled_size


def _arena_mask(sc: int):
    """Boolean disk mask over the arena (excludes the black corners / vignette)."""
    import numpy as np
    yy, xx = np.mgrid[0:sc, 0:sc]
    return (xx - sc / 2) ** 2 + (yy - sc / 2) ** 2 <= (0.92 * sc / 2) ** 2


def deghost(bg, stack, percentile: float = 98, margin: int = 15,
            fill: str = "neighbor", fill_sigma=None):
    """Lift a baked-in dark blob (a fly the background modelling failed to exclude
    at a very-long-dwell spot) to the true arena brightness — fixture-safe.

    Detection (fixture guard) is unchanged regardless of ``fill``: within the arena
    disk, a pixel is flagged only where the per-pixel high ``percentile`` of the
    sample ``stack`` (a robust near-max) exceeds ``bg`` by more than ``margin`` gray
    levels. A dwelling fly's pixel is bright whenever the fly briefly leaves, so it
    trips the gate; a genuinely static dark fixture (e.g. a central port) is dark at
    the near-max too and is never flagged, as are clean pixels.

    ``fill`` chooses the replacement value for flagged pixels:

    - ``"neighbor"`` (default): a Gaussian-weighted mixture of *nearby confidently-
      floor* pixels (inside the arena, not flagged, and not a dark fixture) — an
      unbiased estimate of the true floor. The per-pixel near-max is estimated from
      only the few fly-absent frames and so overshoots (see the brighter-patch
      failure mode); borrowing from well-sampled neighbors instead fills the hole to
      the surrounding floor level, avoiding the phantom dark candidate a too-bright
      patch would create. ``fill_sigma`` sets the borrow radius (default ≈ the hole
      radius). Pixels with no reachable floor neighbor fall back to the near-max.
    - ``"nearmax"``: the original behavior — replace with the per-pixel high
      percentile itself (kept for A/B and as the fallback).

    Returns a uint8 background.
    """
    import numpy as np
    import cv2
    sc = bg.shape[0]
    arena = _arena_mask(sc)
    p_hi = np.percentile(stack, percentile, axis=0)
    lift = (p_hi.astype(np.int16) - bg.astype(np.int16) > margin) & arena
    out = bg.copy()
    if not lift.any():
        return out
    if fill == "nearmax":
        out[lift] = p_hi[lift].astype(np.uint8)
        return out

    # neighbor mixture — source only from confidently-floor pixels (drop the ghost
    # itself and any dark fixture, so their darkness can't bleed into the fill).
    floorish = arena & ~lift
    src = floorish
    if floorish.any():
        med = float(np.median(bg[floorish]))
        cand = floorish & (bg.astype(np.int16) >= med - margin)
        if cand.any():
            src = cand
    if not src.any():                                   # nothing to borrow from
        out[lift] = p_hi[lift].astype(np.uint8)
        return out
    if fill_sigma is None:                              # reach ≈ across the hole
        r_hole = float(np.sqrt(float(lift.sum()) / np.pi))
        fill_sigma = float(np.clip(1.5 * r_hole, 3.0, sc / 4.0))
    known = src.astype(np.float32)
    num = cv2.GaussianBlur(bg.astype(np.float32) * known, (0, 0), fill_sigma)
    den = cv2.GaussianBlur(known, (0, 0), fill_sigma)
    filled = num / np.maximum(den, 1e-6)
    take = lift & (den > 1e-3)
    out[take] = np.clip(np.rint(filled[take]), 0, 255).astype(np.uint8)
    residual = lift & ~take                             # unreachable → near-max
    if residual.any():
        out[residual] = p_hi[residual].astype(np.uint8)
    return out


def _deghost_background(bg, frames, cfg):
    """Config-driven :func:`deghost` for the production per-ROI background."""
    return deghost(bg, frames, getattr(cfg, "bg_deghost_percentile", 98),
                   getattr(cfg, "bg_deghost_margin", 15),
                   fill=getattr(cfg, "bg_deghost_fill", "neighbor"))


def _arena_floor(bg, arena):
    """Robust arena floor brightness: median of the brighter half of arena pixels
    (excludes dark fixtures / residual ghosts from the estimate)."""
    import numpy as np
    vals = bg[arena]
    if vals.size == 0:
        return float(np.median(bg))
    med = float(np.median(vals))
    bright = vals[vals >= med]
    return float(np.median(bright)) if bright.size else med


def _ghost_mask(bg, p_hi, arena, margin):
    """Pixels inside the arena where the near-max is >``margin`` brighter than the
    background — a fly baked in where it dwelt (recoverable: it leaves sometimes)."""
    return ((p_hi.astype("int16") - bg.astype("int16")) > margin) & arena


def _iterative_fill(bg, hole, iters=400, k=3):
    """Slowly diffuse surrounding non-hole pixels into ``hole`` (Jacobi relaxation on a
    box blur → the harmonic infill of the boundary floor). Returns a uint8 copy."""
    import numpy as np
    import cv2
    if not hole.any():
        return bg.copy()
    f = bg.astype(np.float32)
    ring = cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~hole
    f[hole] = float(f[ring].mean()) if ring.any() else float(f[~hole].mean())   # warm start
    for _ in range(iters):
        f[hole] = cv2.blur(f, (k, k))[hole]
    out = bg.copy()
    out[hole] = np.clip(np.rint(f[hole]), 0, 255).astype(np.uint8)
    return out


def fill_stationary(bg, cfg, arena_mask=None, exclude_center_frac=0.0):
    """Recover a never-moved fly baked into ``bg`` by filling it from surrounding floor.

    A perfectly-stationary fly is dark in every frame, so no temporal statistic can
    recover the floor at its pixel (p50=p90=p98=max there). But the floor IS observed
    in the *surrounding* pixels, so a spatial fill can: detect every compact local dark
    spot (darker than a smooth local-floor estimate by ``bg_deghost_margin``, fly-sized),
    then slowly diffuse the surrounding non-fly floor into it (:func:`_iterative_fill`).

    This turns an un-trackable inactive fly (baked in → subtraction ≈ 0 → coverage
    collapses) into a correctly-tracked one. Caveat: it also fills a genuine static
    fixture of the same size/darkness (indistinguishable in one clip); the tracker
    tolerates a static extra candidate, but ``exclude_center_frac`` can spare a central
    rig port. Returns ``(bg_filled_uint8, n_filled_px)``.
    """
    import numpy as np
    import cv2
    sc = bg.shape[0]
    arena = _arena_mask(sc) if arena_mask is None else arena_mask
    lo = int(getattr(cfg, "bg_model_fill_min_area", 15))
    hi = int(getattr(cfg, "bg_model_fill_max_area", 4000))
    margin = int(getattr(cfg, "bg_deghost_margin", 15))
    localfloor = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), sc / 8.0)
    darkspot = ((localfloor - bg.astype(np.float32)) > margin) & arena
    if exclude_center_frac > 0:
        yy, xx = np.mgrid[0:sc, 0:sc]
        darkspot &= ((xx - sc / 2) ** 2 + (yy - sc / 2) ** 2) > (exclude_center_frac * sc / 2) ** 2
    ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(darkspot.astype(np.uint8), 8)
    fly = np.zeros_like(darkspot)
    for c in range(1, ncc):
        if lo <= stats[c, cv2.CC_STAT_AREA] <= hi:
            fly |= (lbl == c)
    if not fly.any():
        return bg.copy(), 0
    return _iterative_fill(bg, fly), int(fly.sum())


def refine_stuck_background(bg, cfg, port_xy=None, port_r=0.0, fly_mask=None):
    """Repair a never-moved fly baked into ``bg`` by filling it from surrounding floor.

    This is the second-pass repair for an arena flagged as stuck (collapsed tracking
    coverage). Both blockers from the single-pass attempt are resolved here:

    - **Spare the port.** ``port_xy``/``port_r`` (the detected odor port, ROI-local
      scaled px) is excluded from the fill, so the port stays in the background and
      does not become a false candidate — the regression that sank the naive fill.
    - **Get a fly mask.** ``fly_mask`` (bool, e.g. from a Segment-Anything
      ``segment_fly``) is used if given; otherwise the fly is taken heuristically as
      the darkest arena spot away from the port (valid *because* coverage already told
      us a fly is stuck here — this gate is what makes the heuristic safe).

    The masked fly region is then diffused in from the surrounding floor
    (:func:`_iterative_fill`), so the fly stops being baked in and becomes detectable
    again. Validated on 2025-12-11: coverage 6–17% → ~100%, detections 95–98% at the
    fly (not the port). Returns ``(bg_refined_uint8, fly_xy | None, n_filled_px)``.
    """
    import numpy as np
    import cv2
    sc = bg.shape[0]
    yy, xx = np.mgrid[0:sc, 0:sc]
    # A wide arena disk (not the 0.92R mask) so a fly pinned at the wall (r≈0.9R) isn't
    # clipped away — that boundary clip previously left nothing to fill.
    arena = (xx - sc / 2) ** 2 + (yy - sc / 2) ** 2 <= (0.97 * sc / 2) ** 2
    margin = int(getattr(cfg, "bg_deghost_margin", 15))
    fly_xy = None
    if fly_mask is None:
        wide = arena
        guard = wide.copy()
        if port_xy is not None:
            guard &= (xx - port_xy[0]) ** 2 + (yy - port_xy[1]) ** 2 > (max(port_r, 0.0) + 0.06 * sc) ** 2
        cand = bg.astype(np.int16).copy()
        cand[~guard] = 999
        fy, fx = np.unravel_index(int(cand.argmin()), cand.shape)
        floor = _arena_floor(bg, arena)
        if floor - float(bg[fy, fx]) <= margin:                     # nothing dark enough → no fly
            return bg.copy(), None, 0
        # fly = the dark connected component holding that darkest pixel (its true shape).
        dark = ((floor - bg.astype(np.float32)) > margin) & guard
        ncc, lbl = cv2.connectedComponents(dark.astype(np.uint8), 8)
        fly_mask = lbl == lbl[fy, fx]
        max_area = int(getattr(cfg, "bg_refine_fly_max_area", 4000))
        if int(fly_mask.sum()) > max_area:            # merged into the dim rim → bounded disk
            fr = int(getattr(cfg, "bg_refine_fly_radius", 13))
            fly_mask = (xx - fx) ** 2 + (yy - fy) ** 2 <= fr * fr
        fly_xy = (float(fx), float(fy))
    else:
        ys2, xs2 = np.nonzero(fly_mask)
        if len(xs2):
            fly_xy = (float(xs2.mean()), float(ys2.mean()))
    fly_mask = fly_mask & arena
    if port_xy is not None:                                          # never fill the port
        fly_mask &= (xx - port_xy[0]) ** 2 + (yy - port_xy[1]) ** 2 > (max(port_r, 0.0) + 3) ** 2
    if not fly_mask.any():
        return bg.copy(), None, 0
    return _iterative_fill(bg, fly_mask), fly_xy, int(fly_mask.sum())


def model_background(stack, cfg, tracks=None, arena_mask=None, sample_idxs=None):
    """Adaptive per-ROI background — cheap when the arena is clean, escalating to
    ghost removal only where a fly dwelt long enough to bake in.

    ``stack`` is an ``(N, sc, sc)`` uint8 array of uniform ROI samples (already
    cropped + scaled the way tracking frames are). The tiers:

    - **T0 (fast)** — a per-pixel high percentile (``bg_model_percentile``, default
      p90). Clean whenever no fly dwells past that percentile; that's the common
      case, so easy arenas pay only one ``np.percentile``.
    - **T1 (de-ghost)** — if any arena pixel is baked in (near-max ``margin`` brighter
      than the baseline), run the track-free neighbor-fill :func:`deghost`, which
      recovers the true floor at a dwell spot from surrounding clean pixels.
    - **T2 (fg-excluded)** — only if T1 leaves residual ghosts *and* a provisional
      ``tracks`` frame is supplied: recompute the percentile with the tracked fly
      masked out of each sample, then de-ghost again. Most sophisticated / most work.

    - **T3 (stationary fill, opt-in, default OFF)** — a never-moved fly is dark in
      ~every frame, so no temporal statistic recovers the floor at its pixel
      (p50=p90=p98=max there). :func:`fill_stationary` tries to fix this spatially, by
      diffusing surrounding floor into compact local dark spots. On 2025-12-11 it does
      **not** help: the only compact local dark spot is the central rig port (which it
      then fills → a false candidate every frame), while the truly-stuck wall fly has
      no local contrast (it sits in the already-dim arena periphery) and is not
      detected. So it is gated OFF by ``bg_model_fill_stationary``; the never-moved
      case is instead caught by its tracking outcome (collapsed coverage / QC flags).
      See background_modelling.html for the full negative result.

    Returns ``(bg_uint8, info)`` where ``info`` records the tier used, method, initial
    and final removable-ghost pixel counts, stationary-filled pixels, and build time.
    """
    import numpy as np
    import time

    sc = stack.shape[1]
    arena = _arena_mask(sc) if arena_mask is None else arena_mask
    margin = int(getattr(cfg, "bg_deghost_margin", 15))
    p_lo = float(getattr(cfg, "bg_model_percentile", 90))
    p_hi_pct = float(getattr(cfg, "bg_deghost_percentile", 98))
    t0 = time.perf_counter()

    p_hi = np.percentile(stack, p_hi_pct, axis=0)
    bg = np.percentile(stack, p_lo, axis=0).astype(np.uint8)          # T0 baseline
    ghost = _ghost_mask(bg, p_hi, arena, margin)
    n_ghost0 = int(ghost.sum())
    tier, method = 0, f"p{int(p_lo)}"

    if n_ghost0 > 0:                                                   # T1
        bg = deghost(bg, stack, p_hi_pct, margin, fill="neighbor")
        tier, method = 1, f"p{int(p_lo)}+deghost"
        ghost = _ghost_mask(bg, p_hi, arena, margin)
        if (int(ghost.sum()) > 0 and tracks is not None and len(tracks)
                and sample_idxs is not None):                             # T2
            fge = _fg_excluded(stack, cfg, tracks, p_lo, sample_idxs)
            bg = deghost(fge, stack, p_hi_pct, margin, fill="neighbor")
            tier, method = 2, "fg_excluded+deghost"
            ghost = _ghost_mask(bg, p_hi, arena, margin)

    n_filled = 0
    if getattr(cfg, "bg_model_fill_stationary", True):                # T3
        bg, n_filled = fill_stationary(
            bg, cfg, arena_mask=arena,
            exclude_center_frac=float(getattr(cfg, "bg_model_fill_exclude_center", 0.0)))
        if n_filled > 0:
            tier = max(tier, 3)
            method = method + "+fill" if "+" in method else f"p{int(p_lo)}+fill"

    info = dict(tier=tier, method=method, n_ghost_initial=n_ghost0,
                n_ghost_final=int(ghost.sum()), n_stationary_filled=n_filled,
                build_s=round(time.perf_counter() - t0, 4))
    return np.ascontiguousarray(bg), info


def _fg_excluded(stack, cfg, tracks, percentile, sample_idxs):
    """Per-pixel percentile with the tracked fly masked out of each sample (T2).

    ``tracks`` is a DataFrame with ``frame``/``xs``/``ys`` in scaled-ROI pixels
    (``xs``/``ys`` are reserved pandas method names, so index with brackets).
    ``sample_idxs[k]`` is the source video frame of ``stack[k]``, so the right
    track position is masked out of each sample. Pixels the fly always covers fall
    back to the plain percentile."""
    import numpy as np
    sc = stack.shape[1]
    scale = float(getattr(cfg, "roi_downscale", 1))
    radius = int(max(6, getattr(cfg, "bg_protect_radius_px", 24) / scale))
    tf = tracks.set_index("frame")
    fstack = stack.astype(np.float32)
    yy, xx = np.mgrid[0:sc, 0:sc]
    for k, fidx in enumerate(sample_idxs):
        if int(fidx) in tf.index:
            r = tf.loc[int(fidx)]
            m = (xx - float(r["xs"])) ** 2 + (yy - float(r["ys"])) ** 2 <= radius * radius
            fstack[k][m] = np.nan
    bg = np.nanpercentile(fstack, percentile, axis=0)
    nan = np.isnan(bg)
    if nan.any():
        bg[nan] = np.percentile(stack, percentile, axis=0)[nan]
    return bg.astype(np.uint8)


def build_roi_backgrounds_uniform(video_path, rois, cfg, fly_masks=None, flags=None):
    """Per-ROI GMM backgrounds from uniformly-sampled ROI frames, decoded the SAME
    way the tracker frames are.

    Reproduces the file-path per-ROI GMM *within* the streaming architecture, which
    otherwise fits its GMM on the first-100 consecutive frames — baking in a fly that
    sits still at the start so the tracker loses it. One FFmpeg pass decodes the
    source, keeps every ``K``-th frame (``K`` chosen for ~``gmm_n_samples`` samples
    spread across the whole video → a roaming fly is excluded from the per-pixel GMM
    mode), and runs each ROI through the *same* ``split → crop → scale → gray`` filter
    used for tracking (:func:`build_stream_command`). Sampling through FFmpeg — not
    cv2 — is essential: cv2 ``INTER_AREA`` and FFmpeg's scaler disagree by tens of
    gray levels at the sharp arena rim, and that mismatch pins the tracker to edge
    artifacts for flies near the boundary. Falls back to ``None`` (worker builds its
    own GMM) if anything goes wrong. Returns ``{roi_name: uint8 (scaled, scaled)}``.

    If a dict is passed as ``fly_masks``, it is populated ``{roi_name: bool
    (scaled, scaled)}`` with the GMM two-component ("fly") mask per ROI — pixels
    where the model fit a distinct dark mode, i.e. the footprint the fly visited
    enough to perturb the background. Captured *before* de-ghost, so it shows what
    the raw GMM flagged as fly (a diagnostic for the caller to overlay).

    If a dict is passed as ``flags`` and ``cfg.bg_model == "adaptive"``, it is
    populated ``{roi_name: info}`` with the :func:`model_background` diagnostics per
    ROI (tier used, ghost/stuck pixel counts, and the ``stationary_suspect`` flag).
    """
    import numpy as np
    import cv2
    from .background import fit_gmm_background

    if not rois:
        return None
    downscale = int(cfg.roi_downscale)
    n_samples = int(cfg.gmm_n_samples)
    try:
        cap = cv2.VideoCapture(str(video_path))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        every = max(1, n_total // n_samples) if n_total > 0 else 1

        tmpdir = Path(tempfile.mkdtemp(prefix="anytrack_roibg_"))
        try:
            ffmpeg = get_ffmpeg_path()
            filter_parts, output_args, geom = [], [], {}
            for i, roi in enumerate(rois):
                crop_size, x0, y0, scaled = roi_crop_geometry(roi, downscale)
                fstr = f"[v{i}]crop={crop_size}:{crop_size}:{x0}:{y0}"
                if downscale > 1:
                    fstr += f",scale={scaled}:{scaled}"
                fstr += f",format=gray[out{i}]"
                filter_parts.append(fstr)
                outfile = tmpdir / f"{roi.name}.gray"
                output_args.extend(["-map", f"[out{i}]", "-f", "rawvideo",
                                    "-pix_fmt", "gray", str(outfile)])
                geom[roi.name] = (scaled, outfile)
            head = f"[0:v]select='not(mod(n\\,{every}))',split={len(rois)}"
            split = head + "".join(f"[v{i}]" for i in range(len(rois)))
            cmd = [ffmpeg, "-y", "-nostats", "-i", str(video_path),
                   "-filter_complex", ";".join([split, *filter_parts]),
                   "-vsync", "0", *output_args]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                return None

            out = {}
            for name, (scaled, outfile) in geom.items():
                raw = outfile.read_bytes()
                fbytes = scaled * scaled
                nfr = len(raw) // fbytes
                if nfr == 0:
                    return None
                frames = np.frombuffer(raw[:nfr * fbytes], np.uint8).reshape(nfr, scaled, scaled)
                gmm, thr = fit_gmm_background(
                    frames, bic_improvement=cfg.gmm_bic_improvement, lowp=cfg.gmm_lowp,
                    min_std=cfg.gmm_min_std, reg_covar=cfg.gmm_reg_covar,
                )
                if fly_masks is not None:   # 2-component pixels (thr not NaN) = fly footprint
                    fly_masks[name] = np.ascontiguousarray(~np.isnan(thr))
                if getattr(cfg, "bg_model", "adaptive") == "adaptive":
                    bg, binfo = model_background(frames, cfg)
                    if flags is not None:
                        flags[name] = binfo
                    out[name] = np.ascontiguousarray(bg)
                else:                        # legacy: brighter-component GMM + de-ghost
                    if getattr(cfg, "bg_deghost", False):
                        gmm = _deghost_background(gmm, frames, cfg)
                    out[name] = np.ascontiguousarray(gmm)
            return out
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        return None


def crop_roi_backgrounds(bg_full, rois, downscale):
    """Per-ROI backgrounds cropped from a full-frame background (Phase 5 reuse).

    Uses the same :func:`roi_crop_geometry` as extraction so each crop lines up
    with the tracker's frames, letting workers skip rebuilding a GMM and — for the
    streaming path — giving a uniformly-sampled background instead of the
    first-N-frames one. Returns ``{roi_name: uint8 (scaled_size, scaled_size)}``.
    """
    import numpy as np
    import cv2

    out = {}
    H, W = bg_full.shape[:2]
    for roi in rois:
        crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)
        crop = bg_full[y0:min(y0 + crop_size, H), x0:min(x0 + crop_size, W)]
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:  # ROI off-frame (rare)
            padded = np.zeros((crop_size, crop_size), dtype=bg_full.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
        if scaled_size != crop_size:
            crop = cv2.resize(crop, (scaled_size, scaled_size), interpolation=cv2.INTER_AREA)
        out[roi.name] = np.ascontiguousarray(crop)
    return out


def build_stream_command(input_video, rois, downscale, fifo_paths, hwaccel_flags=None, stride=1):
    """Build the decode-once FFmpeg command for the streaming path.

    ONE decode of the source, split into N streams, each cropped+scaled+``gray``
    and written as raw gray video to its FIFO (``fifo_paths[roi.name]``). With
    ``stride > 1`` a ``select`` filter emits only every Nth frame (fewer frames
    decoded + tracked; skipped frames are interpolated downstream). Returns
    ``(cmd, results, sizes)`` where ``results`` maps roi_name → PreprocessResult
    (video_path = the FIFO) and ``sizes`` maps roi_name → scaled_size (the raw
    gray frame is ``scaled_size × scaled_size`` bytes). ``hwaccel_flags`` (e.g.
    ``["-hwaccel", "videotoolbox"]``) are inserted before ``-i``.
    """
    ffmpeg = get_ffmpeg_path()
    stride = max(1, int(stride))
    filter_parts: List[str] = []
    output_args: List[str] = []
    results: Dict[str, PreprocessResult] = {}
    sizes: Dict[str, int] = {}

    for i, roi in enumerate(rois):
        crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)
        fstr = f"[v{i}]crop={crop_size}:{crop_size}:{x0}:{y0}"
        if downscale > 1:
            fstr += f",scale={scaled_size}:{scaled_size}"
        fstr += f",format=gray[out{i}]"
        filter_parts.append(fstr)
        output_args.extend([
            "-map", f"[out{i}]", "-f", "rawvideo", "-pix_fmt", "gray",
            str(fifo_paths[roi.name]),
        ])
        results[roi.name] = PreprocessResult(
            roi_name=roi.name,
            video_path=Path(fifo_paths[roi.name]),
            original_roi=roi,
            scale_factor=float(downscale),
            crop_offset=(x0, y0),
        )
        sizes[roi.name] = scaled_size

    # Optional temporal subsampling: keep every Nth decoded frame (comma escaped).
    if stride > 1:
        head = f"[0:v]select='not(mod(n\\,{stride}))',split={len(rois)}"
    else:
        head = f"[0:v]split={len(rois)}"
    split = head + "".join(f"[v{i}]" for i in range(len(rois)))
    filter_complex = ";".join([split, *filter_parts])
    cmd = [
        ffmpeg, "-y", "-nostats",
        *(hwaccel_flags or []),
        "-i", str(input_video),
        "-filter_complex", filter_complex,
        "-vsync", "0",   # rawvideo passthrough: emit exactly the (selected) frames
        *output_args,
    ]
    return cmd, results, sizes


def extract_roi_video(
    input_video: Path,
    roi: CircleROI,
    output_path: Path,
    downscale: int = 2,
    use_hw_encode: bool = True,
    crf: int = 18,
) -> PreprocessResult:
    """
    Extract a single ROI sub-video using FFmpeg.

    Args:
        input_video: Path to source video
        roi: CircleROI to extract
        output_path: Path for output video
        downscale: Downscale factor (1, 2, or 4)
        use_hw_encode: Try hardware encoding (VideoToolbox on macOS)
        crf: Constant rate factor (quality, lower = better, 18 is visually lossless)

    Returns:
        PreprocessResult with extraction details
    """
    ffmpeg = get_ffmpeg_path()

    # Square crop around the ROI (even downscaled dims for libx264).
    crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)

    # Build filter chain
    filters = [f"crop={crop_size}:{crop_size}:{x0}:{y0}"]

    if downscale > 1:
        filters.append(f"scale={scaled_size}:{scaled_size}")

    filter_chain = ",".join(filters)

    # Choose encoder
    if use_hw_encode:
        # Try VideoToolbox on macOS
        encoder = "h264_videotoolbox"
        encoder_opts = ["-q:v", "50"]  # Quality 0-100
    else:
        encoder = "libx264"
        encoder_opts = ["-preset", "ultrafast", "-crf", str(crf)]

    # Build command
    cmd = [
        ffmpeg,
        "-y",  # Overwrite output
        "-i", str(input_video),
        "-vf", filter_chain,
        "-c:v", encoder,
        *encoder_opts,
        "-an",  # No audio
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            # If hardware encoder failed, fall back to software
            if use_hw_encode and "videotoolbox" in result.stderr.lower():
                return extract_roi_video(
                    input_video, roi, output_path,
                    downscale=downscale,
                    use_hw_encode=False,
                    crf=crf,
                )
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out during ROI extraction")

    return PreprocessResult(
        roi_name=roi.name,
        video_path=output_path,
        original_roi=roi,
        scale_factor=float(downscale),
        crop_offset=(x0, y0),
    )


def _extract_roi_videos_single_pass(
    input_video: Path,
    rois: List[CircleROI],
    output_dir: Path,
    downscale: int = 2,
    use_hw_encode: bool = True,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    total_frames: int = 0,
) -> Dict[str, PreprocessResult]:
    """
    Extract all ROI videos in a single FFmpeg pass.

    This decodes the input video once and writes all outputs simultaneously,
    which is faster than decoding multiple times.
    """
    ffmpeg = get_ffmpeg_path()

    # Build filter_complex for all ROIs
    filter_parts = []
    output_args = []
    results: Dict[str, PreprocessResult] = {}

    for i, roi in enumerate(rois):
        crop_size, x0, y0, scaled_size = roi_crop_geometry(roi, downscale)

        # Filter for this ROI: [vN]crop=...,scale=...[outN]  (fed by split, below)
        filter_str = f"[v{i}]crop={crop_size}:{crop_size}:{x0}:{y0}"
        if downscale > 1:
            filter_str += f",scale={scaled_size}:{scaled_size}"
        filter_str += f"[out{i}]"
        filter_parts.append(filter_str)

        # Output args for this ROI
        output_path = output_dir / f"roi_{roi.name}.mp4"

        # Single-pass MUST use software encode: 4 simultaneous VideoToolbox
        # sessions serialize on the HW encoder (~1 fps). libx264 ultrafast across
        # cores is ~2x faster than the sequential-HW path on this hardware.
        encoder = "libx264"
        encoder_opts = ["-preset", "ultrafast", "-crf", "18"]

        output_args.extend([
            "-map", f"[out{i}]",
            "-c:v", encoder,
            *encoder_opts,
            "-an",
            str(output_path),
        ])

        results[roi.name] = PreprocessResult(
            roi_name=roi.name,
            video_path=output_path,
            original_roi=roi,
            scale_factor=float(downscale),
            crop_offset=(x0, y0),
        )

    # Decode the source ONCE, split it into N identical streams, crop each —
    # this replaces N separate full-video decode passes with a single decode.
    split = f"[0:v]split={len(rois)}" + "".join(f"[v{i}]" for i in range(len(rois)))
    filter_complex = ";".join([split, *filter_parts])

    # -progress pipe:1 streams `frame=N` to stdout so we can draw a decode bar;
    # -nostats silences the (now redundant) stderr progress stats.
    cmd = [
        ffmpeg,
        "-y",
        "-nostats",
        "-progress", "pipe:1",
        "-i", str(input_video),
        "-filter_complex", filter_complex,
        *output_args,
    ]

    returncode, stderr = _run_ffmpeg_with_progress(cmd, total_frames, progress_hook)
    if returncode != 0:
        # Single-pass graph failed (e.g. HW-encoder contention across 4
        # simultaneous outputs) or was killed by the watchdog. Fall back to the
        # proven sequential extractor, preserving the encoder choice — never
        # worse than before.
        return _extract_roi_videos_sequential(
            input_video, rois, output_dir,
            downscale=downscale,
            use_hw_encode=use_hw_encode,
            progress_hook=progress_hook,
        )

    return results


def _extract_roi_videos_sequential(
    input_video: Path,
    rois: List[CircleROI],
    output_dir: Path,
    downscale: int = 2,
    use_hw_encode: bool = True,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
) -> Dict[str, PreprocessResult]:
    """Sequential extraction fallback."""
    results: Dict[str, PreprocessResult] = {}

    for i, roi in enumerate(rois):
        if progress_hook:
            progress_hook("preprocessing", {
                "roi": roi.name,
                "index": i,
                "total": len(rois),
                "percent": float(i) / len(rois),
            })

        output_path = output_dir / f"roi_{roi.name}.mp4"

        result = extract_roi_video(
            input_video=input_video,
            roi=roi,
            output_path=output_path,
            downscale=downscale,
            use_hw_encode=use_hw_encode,
        )

        results[roi.name] = result

    return results


def extract_roi_videos(
    input_video: Path,
    rois: List[CircleROI],
    output_dir: Optional[Path] = None,
    downscale: int = 2,
    use_hw_encode: bool = True,
    progress_hook: Optional[Callable[[str, dict], None]] = None,
    total_frames: int = 0,
) -> Dict[str, PreprocessResult]:
    """
    Extract all ROI sub-videos using FFmpeg.

    Args:
        input_video: Path to source video
        rois: List of CircleROIs to extract
        output_dir: Directory for output videos (uses temp dir if None)
        downscale: Downscale factor (1, 2, or 4)
        use_hw_encode: Try hardware encoding
        progress_hook: Optional callback for progress updates

    Returns:
        Dictionary mapping roi_name -> PreprocessResult
    """
    if not rois:
        raise ValueError("No ROIs provided for extraction")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="anytrack_roi_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single-pass: decode the source ONCE and crop all ROIs (falls back to the
    # sequential per-ROI extractor if the single FFmpeg graph fails).
    if progress_hook:
        progress_hook("preprocessing",
                      {"roi": "single-pass", "index": 0, "total": 1, "percent": 0.0})
    results = _extract_roi_videos_single_pass(
        input_video=input_video,
        rois=rois,
        output_dir=output_dir,
        downscale=downscale,
        use_hw_encode=use_hw_encode,
        progress_hook=progress_hook,
        total_frames=total_frames,
    )

    if progress_hook:
        progress_hook("preprocessing", {
            "roi": "complete",
            "index": len(rois),
            "total": len(rois),
            "percent": 1.0,
        })

    return results


def cleanup_roi_videos(results: Dict[str, PreprocessResult]) -> None:
    """
    Remove extracted ROI video files.

    Args:
        results: Dictionary of PreprocessResults to clean up
    """
    for result in results.values():
        try:
            if result.video_path.exists():
                result.video_path.unlink()
        except OSError:
            pass  # Ignore cleanup errors

    # Try to remove parent directory if empty
    if results:
        first_result = next(iter(results.values()))
        parent = first_result.video_path.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def estimate_extraction_time(
    video_path: Path,
    n_rois: int,
    downscale: int = 2,
) -> float:
    """
    Estimate extraction time in seconds.

    This is a rough estimate based on typical FFmpeg performance.
    Actual time varies based on hardware and video codec.

    Args:
        video_path: Path to source video
        n_rois: Number of ROIs to extract
        downscale: Downscale factor

    Returns:
        Estimated time in seconds
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    duration_s = frame_count / fps

    # Rough estimate: 2x realtime per ROI with hardware encode
    # Slower with software encode or larger downscale
    base_factor = 0.5 if downscale >= 2 else 1.0

    return duration_s * n_rois * base_factor