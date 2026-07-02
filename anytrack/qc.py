"""
Quality-control artifacts for a finished tracking run (Milestone A9).

Everything here is **post-hoc**: it is derived from the tracking dataframe plus
the source video, so no re-tracking is needed. Produces

  1. an annotated **overlay video** (ROI circles, centroid, fading trail,
     per-frame status color),
  2. **diagnostic plots** (missing-frame fraction, speed/area distributions,
     trajectory paths),
  3. per-frame **failure flags** + a **summary metrics** report.

Missed frames are recovered as gaps in the ``frame`` index (the tracker emits no
row when it finds no object), so ``no_candidate`` needs no tracker changes. The
``multi_candidate`` flag and a per-frame ``contrast`` distribution light up once
the Tier-2 instrumentation columns (``n_candidates``, ``contrast``) are present;
until then they degrade gracefully. Background-drift-over-time waits on A4.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

from .writer import write_diagnostics


# Per-frame boolean failure-flag columns produced by :func:`compute_flags`.
FLAG_COLUMNS = ["flag_area", "flag_jump", "flag_crop_oob", "flag_multi"]

# Visually distinct BGR colors cycled per ROI (shared by overlay + plots).
_ROI_PALETTE_BGR = [
    (244, 133, 66), (83, 168, 52), (7, 193, 255), (53, 67, 234),
    (188, 71, 171), (193, 172, 0), (67, 112, 255), (36, 157, 158),
]


def roi_color_map(names) -> Dict[str, tuple]:
    """Map each ROI name to a stable BGR color (sorted for determinism)."""
    uniq = sorted({str(n) for n in names})
    return {n: _ROI_PALETTE_BGR[i % len(_ROI_PALETTE_BGR)] for i, n in enumerate(uniq)}


def _bgr_to_mpl(bgr: tuple) -> tuple:
    """Convert a 0-255 BGR tuple to a matplotlib 0-1 RGB tuple."""
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


def _read_tracks(path: Path) -> pd.DataFrame:
    """Read a finished session's tracks table (parquet or CSV)."""
    return pd.read_csv(path) if Path(path).suffix.lower() == ".csv" else pd.read_parquet(path)


def compute_flags(df: pd.DataFrame, video, cfg) -> pd.DataFrame:
    """Return a copy of ``df`` with per-frame boolean failure-flag columns.

    - ``flag_area``: ellipse area outside ``[expected_fly_area_min, max]``.
    - ``flag_jump``: centroid jump from the previous frame > ``max_jump_px``.
    - ``flag_crop_oob``: centroid within ``crop_size/2`` of a frame edge.
    - ``flag_multi``: >1 candidate at this frame (needs ``n_candidates``; else False).
    """
    if df.empty:
        out = df.copy()
        for c in FLAG_COLUMNS:
            out[c] = pd.Series(dtype=bool)
        return out

    out = df.sort_values(["roi", "track_id", "frame"]).reset_index(drop=True).copy()

    amin = float(getattr(cfg, "expected_fly_area_min", 0))
    amax = float(getattr(cfg, "expected_fly_area_max", np.inf))
    if "area" in out.columns:
        out["flag_area"] = (out["area"] < amin) | (out["area"] > amax)
    else:
        out["flag_area"] = False

    dx = out.groupby(["roi", "track_id"])["x"].diff()
    dy = out.groupby(["roi", "track_id"])["y"].diff()
    jump = np.hypot(dx, dy)
    out["flag_jump"] = (jump > float(getattr(cfg, "max_jump_px", np.inf))).fillna(False)

    half = float(getattr(cfg, "crop_size", 0)) / 2.0
    w = int(getattr(video, "width", 0) or 0)
    h = int(getattr(video, "height", 0) or 0)
    if half > 0 and w > 0 and h > 0:
        out["flag_crop_oob"] = (
            (out["x"] - half < 0) | (out["y"] - half < 0)
            | (out["x"] + half > w) | (out["y"] + half > h)
        )
    else:
        out["flag_crop_oob"] = False

    if "n_candidates" in out.columns:
        out["flag_multi"] = out["n_candidates"].fillna(1) > 1
    else:
        out["flag_multi"] = False

    return out


def missing_report(df: pd.DataFrame, video) -> Dict[str, Dict[str, float]]:
    """Per-ROI missing-frame stats from gaps in the ``frame`` index."""
    if df.empty:
        return {}
    n_total = int(getattr(video, "frame_count", 0) or 0)
    if n_total <= 0:
        n_total = int(df["frame"].max()) + 1

    rep: Dict[str, Dict[str, float]] = {}
    for roi, g in df.groupby("roi"):
        present = int(g["frame"].nunique())
        missing = max(0, n_total - present)
        rep[str(roi)] = {
            "frames_present": present,
            "frames_expected": n_total,
            "frames_missing": missing,
            "missing_fraction": (missing / n_total) if n_total else 0.0,
        }
    return rep


def summarize(df: pd.DataFrame, video, cfg) -> Dict[str, Any]:
    """Compact per-run + per-ROI QC metrics (JSON-serializable)."""
    flagged = compute_flags(df, video, cfg)
    miss = missing_report(df, video)

    per_roi: Dict[str, Dict[str, Any]] = {}
    for roi, g in flagged.groupby("roi"):
        stats: Dict[str, Any] = dict(miss.get(str(roi), {}))
        for c in FLAG_COLUMNS:
            stats[c] = int(g[c].sum())
        if "speed_mm_s" in g.columns:
            stats["speed_mm_s_mean"] = float(g["speed_mm_s"].mean())
            stats["speed_mm_s_max"] = float(g["speed_mm_s"].max())
        if "area" in g.columns:
            stats["area_mean"] = float(g["area"].mean())
        if "contrast" in g.columns:
            stats["contrast_mean"] = float(g["contrast"].mean())
        per_roi[str(roi)] = stats

    return {
        "n_rois": int(df["roi"].nunique()) if not df.empty else 0,
        "n_frames": int(getattr(video, "frame_count", 0) or 0),
        "per_roi": per_roi,
    }


def plot_diagnostics(df: pd.DataFrame, video, cfg, out_dir: Path) -> List[Path]:
    """Write a single 2x2 diagnostics figure; returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")  # headless — never pop a window
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if df.empty:
        return []

    miss = missing_report(df, video)
    rois = list(miss.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].bar(rois, [miss[r]["missing_fraction"] for r in rois], color="#c0504d")
    axes[0, 0].set_title("Missing-frame fraction")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis="x", rotation=45)

    if "speed_mm_s" in df.columns:
        for roi, g in df.groupby("roi"):
            axes[0, 1].hist(g["speed_mm_s"].dropna(), bins=50, alpha=0.5, label=str(roi))
        axes[0, 1].set_title("Speed distribution (mm/s)")
        axes[0, 1].legend(fontsize=7)
    else:
        axes[0, 1].set_visible(False)

    # Contrast distribution (Tier-2 column; hidden until instrumented).
    if "contrast" in df.columns and df["contrast"].notna().any():
        for roi, g in df.groupby("roi"):
            axes[0, 2].hist(g["contrast"].dropna(), bins=50, alpha=0.5, label=str(roi))
        axes[0, 2].set_title("Detection contrast (bg diff)")
    else:
        axes[0, 2].set_visible(False)

    if "area" in df.columns:
        for roi, g in df.groupby("roi"):
            axes[1, 0].hist(g["area"].dropna(), bins=50, alpha=0.5, label=str(roi))
        axes[1, 0].axvline(float(getattr(cfg, "expected_fly_area_min", 0)), color="r", ls="--", lw=1)
        axes[1, 0].axvline(float(getattr(cfg, "expected_fly_area_max", 0)), color="r", ls="--", lw=1)
        axes[1, 0].set_title("Ellipse area (px), red = expected range")
    else:
        axes[1, 0].set_visible(False)

    # Candidate-count distribution (Tier-2 column; hidden until instrumented).
    if "n_candidates" in df.columns:
        vc = df["n_candidates"].value_counts().sort_index()
        axes[1, 1].bar([str(int(k)) for k in vc.index], vc.to_numpy(), color="#4f81bd")
        axes[1, 1].set_title("Candidates per detected frame")
        axes[1, 1].set_xlabel("n_candidates")
    else:
        axes[1, 1].set_visible(False)

    for roi, g in df.groupby("roi"):
        gg = g.sort_values("frame")
        axes[1, 2].plot(gg["x"], gg["y"], lw=0.5, label=str(roi))
    axes[1, 2].set_title("Trajectory (full-frame px)")
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_aspect("equal", "datalim")

    fig.tight_layout()
    path = out_dir / "qc_diagnostics.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def plot_timeseries(df: pd.DataFrame, cfg, out_dir: Path) -> List[Path]:
    """Per-ROI timeseries of detection size/count and ellipse properties.

    One stacked panel per available metric (area, candidate count, ellipse
    major/minor/angle), lines colored per ROI (same palette as the overlay).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if df.empty:
        return []

    metrics = [
        ("area", "contour/ellipse\narea (px)"),
        ("n_candidates", "candidates\nper frame"),
        ("major", "ellipse\nmajor (px)"),
        ("minor", "ellipse\nminor (px)"),
        ("angle_deg", "ellipse\nangle (deg)"),
    ]
    present = [(c, lbl) for c, lbl in metrics if c in df.columns]
    if not present:
        return []

    colors = roi_color_map(df["roi"].unique())
    x = "t_s" if "t_s" in df.columns else "frame"
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 2.2 * len(present)),
                             sharex=True, squeeze=False)
    for ax, (col, lbl) in zip(axes[:, 0], present):
        for roi, g in df.groupby("roi"):
            gg = g.sort_values(x)
            ax.plot(gg[x], gg[col], lw=0.6, label=str(roi),
                    color=_bgr_to_mpl(colors[str(roi)]))
        ax.set_ylabel(lbl, fontsize=8)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=min(4, max(1, df["roi"].nunique())), loc="upper right")
    axes[-1, 0].set_xlabel(x)
    fig.suptitle("Per-ROI timeseries")
    fig.tight_layout()
    path = out_dir / "qc_timeseries.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def plot_coverage(df: pd.DataFrame, video, out_dir: Path) -> List[Path]:
    """Per-ROI tracked-vs-missing timeline (raster) — highlights dropouts.

    Rows are ROIs; the horizontal axis is frame index; shading is the fraction
    of frames tracked in each time bin (dark = tracked, white = missing).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if df.empty:
        return []

    n_total = int(getattr(video, "frame_count", 0) or 0) or (int(df["frame"].max()) + 1)
    if n_total <= 0:
        return []
    rois = sorted(df["roi"].astype(str).unique())
    n_bins = int(min(n_total, 2000))
    edges = np.linspace(0, n_total, n_bins + 1)
    binw = np.maximum(np.diff(edges), 1e-9)

    mat = np.zeros((len(rois), n_bins), dtype=float)
    for i, roi in enumerate(rois):
        frames = df.loc[df["roi"].astype(str) == roi, "frame"].astype(int).to_numpy()
        frames = frames[(frames >= 0) & (frames < n_total)]
        counts, _ = np.histogram(frames, bins=edges)
        mat[i] = np.clip(counts / binw, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, 0.5 * len(rois) + 1.6))
    ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0,
              extent=[0, n_total, len(rois) - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois)
    ax.set_xlabel("frame")
    ax.set_title(f"Tracked-frame coverage (white = missing detections; {n_total} frames)")
    fig.tight_layout()
    path = out_dir / "qc_coverage.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def render_overlay(
    video,
    df: pd.DataFrame,
    cfg,
    out_path: Path,
    max_frames: int = 0,
    trail_len: int = 30,
    show_progress: bool = False,
) -> Tuple[Optional[Path], int]:
    """Burn tracking annotations onto the source video.

    Draws ROI circles (if ``video.rois`` is available), a fading trail per
    track, and a centroid dot colored green (clean) or red (any failure flag).
    Returns ``(path, frames_written)``; ``path`` is None if no encoder is
    available. ``max_frames=0`` renders the whole video.
    """
    out_path = Path(out_path)
    flagged = compute_flags(df, video, cfg)

    frame_groups = {int(f): g for f, g in flagged.groupby("frame")}
    trails: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, g in flagged.groupby(["roi", "track_id"]):
        gg = g.sort_values("frame")
        trails[key] = (gg["frame"].to_numpy(), gg["x"].to_numpy(), gg["y"].to_numpy())

    cap = cv2.VideoCapture(str(video.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(getattr(video, "fps_nominal", 0) or 0) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return None, 0

    rois = list(getattr(video, "rois", []) or [])
    # One stable color per ROI, shared with the timeseries plots. Cover both the
    # detected ROIs and any roi names present only in the dataframe.
    names = [r.name for r in rois] + [str(n) for n in flagged.get("roi", pd.Series(dtype=str)).unique()]
    colors = roi_color_map(names)

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            total = max_frames or int(getattr(video, "frame_count", 0) or 0) or None
            pbar = tqdm(total=total, desc="  overlay",
                        bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        leave=True)
        except Exception:
            pbar = None

    fidx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # ROI arenas: filled label + circle in the ROI's own color.
        for roi in rois:
            col = colors.get(str(roi.name), (90, 90, 90))
            c = (int(roi.cx), int(roi.cy))
            cv2.circle(frame, c, int(roi.r), col, 2)
            cv2.putText(frame, str(roi.name), (int(roi.cx - roi.r), int(roi.cy - roi.r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # Fading trails, colored per ROI.
        for (roi_name, _tid), (fr, xs, ys) in trails.items():
            col = colors.get(str(roi_name), (0, 180, 180))
            lo = int(np.searchsorted(fr, fidx - trail_len))
            hi = int(np.searchsorted(fr, fidx, side="right"))
            if hi - lo >= 2:
                pts = np.stack([xs[lo:hi], ys[lo:hi]], axis=1).astype(np.int32)
                cv2.polylines(frame, [pts], False, col, 1)

        # Centroids: ROI-colored dot; a red ring marks a flagged frame.
        rows = frame_groups.get(fidx)
        if rows is not None:
            for _, r in rows.iterrows():
                col = colors.get(str(r["roi"]), (0, 255, 0))
                cx, cy = int(r["x"]), int(r["y"])
                cv2.circle(frame, (cx, cy), 4, col, -1)
                if any(bool(r.get(c, False)) for c in FLAG_COLUMNS):
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), 2)

        cv2.putText(frame, f"frame {fidx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        written += 1
        fidx += 1
        if pbar is not None:
            pbar.update(1)
        if max_frames and written >= max_frames:
            break

    if pbar is not None:
        pbar.close()
    cap.release()
    writer.release()
    return out_path, written


def run_qc(
    video,
    df: pd.DataFrame,
    cfg,
    out_dir: Path,
    overlay: bool = True,
    max_frames: int = 0,
    show_progress: bool = False,
    background=None,
    save_background: bool = True,
) -> Dict[str, Any]:
    """Produce all QC artifacts into ``out_dir`` and return a manifest dict.

    Writes the model background (``qc_background.png``), a distributions figure,
    per-ROI timeseries, a coverage raster, per-frame failure flags, a JSON
    summary and (optionally) an annotated overlay video. A prebuilt
    ``background`` image is reused; otherwise it is built from the video when
    ``save_background`` is set.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Background PNG (the clean-plate reference the tracker subtracts against).
    background_path: Optional[Path] = None
    bg = background
    if bg is None and save_background:
        try:
            from .background import build_background_image
            bg = build_background_image(video.video_path, cfg)
        except Exception:
            bg = None
    if bg is not None:
        background_path = out_dir / "qc_background.png"
        cv2.imwrite(str(background_path), bg)

    flagged = compute_flags(df, video, cfg)
    flags_path = out_dir / "qc_flags.parquet"
    write_diagnostics(flagged, flags_path, fmt="parquet")

    summary = summarize(df, video, cfg)
    summary_path = out_dir / "qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plots = (
        plot_diagnostics(df, video, cfg, out_dir)
        + plot_timeseries(df, cfg, out_dir)
        + plot_coverage(df, video, out_dir)
    )

    overlay_path: Optional[Path] = None
    n_overlay = 0
    if overlay:
        overlay_path, n_overlay = render_overlay(
            video, df, cfg, out_dir / "qc_overlay.mp4",
            max_frames=max_frames, show_progress=show_progress,
        )

    return {
        "summary": summary,
        "summary_path": summary_path,
        "flags_path": flags_path,
        "background_path": background_path,
        "plots": plots,
        "overlay_path": overlay_path,
        "overlay_frames": n_overlay,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: generate QC artifacts from a finished tracking run (A9).

    Loads a tracks table + the source video and writes overlay/plots/flags/
    summary into an output directory. Arena circles in the overlay require ROI
    geometry, which a raw tracks table lacks — run QC from a session to get
    them; the CLI still draws centroids, trails, flags, and all metrics.
    """
    import argparse
    from types import SimpleNamespace
    from .config import load_config

    ap = argparse.ArgumentParser(
        prog="anytrack-qc",
        description="Generate QC artifacts (overlay, plots, flags, summary) from a finished run.",
    )
    ap.add_argument("--video", required=True, type=Path, help="Source video.")
    ap.add_argument("--tracks", required=True, type=Path,
                    help="Tracks table (.parquet or .csv) with roi/track_id/frame/x/y columns.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <cfg.output_dir or .>/qc).")
    ap.add_argument("--no-overlay", action="store_true", help="Skip the overlay video (fast).")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Cap overlay frames (0 = whole video).")
    args = ap.parse_args(argv)

    cfg = load_config()
    if not args.video.exists():
        ap.error(f"video not found: {args.video}")
    if not args.tracks.exists():
        ap.error(f"tracks not found: {args.tracks}")

    df = _read_tracks(args.tracks)

    cap = cv2.VideoCapture(str(args.video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    video = SimpleNamespace(video_path=args.video, width=w, height=h, frame_count=n,
                            fps_nominal=fps, rois=[])

    out_dir = args.out_dir or (Path(getattr(cfg, "output_dir", "") or ".") / "qc")
    res = run_qc(video, df, cfg, out_dir, overlay=not args.no_overlay,
                 max_frames=args.max_frames, show_progress=True)

    print(f"QC written to {out_dir}")
    print(f"  summary: {res['summary_path']}")
    print(f"  flags:   {res['flags_path']}")
    for p in res["plots"]:
        print(f"  plot:    {p}")
    if res["overlay_path"] is not None:
        print(f"  overlay: {res['overlay_path']} ({res['overlay_frames']} frames)")
    elif not args.no_overlay:
        print("  overlay: skipped (no mp4 encoder available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
