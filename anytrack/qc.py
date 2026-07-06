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
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

from .writer import write_diagnostics
from .preprocess import check_ffmpeg_available, get_ffmpeg_path


# Per-frame boolean failure-flag columns produced by :func:`compute_flags`.
FLAG_COLUMNS = ["flag_area", "flag_jump", "flag_crop_oob", "flag_multi", "flag_low_contrast"]

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

    if "contrast" in out.columns:
        thr = float(getattr(cfg, "qc_min_contrast", 0.0))
        out["flag_low_contrast"] = out["contrast"].notna() & (out["contrast"] < thr)
    else:
        out["flag_low_contrast"] = False

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

    roi_r = {r.name: float(r.r) for r in (getattr(video, "rois", None) or [])}
    wall_thr = float(getattr(cfg, "qc_edge_pinned_frac", 0.9))
    lowact_thr = float(getattr(cfg, "qc_low_activity_mm_s", 0.05))

    per_roi: Dict[str, Dict[str, Any]] = {}
    for roi, g in flagged.groupby("roi"):
        stats: Dict[str, Any] = dict(miss.get(str(roi), {}))
        for c in FLAG_COLUMNS:
            stats[c] = int(g[c].sum())
        if "speed_mm_s" in g.columns:
            stats["speed_mm_s_mean"] = float(g["speed_mm_s"].mean())
            stats["speed_mm_s_max"] = float(g["speed_mm_s"].max())
            stats["low_activity"] = bool(g["speed_mm_s"].mean() < lowact_thr)
        if "area" in g.columns:
            stats["area_mean"] = float(g["area"].mean())
        if "contrast" in g.columns:
            stats["contrast_mean"] = float(g["contrast"].mean())
        # Edge-pinned: fly spends >wall_thr of frames in the outer arena (a fly
        # parked at the wall the whole clip → inactive fly or a stuck/edge-pinned
        # track; also un-removable from the background — see background_methods.html).
        rr = roi_r.get(str(roi))
        if rr and "r_center_px" in g.columns:
            wall = float((g["r_center_px"] > 0.85 * rr).mean())
            stats["wall_fraction"] = wall
            stats["edge_pinned"] = bool(wall > wall_thr)
        per_roi[str(roi)] = stats

    return {
        "n_rois": int(df["roi"].nunique()) if not df.empty else 0,
        "n_frames": int(getattr(video, "frame_count", 0) or 0),
        "n_edge_pinned": sum(1 for s in per_roi.values() if s.get("edge_pinned")),
        "n_low_activity": sum(1 for s in per_roi.values() if s.get("low_activity")),
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
        ("contrast", "detection\ncontrast"),
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


def plot_kinematics(df: pd.DataFrame, cfg, out_dir: Path) -> List[Path]:
    """Per-ROI kinematics timeseries (speed, angular speed, distance-from-center)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if df.empty:
        return []

    metrics = [
        ("speed_mm_s", "speed (mm/s)"),
        ("ang_speed_deg_s", "angular speed\n(deg/s)"),
        ("r_center_mm", "dist. from\ncenter (mm)"),
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
    fig.suptitle("Per-ROI kinematics")
    fig.tight_layout()
    path = out_dir / "qc_kinematics.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def plot_drift(df: pd.DataFrame, cfg, out_dir: Path) -> List[Path]:
    """Per-ROI background brightness drift over time.

    Only produced when drift data exists (i.e. bg_drift_correction was on);
    returns [] otherwise, so it doesn't clutter the default (static-bg) QC.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if df.empty or "bg_drift" not in df.columns or not df["bg_drift"].notna().any():
        return []

    colors = roi_color_map(df["roi"].unique())
    x = "t_s" if "t_s" in df.columns else "frame"
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for roi, g in df.groupby("roi"):
        gg = g.sort_values(x)
        ax.plot(gg[x], gg["bg_drift"], lw=0.7, label=str(roi),
                color=_bgr_to_mpl(colors[str(roi)]))
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel("bg brightness drift")
    ax.set_xlabel(x)
    ax.set_title("Background drift over time (bg_drift_correction)")
    ax.legend(fontsize=7, ncol=min(4, max(1, df["roi"].nunique())))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "qc_drift.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def plot_occupancy(df: pd.DataFrame, video, out_dir: Path) -> List[Path]:
    """Per-ROI 2D occupancy heatmap of centroid positions (thigmotaxis etc.)."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return []

    rois = sorted(df["roi"].astype(str).unique())
    geom = {str(r.name): r for r in (getattr(video, "rois", []) or [])}
    ncol = min(len(rois), 3)
    nrow = math.ceil(len(rois) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, roi in enumerate(rois):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        g = df[df["roi"].astype(str) == roi]
        ax.hist2d(g["x"].to_numpy(), g["y"].to_numpy(), bins=50, cmap="magma")
        ax.set_title(f"{roi} occupancy", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        r = geom.get(roi)
        if r is not None:
            ax.add_patch(plt.Circle((r.cx, r.cy), r.r, fill=False, color="cyan", lw=1))
    fig.tight_layout()
    path = out_dir / "qc_occupancy.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def render_flagged_montage(video, df: pd.DataFrame, cfg, out_path: Path) -> Tuple[Optional[Path], int]:
    """Thumbnail grid of the worst flagged frames (so you can see *why* flagged).

    Extracts centroid crops from the source video at flagged frames, tiles them
    into a grid, and annotates each with ROI/frame and the flags that fired.
    """
    from .cropper import extract_crop

    out_path = Path(out_path)
    flagged = compute_flags(df, video, cfg)
    present_flags = [c for c in FLAG_COLUMNS if c in flagged.columns]
    if not present_flags:
        return None, 0
    mask = flagged[present_flags].any(axis=1)
    fl = flagged[mask]
    if fl.empty:
        return None, 0

    max_tiles = int(getattr(cfg, "qc_montage_max", 25))
    tile = int(getattr(cfg, "qc_montage_tile", 96))
    if len(fl) > max_tiles:  # spread the sample across the flagged set
        fl = fl.iloc[np.linspace(0, len(fl) - 1, max_tiles).astype(int)]

    need: Dict[int, List] = {}
    for _, r in fl.iterrows():
        need.setdefault(int(r["frame"]), []).append(r)

    cap = cv2.VideoCapture(str(video.video_path))
    if not cap.isOpened():
        return None, 0
    max_needed = max(need)
    tiles: List[np.ndarray] = []
    fidx = 0
    while True:
        okk, frame = cap.read()
        if not okk:
            break
        if fidx in need:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            for r in need[fidx]:
                crop, _, _, _ = extract_crop(gray, float(r["x"]), float(r["y"]), tile)
                cimg = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                flags = [c.replace("flag_", "") for c in present_flags if bool(r.get(c, False))]
                cv2.putText(cimg, f"{r['roi']} f{int(r['frame'])}", (3, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(cimg, ",".join(flags), (3, tile - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1, cv2.LINE_AA)
                tiles.append(cimg)
        fidx += 1
        if fidx > max_needed:
            break
    cap.release()

    if not tiles:
        return None, 0
    ncol = min(5, len(tiles))
    nrow = (len(tiles) + ncol - 1) // ncol
    grid = np.zeros((nrow * tile, ncol * tile, 3), np.uint8)
    for k, c in enumerate(tiles):
        rr, cc = divmod(k, ncol)
        grid[rr * tile:(rr + 1) * tile, cc * tile:(cc + 1) * tile] = c
    cv2.imwrite(str(out_path), grid)
    return out_path, len(tiles)


def write_html_report(out_dir: Path, summary: Dict[str, Any], artifacts: Dict[str, Optional[Path]]) -> Path:
    """Bundle the QC summary table + all PNGs into a single shareable HTML page."""
    out_dir = Path(out_dir)

    # Summary table (per-ROI stats).
    per_roi = summary.get("per_roi", {})
    cols: List[str] = []
    for stats in per_roi.values():
        for k in stats:
            if k not in cols:
                cols.append(k)
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for roi, stats in per_roi.items():
        cells = "".join(f"<td>{stats.get(c, '')}</td>" for c in cols)
        rows += f"<tr><th>{roi}</th>{cells}</tr>"
    table = f"<table><tr><th>roi</th>{head}</tr>{rows}</table>" if per_roi else "<p>No data.</p>"

    # Optional pose-flag summary line.
    pose = summary.get("pose") or {}
    pose_html = ""
    if pose:
        rate = lambda k: f"{100 * pose.get(k, 0.0):.1f}%"   # noqa: E731
        pose_html = (f"<p><b>Pose</b> ({pose.get('n_instances', 0)} instances): "
                     f"low-confidence {rate('flag_low_conf')}, missing {rate('flag_missing')}, "
                     f"wing-swap {rate('flag_wing_swap')}, head/tail flip {rate('flag_headtail')}.</p>")

    # Embed each artifact image by its filename (relative to this HTML).
    order = ["background", "overlay", "diagnostics", "timeseries", "kinematics",
             "drift", "occupancy", "coverage", "pose_confidence", "montage"]
    imgs = ""
    for name in order:
        p = artifacts.get(name)
        if p is None:
            continue
        p = Path(p)
        if p.suffix.lower() == ".mp4":
            imgs += (f"<figure><figcaption>{name}</figcaption>"
                     f"<video controls width='900' src='{p.name}'></video></figure>")
        else:
            imgs += (f"<figure><figcaption>{name}</figcaption>"
                     f"<img src='{p.name}' style='max-width:100%'></figure>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>anytrack QC report</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
 h1{{font-size:20px}} table{{border-collapse:collapse;margin:12px 0}}
 th,td{{border:1px solid #ccc;padding:4px 8px;font-size:13px;text-align:right}}
 figure{{margin:18px 0}} figcaption{{font-weight:600;margin-bottom:6px}}
</style></head><body>
<h1>anytrack QC report</h1>
<p>{summary.get('n_rois', 0)} ROI(s), {summary.get('n_frames', 0)} frames.</p>
{pose_html}
{table}
{imgs}
</body></html>"""
    path = out_dir / "qc_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def _resolve_roi_bbox(name: str, video, df: pd.DataFrame, pad_frac: float = 0.06):
    """Full-res ``(x0, y0, x1, y1)`` for a ROI: its geometry if known, else the
    centroid extent in ``df`` (padded). Returns None if the ROI is unknown."""
    for r in getattr(video, "rois", []) or []:
        if str(r.name) == str(name):
            return (max(0.0, r.cx - r.r), max(0.0, r.cy - r.r), r.cx + r.r, r.cy + r.r)
    g = df[df["roi"].astype(str) == str(name)] if "roi" in df.columns else df.iloc[0:0]
    if g.empty:
        return None
    # No ROI geometry (e.g. the standalone CLI): approximate a square window from
    # the centroid extent, padded and floored so a near-stationary fly still gets
    # a usable view. The session path uses the true arena bbox above instead.
    x0, x1 = float(g["x"].min()), float(g["x"].max())
    y0, y1 = float(g["y"].min()), float(g["y"].max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max((x1 - x0) / 2.0, (y1 - y0) / 2.0) * (1.0 + pad_frac) + 120.0
    half = max(half, 160.0)
    return (cx - half, cy - half, cx + half, cy + half)


def render_overlay(
    video,
    df: pd.DataFrame,
    cfg,
    out_path: Path,
    max_frames: int = 0,
    trail_len: int = 30,
    show_progress: bool = False,
    pose_df: Optional[pd.DataFrame] = None,
    crop_roi: Optional[str] = None,
    follow: Optional[str] = None,
    follow_size: int = 256,
    crop_output_size: int = 0,
) -> Tuple[Optional[Path], int]:
    """Burn tracking annotations onto the source video.

    Draws ROI circles + name labels, a fading trail per track, and a centroid
    dot (ROI-colored) with a red ring on flagged frames. The video is written
    **downscaled** (``cfg.qc_overlay_downscale``) and, when FFmpeg is available,
    decoded and H.264-encoded through FFmpeg pipes — far faster and far smaller
    than full-res cv2 ``mp4v``. Falls back to cv2 if FFmpeg is missing.

    Returns ``(path, frames_written)``; ``path`` is None if nothing could be
    written. ``max_frames=0`` renders the whole video.
    """
    out_path = Path(out_path)
    flagged = compute_flags(df, video, cfg)

    cap = cv2.VideoCapture(str(video.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(getattr(video, "fps_nominal", 0) or 0) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_w <= 0 or src_h <= 0:
        return None, 0

    ds = max(1, int(getattr(cfg, "qc_overlay_downscale", 2)))
    w = max(2, src_w // ds); w -= w % 2   # even dims for H.264/yuv420p
    h = max(2, src_h // ds); h -= h % 2
    sx = w / src_w
    sy = h / src_h
    crf = int(getattr(cfg, "qc_overlay_crf", 23))
    stride = max(1, int(getattr(cfg, "qc_overlay_stride", 1)))
    hw_wanted = bool(getattr(cfg, "qc_overlay_hw", True))
    hw_bitrate = str(getattr(cfg, "qc_overlay_hw_bitrate", "6M"))

    rois = list(getattr(video, "rois", []) or [])
    names = [r.name for r in rois] + [str(n) for n in flagged.get("roi", pd.Series(dtype=str)).unique()]
    colors = roi_color_map(names)

    # Annotation geometry is kept in FULL-RES coords; a per-frame transform maps
    # it into the output canvas. For crops we scale the source region UP to the
    # output size *before* drawing, so the skeleton/labels stay crisp.
    rois_full = [(str(r.name), float(r.cx), float(r.cy), float(r.r),
                  colors.get(str(r.name), (90, 90, 90))) for r in rois]
    frame_groups = {int(f): g for f, g in flagged.groupby("frame")}
    trails: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (rn, tid), g in flagged.groupby(["roi", "track_id"]):
        gg = g.sort_values("frame")
        trails[(rn, tid)] = (gg["frame"].to_numpy(), gg["x"].to_numpy(), gg["y"].to_numpy())
    F = cv2.FONT_HERSHEY_SIMPLEX

    pose_prep = None
    if pose_df is not None and not pose_df.empty:
        from .pose.qc_pose import prepare_pose_overlay, draw_pose
        from .pose.skeleton import get_skeleton
        pose_prep = prepare_pose_overlay(pose_df, get_skeleton(cfg), cfg)

    def _even(v, hi=None):
        v = max(2, int(round(v))); v -= v % 2
        return min(v, hi - (hi % 2)) if hi else v

    # --- resolve crop geometry (square, full-res) -> output canvas -------------
    crop_active = bool(crop_roi or follow)
    center_fn = None      # source_fidx -> (cx, cy) full-res crop center
    csize = 0             # full-res square crop side
    if follow:
        fname, _, ftid = str(follow).partition(":")
        fg = flagged[flagged.get("roi", pd.Series(index=flagged.index, dtype=str)).astype(str) == fname]
        if ftid and "track_id" in fg.columns:
            fg = fg[fg["track_id"].astype(str) == ftid]
        elif "track_id" in fg.columns and not fg.empty:
            fg = fg[fg["track_id"] == fg["track_id"].value_counts().idxmax()]
        fg = fg.dropna(subset=["x", "y"]).sort_values("frame")
        if fg.empty:
            raise ValueError(f"--follow: no track for {follow!r} in the data")
        ffr = fg["frame"].to_numpy(); fcx = fg["x"].to_numpy(); fcy = fg["y"].to_numpy()
        csize = _even(min(follow_size, src_w, src_h))

        def center_fn(fidx, ffr=ffr, fcx=fcx, fcy=fcy):
            j = max(0, min(int(np.searchsorted(ffr, fidx, side="right")) - 1, len(ffr) - 1))
            return float(fcx[j]), float(fcy[j])
    elif crop_roi:
        bbox = _resolve_roi_bbox(str(crop_roi), video, flagged)
        if bbox is None:
            raise ValueError(f"--crop-roi: unknown ROI {crop_roi!r}")
        bx0, by0, bx1, by1 = bbox
        ccx, ccy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        csize = _even(min(max(bx1 - bx0, by1 - by0), src_w, src_h))
        center_fn = lambda fidx, ccx=ccx, ccy=ccy: (ccx, ccy)   # noqa: E731

    def crop_origin(fidx):
        cx, cy = center_fn(fidx)
        ox = int(min(max(0, round(cx - csize / 2)), src_w - csize))
        oy = int(min(max(0, round(cy - csize / 2)), src_h - csize))
        return ox, oy

    out_size = _even(int(getattr(cfg, "qc_crop_output_size", 800)) if crop_output_size <= 0
                     else crop_output_size)
    if crop_active:
        scl = out_size / csize
        enc_w = enc_h = out_size
    else:
        scl = None
        enc_w, enc_h = w, h

    mag = scl if crop_active else max(sx, sy)   # annotation size follows magnification
    th = max(1, round(2 * mag)); dot = max(2, round(4 * mag))
    ring = max(4, round(8 * mag)); pose_radius = max(2, round(3 * mag))

    def draw(canvas, fidx, ox, oy, scx, scy):
        """Draw all annotations onto ``canvas`` mapping full-res coords via
        ``(X-ox)*scx, (Y-oy)*scy``."""
        def tx(X):
            return int(round((X - ox) * scx))

        def ty(Y):
            return int(round((Y - oy) * scy))
        for (nm, cx, cy, rr, col) in rois_full:
            cv2.circle(canvas, (tx(cx), ty(cy)), max(1, int(rr * scx)), col, th)
            cv2.putText(canvas, nm, (tx(cx - rr), ty(cy - rr) - 4), F, 0.5, col, 1, cv2.LINE_AA)
        for (rn, _tid), (fr, xs, ys) in trails.items():
            col = colors.get(str(rn), (0, 180, 180))
            lo = int(np.searchsorted(fr, fidx - trail_len))
            hi = int(np.searchsorted(fr, fidx, side="right"))
            if hi - lo >= 2:
                pts = np.stack([(xs[lo:hi] - ox) * scx, (ys[lo:hi] - oy) * scy], axis=1).astype(np.int32)
                cv2.polylines(canvas, [pts], False, col, 1)
        rows = frame_groups.get(fidx)
        if rows is not None:
            for _, r in rows.iterrows():
                col = colors.get(str(r["roi"]), (0, 255, 0))
                cx, cy = tx(r["x"]), ty(r["y"])
                cv2.circle(canvas, (cx, cy), dot, col, -1)
                if any(bool(r.get(c, False)) for c in FLAG_COLUMNS):
                    cv2.circle(canvas, (cx, cy), ring, (0, 0, 255), th)
        if pose_prep is not None:
            draw_pose(canvas, fidx, pose_prep, lambda X, Y: (tx(X), ty(Y)),
                      thickness=th, radius=pose_radius)
        cv2.putText(canvas, f"frame {fidx}", (8, 22), F, 0.6, (255, 255, 255), 1)

    # Per-backend frame transforms (ffmpeg gets a pre-filtered frame; cv2 a full
    # frame). Crop = scale source region up to out_size, THEN draw.
    if crop_active:
        def _crop_scale(frame_full, fidx):
            ox, oy = crop_origin(fidx)
            sub = frame_full[oy:oy + csize, ox:ox + csize]
            canvas = cv2.resize(sub, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            draw(canvas, fidx, ox, oy, scl, scl)
            return canvas

        cv2_transform = _crop_scale
        if follow:
            ff_transform = _crop_scale                      # ffmpeg delivers full-res
        else:
            def ff_transform(frame, fidx):                  # ffmpeg pre-cropped+scaled
                ox, oy = crop_origin(fidx)
                draw(frame, fidx, ox, oy, scl, scl)
                return frame
    else:
        def ff_transform(frame, fidx):                      # ffmpeg pre-scaled to (w,h)
            draw(frame, fidx, 0, 0, sx, sy)
            return frame

        def cv2_transform(frame, fidx):
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            draw(frame, fidx, 0, 0, sx, sy)
            return frame

    # ffmpeg decode filter + delivered size per mode.
    fs = f"framestep=step={stride}," if stride > 1 else ""
    if not crop_active:
        dec_vf = f"{fs}scale={w}:{h}"; dec_w, dec_h = w, h
    elif follow:
        dec_vf = f"framestep=step={stride}" if stride > 1 else ""; dec_w, dec_h = src_w, src_h
    else:
        ox0, oy0 = crop_origin(0)
        dec_vf = f"{fs}crop={csize}:{csize}:{ox0}:{oy0},scale={out_size}:{out_size}"
        dec_w, dec_h = out_size, out_size

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            # The bar counts OUTPUT frames (one per written frame). With a stride
            # only every Nth source frame is written, so scale the total down.
            fc = int(getattr(video, "frame_count", 0) or 0)
            total = max_frames or ((fc + stride - 1) // stride if fc else None)
            pbar = tqdm(total=total, desc="  overlay",
                        bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        leave=True)
        except Exception:
            pbar = None

    written = 0
    try:
        if check_ffmpeg_available():
            # Prefer hardware H.264 (a single VideoToolbox stream is fast here,
            # unlike the 4-way contention in preprocessing); fall back to
            # libx264, then to cv2 mp4v.
            kinds = []
            if hw_wanted and _videotoolbox_available():
                kinds.append("hw")
            kinds.append("sw")
            for kind in kinds:
                try:
                    written = _overlay_ffmpeg(video.video_path, out_path, dec_w, dec_h, fps, crf,
                                              max_frames, ff_transform, pbar,
                                              stride=stride, encoder=kind, bitrate=hw_bitrate,
                                              enc_w=enc_w, enc_h=enc_h, dec_vf=dec_vf)
                    if written > 0:
                        break
                except Exception:
                    written = 0
                    continue
        if written <= 0:
            written = _overlay_cv2(video.video_path, out_path, fps, max_frames,
                                   cv2_transform, pbar, stride=stride,
                                   enc_w=enc_w, enc_h=enc_h)
    finally:
        if pbar is not None:
            pbar.close()

    if written <= 0:
        return None, 0
    return out_path, written


_VT_AVAILABLE: Optional[bool] = None


def _videotoolbox_available() -> bool:
    """Whether this FFmpeg build exposes the h264_videotoolbox encoder (cached)."""
    global _VT_AVAILABLE
    if _VT_AVAILABLE is None:
        try:
            out = subprocess.run([get_ffmpeg_path(), "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=15)
            _VT_AVAILABLE = "h264_videotoolbox" in out.stdout
        except Exception:
            _VT_AVAILABLE = False
    return _VT_AVAILABLE


def _overlay_ffmpeg(video_path, out_path, dec_w, dec_h, fps, crf, max_frames, annotate, pbar,
                    stride: int = 1, encoder: str = "sw", bitrate: str = "6M",
                    enc_w: Optional[int] = None, enc_h: Optional[int] = None,
                    dec_vf: str = "") -> int:
    """Decode (via ``dec_vf``, delivering ``dec_w×dec_h`` frames) and H.264-encode
    the annotated overlay through FFmpeg pipes. ``encoder`` is "hw" (VideoToolbox)
    or "sw" (libx264). ``annotate(frame, fidx)`` returns the frame to encode
    (``enc_w×enc_h``; defaults to ``dec_w/dec_h``)."""
    ffmpeg = get_ffmpeg_path()
    enc_w = int(enc_w or dec_w); enc_h = int(enc_h or dec_h)
    dec_cmd = [ffmpeg, "-v", "error", "-i", str(video_path)]
    if dec_vf:
        dec_cmd += ["-vf", dec_vf]
    dec_cmd += ["-vsync", "vfr"]
    if max_frames:
        dec_cmd += ["-frames:v", str(int(max_frames))]
    dec_cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    if encoder == "hw":
        venc = ["-c:v", "h264_videotoolbox", "-b:v", bitrate]
    else:
        venc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf)]
    enc_cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{enc_w}x{enc_h}", "-r", f"{fps:.6f}", "-i", "-",
               *venc, "-pix_fmt", "yuv420p", str(out_path)]

    frame_bytes = dec_w * dec_h * 3
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    written = 0
    try:
        while True:
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(dec_h, dec_w, 3).copy()
            out = annotate(frame, written * stride)  # k-th kept frame == source frame k*stride
            if out is None:
                out = frame
            enc.stdin.write(np.ascontiguousarray(out).tobytes())
            written += 1
            if pbar is not None:
                pbar.update(1)
            if max_frames and written >= max_frames:
                break
    finally:
        try:
            dec.stdout.close()
        except Exception:
            pass
        dec.terminate()
        try:
            enc.stdin.close()
        except Exception:
            pass
        enc.wait()
        dec.wait()
    if enc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg encode failed (returncode {enc.returncode}, encoder={encoder})")
    return written


def _overlay_cv2(video_path, out_path, fps, max_frames, transform, pbar, stride: int = 1,
                 enc_w: int = 0, enc_h: int = 0) -> int:
    """Fallback: cv2 decode (full frame) + ``transform`` + mp4v write (no FFmpeg).
    ``transform(full_frame, fidx)`` returns the ``enc_w×enc_h`` frame to write."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (int(enc_w), int(enc_h)))
    if not writer.isOpened():
        cap.release()
        return 0
    written = 0
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (fidx % stride != 0):
            fidx += 1
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out = np.ascontiguousarray(transform(frame, fidx))
        writer.write(out)
        written += 1
        fidx += 1
        if pbar is not None:
            pbar.update(1)
        if max_frames and written >= max_frames:
            break
    cap.release()
    writer.release()
    return written


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
    pose_df: Optional[pd.DataFrame] = None,
    crop_roi: Optional[str] = None,
    follow: Optional[str] = None,
    follow_size: int = 256,
    crop_output_size: int = 0,
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

    diagnostics = plot_diagnostics(df, video, cfg, out_dir)
    timeseries = plot_timeseries(df, cfg, out_dir)
    kinematics = plot_kinematics(df, cfg, out_dir)
    occupancy = plot_occupancy(df, video, out_dir)
    coverage = plot_coverage(df, video, out_dir)
    drift = plot_drift(df, cfg, out_dir)  # empty unless bg_drift_correction was on
    plots = diagnostics + timeseries + kinematics + occupancy + coverage + drift

    # Pose QC (B5): confidence plot + per-instance flags, when a pose table is given.
    pose_confidence: List[Path] = []
    pose_flags_path: Optional[Path] = None
    if pose_df is not None and not pose_df.empty:
        from .pose.qc_pose import compute_pose_flags, plot_pose_confidence, pose_flag_summary
        from .pose.skeleton import get_skeleton
        pose_flags = compute_pose_flags(pose_df, get_skeleton(cfg), cfg)
        pose_flags_path = out_dir / "qc_pose_flags.parquet"
        write_diagnostics(pose_flags, pose_flags_path, fmt="parquet")
        summary["pose"] = pose_flag_summary(pose_flags)
        pose_confidence = plot_pose_confidence(pose_df, cfg, out_dir)
        plots = plots + pose_confidence
        summary_path.write_text(json.dumps(summary, indent=2))  # refresh with pose stats

    montage_path, n_montage = render_flagged_montage(
        video, df, cfg, out_dir / "qc_flagged_montage.png"
    )

    overlay_path: Optional[Path] = None
    n_overlay = 0
    if overlay:
        overlay_path, n_overlay = render_overlay(
            video, df, cfg, out_dir / "qc_overlay.mp4",
            max_frames=max_frames, show_progress=show_progress, pose_df=pose_df,
            crop_roi=crop_roi, follow=follow, follow_size=follow_size,
            crop_output_size=crop_output_size,
        )

    def _first(paths):
        return paths[0] if paths else None

    report_path = write_html_report(out_dir, summary, {
        "background": background_path,
        "overlay": overlay_path,
        "diagnostics": _first(diagnostics),
        "timeseries": _first(timeseries),
        "kinematics": _first(kinematics),
        "drift": _first(drift),
        "occupancy": _first(occupancy),
        "coverage": _first(coverage),
        "pose_confidence": _first(pose_confidence),
        "montage": montage_path,
    })

    return {
        "summary": summary,
        "summary_path": summary_path,
        "flags_path": flags_path,
        "pose_flags_path": pose_flags_path,
        "background_path": background_path,
        "plots": plots,
        "montage_path": montage_path,
        "montage_tiles": n_montage,
        "report_path": report_path,
        "overlay_path": overlay_path,
        "overlay_frames": n_overlay,
    }


def _qc_tracks_for(v, args, single: bool):
    """Resolve the tracks table for a video. Single: ``--tracks`` verbatim (a
    file). Batch (``--tracks`` is a dir): ``<stem>_tracks.parquet`` inside it."""
    if single:
        return args.tracks
    return Path(args.tracks) / f"{Path(v).stem}_tracks.parquet"


def _qc_pose_for(v, args, single: bool):
    """Resolve the optional pose table (mirrors :func:`_qc_tracks_for`)."""
    if args.pose is None:
        return None
    if single:
        return args.pose
    return Path(args.pose) / f"{Path(v).stem}_pose.parquet"


def _qc_out_for(v, args, cfg, single: bool) -> Path:
    """QC output dir: single → the base dir; batch → ``<base>/<stem>_qc``."""
    base = Path(args.out_dir or (Path(getattr(cfg, "output_dir", "") or ".") / "qc"))
    return base if single else base / f"{Path(v).stem}_qc"


def _qc_one(video_path, idx: int, n: int, *, args, cfg, single: bool):
    """QC one video; return the run_qc result dict (or None on skip/failure).

    Batch mode (``n > 1``) renders a compact animated per-video line; single mode
    keeps the verbose written-to listing. Serial by design — QC plotting
    (matplotlib) + frame annotation run in-process, so they don't thread-parallel
    the way tracking's subprocess pool does.
    """
    from types import SimpleNamespace
    batch = n > 1
    bp = None
    if batch:
        from .cli_progress import BatchProgress
        bp = BatchProgress(idx, n, Path(video_path).name)

    tp = _qc_tracks_for(video_path, args, single)
    if tp is None or not Path(tp).exists():
        msg = f"skip: no tracks ({Path(tp).name if tp else '—'})"
        bp.finish(msg, ok=False) if bp else print(f"{Path(video_path).name}: {msg}")
        return None

    if batch:
        bp.phase("qc", "reading tracks…")
    df = _read_tracks(tp)
    pose_df = None
    pp = _qc_pose_for(video_path, args, single)
    if pp is not None and Path(pp).exists():
        pose_df = _read_tracks(pp)

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    video = SimpleNamespace(video_path=Path(video_path), width=w, height=h,
                            frame_count=nfr, fps_nominal=fps, rois=[])

    out_dir = _qc_out_for(video_path, args, cfg, single)
    if batch:
        bp.phase("qc", "overlay…" if not args.no_overlay else "plots…")
    res = run_qc(video, df, cfg, out_dir, overlay=not args.no_overlay,
                 max_frames=args.max_frames, show_progress=not batch, pose_df=pose_df,
                 crop_roi=args.crop_roi, follow=args.follow, follow_size=args.follow_size,
                 crop_output_size=args.crop_size)

    if batch:
        ov = (f"{res['overlay_frames']} overlay frames" if res.get("overlay_path")
              else ("plots only" if args.no_overlay else "overlay skipped"))
        bp.finish(f"{len(res['plots'])} plots · {ov} → {out_dir.name}")
    else:
        print(f"QC written to {out_dir}")
        print(f"  summary: {res['summary_path']}")
        print(f"  flags:   {res['flags_path']}")
        for p in res["plots"]:
            print(f"  plot:    {p}")
        if res["overlay_path"] is not None:
            print(f"  overlay: {res['overlay_path']} ({res['overlay_frames']} frames)")
        elif not args.no_overlay:
            print("  overlay: skipped (no mp4 encoder available)")
    return res


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: generate QC artifacts from a finished tracking run (A9).

    ``--video`` is a single file or a **directory** (batch): the tracks/pose/out
    args are then directories and each video resolves to ``<stem>_tracks.parquet``
    / ``<stem>_pose.parquet`` / ``<base>/<stem>_qc`` (mirrors anytrack-run/-label).

    Loads a tracks table + the source video and writes overlay/plots/flags/
    summary into an output directory. Arena circles in the overlay require ROI
    geometry, which a raw tracks table lacks — run QC from a session to get
    them; the CLI still draws centroids, trails, flags, and all metrics.
    """
    import argparse
    from .config import load_config

    ap = argparse.ArgumentParser(
        prog="anytrack-qc",
        description="Generate QC artifacts (overlay, plots, flags, summary) from a finished run.",
    )
    ap.add_argument("--video", required=True, type=Path, help="Source video.")
    ap.add_argument("--tracks", required=True, type=Path,
                    help="Tracks table (.parquet or .csv) with roi/track_id/frame/x/y columns.")
    ap.add_argument("--pose", type=Path, default=None,
                    help="Pose table (<stem>_pose.parquet) to add skeleton overlay + confidence + flags.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <cfg.output_dir or .>/qc).")
    ap.add_argument("--no-overlay", action="store_true", help="Skip the overlay video (fast).")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Cap overlay frames (0 = whole video).")
    ap.add_argument("--overlay-stride", type=int, default=None,
                    help="Render every Nth frame in the overlay (default from config: 5; 1 = every frame).")
    ap.add_argument("--qc-full", "--qc_full", dest="qc_full", action="store_true",
                    help="Full-quality overlay: stride=1, downscale=1, crf=16.")
    ap.add_argument("--crop-roi", default=None,
                    help="Crop the overlay to one ROI (arena) by name, e.g. arena_02.")
    ap.add_argument("--follow", default=None,
                    help="Crop to a moving window that follows a track: 'ROI' or 'ROI:track_id'.")
    ap.add_argument("--follow-size", type=int, default=256,
                    help="Follow-window size in full-res px (default 256).")
    ap.add_argument("--crop-size", type=int, default=0,
                    help="Output px (square) for a cropped overlay (default from config: 800).")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.qc_full:
        cfg.qc_overlay_stride = 1
        cfg.qc_overlay_downscale = 1
        cfg.qc_overlay_crf = 16
    if args.overlay_stride is not None:  # explicit stride wins over --qc-full
        cfg.qc_overlay_stride = max(1, args.overlay_stride)

    from .video_select import resolve_videos, validate_openable

    # --video may be a single file (unchanged) or a directory (batch mode). In
    # batch mode --tracks/--pose/--out-dir are directories, resolved per video.
    single = not Path(args.video).is_dir()
    if single:
        if not args.video.exists():
            ap.error(f"video not found: {args.video}")
        if not args.tracks.exists():
            ap.error(f"tracks not found: {args.tracks}")
        if args.pose is not None and not args.pose.exists():
            ap.error(f"pose table not found: {args.pose}")
    else:
        if Path(args.tracks).suffix.lower() in (".parquet", ".pq", ".csv"):
            ap.error("batch mode (--video is a directory): --tracks must be a directory")
        if args.pose is not None and Path(args.pose).suffix.lower() in (".parquet", ".pq", ".csv"):
            ap.error("batch mode (--video is a directory): --pose must be a directory")

    def _validator(v):
        ok, reason = validate_openable(v)
        if not ok:
            return False, reason
        tp = _qc_tracks_for(v, args, single=False)
        return (True, reason) if (tp and Path(tp).exists()) else (False, "no tracks")

    videos = resolve_videos(args.video, _validator)
    if not videos:
        ap.error(f"no videos to QC for {args.video}")

    n = len(videos)
    if n == 1:
        return 0 if _qc_one(videos[0], 1, 1, args=args, cfg=cfg, single=single) is not None else 1

    # Batch: banner + per-video animated line + summary. Serial (QC plotting +
    # annotation are in-process / matplotlib is not thread-safe).
    import time
    from .cli_progress import batch_banner, header, _fmt_dt
    batch_banner(n, "QC")
    t0 = time.perf_counter()
    results = [_qc_one(v, i + 1, n, args=args, cfg=cfg, single=single)
               for i, v in enumerate(videos)]
    n_ok = sum(1 for r in results if r is not None)
    header(f"QC done — {n_ok}/{n} video(s) in {_fmt_dt(time.perf_counter() - t0)}")
    return 0 if n_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
