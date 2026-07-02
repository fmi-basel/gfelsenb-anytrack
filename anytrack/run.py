"""
Headless full-pipeline run.

Runs the same :class:`TrackingSession` the GUI drives (ROI detection → track →
kinematics → additive ``x_roi/y_roi``) on a video, then writes the tracks table
to a chosen path. Optionally chains QC (A9) and dynamic crop export (A5). This
is the CLI counterpart to the GUI's "Run tracking", so the whole analysis can
be tested and saved from the command line without opening a window.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .io import load_video_asset
from .session import TrackingSession
from .writer import write_tracks, default_output_path
from .benchmark import ensure_rois
from .background import build_background_image
from .cli_progress import TqdmProgress, header, step, ok, info


# File suffixes that mean "--output is a file path"; anything else is a directory.
_TRACK_SUFFIXES = {".parquet", ".pq", ".csv"}


def _resolve_output_path(output, cfg, video):
    """Resolve the tracks output path.

    - ``None`` → the config default (``default_output_path``).
    - a path with a ``.parquet``/``.pq``/``.csv`` suffix → used verbatim (file).
    - anything else (a directory, e.g. ``out/`` or ``~/results``) → auto-named
      ``<input_stem>_tracks.parquet`` inside that directory.
    """
    if output is None:
        return default_output_path(cfg, video, kind="tracks")
    out = Path(output)
    if out.suffix.lower() in _TRACK_SUFFIXES:
        return out
    stem = Path(video.video_path).stem
    return out / f"{stem}_tracks.parquet"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="anytrack-run",
        description="Run the full tracking pipeline headless and write the results to a chosen path.",
    )
    ap.add_argument("--video", required=True, type=Path, help="Source video.")
    ap.add_argument("--timing", type=Path, default=None,
                    help="Per-frame timing CSV (default: <video>.csv).")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Output tracks path. A .parquet/.csv file is used as-is; "
                         "a directory (no file extension) auto-names it "
                         "<input_stem>_tracks.parquet inside it. "
                         "Default: <cfg.output_dir or video dir>/<stem>_tracks.<fmt>.")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fast", dest="fast", action="store_true", help="Force fast mode.")
    mode.add_argument("--no-fast", dest="fast", action="store_false",
                      help="Force legacy single-pass mode.")
    ap.set_defaults(fast=None)

    ap.add_argument("--downscale", type=int, choices=[1, 2, 4], default=None,
                    help="ROI downscale factor for fast mode (overrides config).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel tracking workers for fast mode (overrides config).")
    ap.add_argument("--qc", action="store_true",
                    help="Also write QC artifacts (overlay/plots/flags/summary).")
    ap.add_argument("--qc-max-frames", type=int, default=0,
                    help="Cap QC overlay frames (0 = whole video).")
    ap.add_argument("--qc-overlay-stride", type=int, default=None,
                    help="Render every Nth frame in the QC overlay (default from config: 5; 1 = every frame).")
    ap.add_argument("--crops", action="store_true",
                    help="Also export centroid-centered crops (A5).")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.fast is not None:
        cfg.fast_mode = args.fast
    if args.downscale is not None:
        cfg.roi_downscale = args.downscale
    if args.workers is not None:
        cfg.n_tracking_workers = args.workers
    if args.qc_overlay_stride is not None:
        cfg.qc_overlay_stride = max(1, args.qc_overlay_stride)

    if not args.video.exists():
        ap.error(f"video not found: {args.video}")
    timing = args.timing or args.video.with_suffix(".csv")
    if not Path(timing).exists():
        ap.error(f"timing CSV not found: {timing} (pass --timing)")

    video = load_video_asset(args.video, Path(timing))

    header(f"anytrack · {args.video.name}")
    info(f"mode: {'fast' if cfg.fast_mode else 'legacy'}   frames: {video.frame_count}   timing: {Path(timing).name}")

    step("Detect ROIs (background + arena detection) …")
    # Build the model background once; reuse it for ROI detection and QC.
    bg_img = build_background_image(video.video_path, cfg) if (not video.rois or args.qc) else None
    ensure_rois(video, cfg, background=bg_img)
    if not video.rois:
        info("No ROIs detected; nothing to track.")
        return 1
    ok(f"{len(video.rois)} ROI(s): {', '.join(r.name for r in video.rois)}")

    progress = TqdmProgress()
    t0 = time.perf_counter()
    session = TrackingSession(cfg=cfg, video=video)
    df = session.run(progress_hook=progress)
    progress.close()
    dt = time.perf_counter() - t0

    out = _resolve_output_path(args.output, cfg, video)
    fmt = "csv" if out.suffix.lower() == ".csv" else None
    write_tracks(df, out, fmt=fmt)
    rows_s = (len(df) / dt) if dt > 0 else 0.0
    ok(f"tracked {len(df)} rows in {dt:.1f}s ({rows_s:.0f} rows/s)")
    info(f"→ {out}")

    if args.qc:
        from .qc import run_qc
        step("QC (overlay + plots + flags + summary) …")
        qc_dir = out.parent / f"{out.stem}_qc"
        res = run_qc(video, df, cfg, qc_dir, overlay=True,
                     max_frames=args.qc_max_frames, show_progress=True,
                     background=bg_img)
        ok(f"QC → {qc_dir}" + (f" (overlay {res['overlay_frames']} frames)"
                               if res.get("overlay_path") else " (overlay skipped)"))
        if res.get("report_path"):
            info(f"report: {res['report_path']}")

    if args.crops:
        from .cropper import export_crops
        step("Export centroid crops …")
        crops_dir = out.parent / f"{out.stem}_crops"
        manifest = export_crops(video, df, cfg, out_dir=crops_dir, show_progress=True)
        ok(f"exported {len(manifest)} crops → {crops_dir}")

    header("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
