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
                    help="Output tracks path (.parquet or .csv). "
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

    if not args.video.exists():
        ap.error(f"video not found: {args.video}")
    timing = args.timing or args.video.with_suffix(".csv")
    if not Path(timing).exists():
        ap.error(f"timing CSV not found: {timing} (pass --timing)")

    video = load_video_asset(args.video, Path(timing))

    print(f"detecting ROIs on {args.video.name} ...")
    ensure_rois(video, cfg)
    if not video.rois:
        print("No ROIs detected; nothing to track.")
        return 1
    print(f"{len(video.rois)} ROI(s); fast_mode={cfg.fast_mode}")

    # Coarse progress: print each phase and 10% buckets, never per-frame spam.
    state = {"bucket": -1}

    def progress(event: str, payload: dict) -> None:
        if event == "done":
            return
        pct = payload.get("percent")
        if pct is None:
            label = payload.get("roi", "")
            print(f"  [{event}] {label}".rstrip())
            return
        bucket = int(pct * 10)
        if bucket != state["bucket"]:
            state["bucket"] = bucket
            print(f"  [{event}] {pct * 100:4.0f}%  {payload.get('roi', '')}".rstrip())

    t0 = time.perf_counter()
    session = TrackingSession(cfg=cfg, video=video)
    df = session.run(progress_hook=progress)
    dt = time.perf_counter() - t0

    out = Path(args.output) if args.output else default_output_path(cfg, video, kind="tracks")
    fmt = "csv" if out.suffix.lower() == ".csv" else None
    write_tracks(df, out, fmt=fmt)
    fps = (len(df) / dt) if dt > 0 else 0.0
    print(f"tracked {len(df)} rows in {dt:.1f}s ({fps:.0f} rows/s) -> {out}")

    if args.qc:
        from .qc import run_qc
        qc_dir = out.parent / f"{out.stem}_qc"
        run_qc(video, df, cfg, qc_dir, overlay=True, max_frames=args.qc_max_frames)
        print(f"QC artifacts -> {qc_dir}")

    if args.crops:
        from .cropper import export_crops
        crops_dir = out.parent / f"{out.stem}_crops"
        manifest = export_crops(video, df, cfg, out_dir=crops_dir)
        print(f"exported {len(manifest)} crops -> {crops_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
