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
from .cli_progress import (
    TqdmProgress, header, step, ok, info, _fmt_dt, batch_banner, batch_summary,
)


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


def _run_one(video_path, idx: int, n: int, *, args, cfg, single, group=None) -> Optional[int]:
    """Run the full pipeline on one video; return its tracked row count (or None).

    Module-scoped (not a closure) so it can be driven by the batch thread pool
    (Phase 4) and the batch benchmark. ``args``/``cfg``/``single`` are passed in
    explicitly. In batch mode (``n > 1``) progress renders as a compact animated
    line — from ``group.acquire(...)`` when a :class:`BatchProgressGroup` is given
    (concurrent), else a standalone ``BatchProgress``; single mode keeps the
    verbose boxed header/step/ok/info layout.
    """
    batch = n > 1
    bp = None
    if batch:
        if group is not None:
            bp = group.acquire(idx, Path(video_path).name)
        else:
            from .cli_progress import BatchProgress
            bp = BatchProgress(idx, n, Path(video_path).name)

    timing = args.timing if (single and args.timing) else Path(video_path).with_suffix(".csv")
    if not Path(timing).exists():
        if bp is not None:
            bp.finish(f"skip: timing CSV not found ({Path(timing).name})", ok=False)
        else:
            info(f"skip {Path(video_path).name}: timing CSV not found ({timing})")
        return None
    video = load_video_asset(Path(video_path), Path(timing))
    # One progress sink for the whole video: the animated line in batch mode,
    # tqdm bars in single mode. Shared across ROI detection → track → pose so
    # each stage draws a frame-based bar.
    progress = bp if batch else TqdmProgress()
    if not batch:
        header(f"anytrack · {Path(video_path).name}")
        info(f"mode: {'fast' if cfg.fast_mode else 'legacy'}   frames: {video.frame_count}   timing: {Path(timing).name}")
        step("Detect ROIs (background + arena detection) …")
    else:
        bp.phase("roi", "detecting ROIs…")

    # Build the model background once; reuse it for ROI detection and QC.
    bg_img = (build_background_image(video.video_path, cfg, progress_hook=progress)
              if (not video.rois or args.qc) else None)
    ensure_rois(video, cfg, background=bg_img)
    if not video.rois:
        if bp is not None:
            bp.finish("no ROIs detected", ok=False)
        else:
            info("No ROIs detected; nothing to track.")
        return None
    if not batch:
        ok(f"{len(video.rois)} ROI(s): {', '.join(r.name for r in video.rois)}")

    # Give the streaming path a uniformly-sampled per-ROI GMM instead of its
    # first-100-consecutive-frames GMM (which bakes in a fly that sits still at the
    # start → the tracker loses it). Built by sampling the whole video and
    # cropping+downscaling per ROI *before* fitting the GMM, matching the file
    # path exactly (downscale-then-GMM avoids arena-rim edge artifacts).
    roi_bgs = None
    if getattr(cfg, "bg_reuse_for_roi", False) and video.rois:
        from .preprocess import build_roi_backgrounds_uniform
        roi_bgs = build_roi_backgrounds_uniform(video.video_path, video.rois, cfg)

    t0 = time.perf_counter()
    session = TrackingSession(cfg=cfg, video=video)
    df = session.run(progress_hook=progress, roi_backgrounds=roi_bgs)
    if not batch:
        progress.close()
    dt = time.perf_counter() - t0

    out = _resolve_output_path(args.output, cfg, video)
    fmt = "csv" if out.suffix.lower() == ".csv" else None
    write_tracks(df, out, fmt=fmt)

    # Optionally dump the per-ROI backgrounds that actually drove tracking (the
    # uniformly-sampled per-ROI GMM + de-ghost from build_roi_backgrounds_uniform),
    # not the full-frame model background QC saves. One PNG per arena, at tracking
    # resolution (roi_downscale). None when bg_reuse_for_roi is off or sampling fell
    # back to the per-worker GMM.
    if getattr(args, "save_roi_bg", False):
        import cv2
        if roi_bgs:
            bg_dir = out.parent / f"{out.stem}_roi_bg"
            bg_dir.mkdir(parents=True, exist_ok=True)
            for name, img in roi_bgs.items():
                cv2.imwrite(str(bg_dir / f"{name}.png"), img)
            if not batch:
                ok(f"saved {len(roi_bgs)} per-ROI background(s) → {bg_dir}")
        elif not batch:
            info("per-ROI backgrounds not built (bg_reuse_for_roi off or sampling "
                 "failed → per-worker GMM); nothing to save.")

    pose_note = ""
    if not batch:
        rows_s = (len(df) / dt) if dt > 0 else 0.0
        ok(f"tracked {len(df)} rows in {dt:.1f}s ({rows_s:.0f} rows/s)")
        info(f"→ {out}")

    if cfg.pose_enabled and getattr(session, "pose_dataframe", None) is not None:
        from .writer import write_pose
        pose_out = out.with_name(out.stem.replace("_tracks", "") + "_pose" + out.suffix)
        write_pose(session.pose_dataframe, pose_out, fmt=fmt)
        pose_note = f" · {len(session.pose_dataframe)} pose rows"
        if not batch:
            ok(f"pose: {len(session.pose_dataframe)} rows ({session.pose_dataframe['keypoint'].nunique()} keypoints) → {pose_out}")
    elif cfg.pose_enabled and not batch:
        info("pose stage produced no output (see warnings above).")

    if args.qc or args.qc_full:
        from .qc import run_qc
        if batch:
            bp.phase("qc", "QC…")
        else:
            step("QC (overlay + plots + flags + summary) …")
        qc_dir = out.parent / f"{out.stem}_qc"
        res = run_qc(video, df, cfg, qc_dir, overlay=True,
                     max_frames=args.qc_max_frames, show_progress=not batch,
                     background=bg_img, pose_df=getattr(session, "pose_dataframe", None),
                     crop_roi=args.crop_roi, follow=args.follow, follow_size=args.follow_size,
                     crop_output_size=args.crop_size)
        if not batch:
            ok(f"QC → {qc_dir}" + (f" (overlay {res['overlay_frames']} frames)"
                                   if res.get("overlay_path") else " (overlay skipped)"))
            if res.get("report_path"):
                info(f"report: {res['report_path']}")

    if args.crops:
        from .cropper import export_crops
        if batch:
            bp.phase("qc", "crops…")
        else:
            step("Export centroid crops …")
        crops_dir = out.parent / f"{out.stem}_crops"
        manifest = export_crops(video, df, cfg, out_dir=crops_dir, show_progress=not batch)
        if not batch:
            ok(f"exported {len(manifest)} crops → {crops_dir}")

    if bp is not None:
        bp.finish(f"{len(video.rois)} ROIs · {len(df)} rows · {_fmt_dt(dt)}{pose_note}")
    return len(df)


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
    ap.add_argument("--stride", type=int, default=None,
                    help="Track every Nth frame and interpolate the rest (streaming path; overrides config).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel tracking workers for fast mode (overrides config).")
    ap.add_argument("--qc", action="store_true",
                    help="Also write QC artifacts (overlay/plots/flags/summary).")
    ap.add_argument("--qc-full", "--qc_full", dest="qc_full", action="store_true",
                    help="Full-quality overlay: stride=1, downscale=1, crf=16 (implies --qc).")
    ap.add_argument("--qc-max-frames", type=int, default=0,
                    help="Cap QC overlay frames (0 = whole video).")
    ap.add_argument("--qc-overlay-stride", type=int, default=None,
                    help="Render every Nth frame in the QC overlay (default from config: 5; 1 = every frame).")
    ap.add_argument("--crop-roi", default=None,
                    help="Crop the QC overlay to one ROI (arena) by name, e.g. arena_02.")
    ap.add_argument("--follow", default=None,
                    help="Crop the QC overlay to a window following a track: 'ROI' or 'ROI:track_id'.")
    ap.add_argument("--follow-size", type=int, default=256,
                    help="Follow-window size in full-res px (default 256).")
    ap.add_argument("--crop-size", type=int, default=0,
                    help="Output px (square) for a cropped QC overlay (default from config: 800).")
    ap.add_argument("--crops", action="store_true",
                    help="Also export centroid-centered crops (A5).")
    ap.add_argument("--save-roi-bg", "--save_roi_bg", dest="save_roi_bg", action="store_true",
                    help="Save the per-ROI backgrounds that drive tracking (uniform "
                         "per-ROI GMM + de-ghost) as one PNG per arena in "
                         "<output_stem>_roi_bg/. Distinct from the full-frame "
                         "qc_background.png written by --qc.")
    ap.add_argument("--pose", action="store_true",
                    help="Also run the pose stage (keypoints) after tracking (B5).")
    ap.add_argument("--pose-model", type=Path, default=None,
                    help="Trained sleap-nn model dir (sets sleap_model_path; implies --pose).")
    ap.add_argument("--pose-device", default=None, help="Pose torch device: auto|mps|cuda|cpu.")
    ap.add_argument("--pose-every-n", type=int, default=None,
                    help="Run pose on every Nth tracked frame (default from config: 1).")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.fast is not None:
        cfg.fast_mode = args.fast
    if args.downscale is not None:
        cfg.roi_downscale = args.downscale
    if args.stride is not None:
        cfg.track_stride = max(1, args.stride)
    if args.workers is not None:
        cfg.n_tracking_workers = args.workers
    if args.qc_full:
        cfg.qc_overlay_stride = 1
        cfg.qc_overlay_downscale = 1
        cfg.qc_overlay_crf = 16
    if args.qc_overlay_stride is not None:  # explicit stride wins over --qc-full
        cfg.qc_overlay_stride = max(1, args.qc_overlay_stride)

    if args.pose_model is not None:
        cfg.sleap_model_path = str(args.pose_model)
        args.pose = True
    if args.pose:
        cfg.pose_enabled = True
    if args.pose_device is not None:
        cfg.pose_device = args.pose_device
    if args.pose_every_n is not None:
        cfg.pose_every_n = max(1, args.pose_every_n)
    if (cfg.pose_enabled and str(cfg.pose_backend).lower().startswith("sleap")
            and not cfg.sleap_model_path):
        ap.error("--pose with the sleap-nn backend needs a trained model: pass --pose-model <dir> "
                 "(or set sleap_model_path in config). Train one with anytrack-train.")

    from .video_select import resolve_videos, validate_openable

    # --video may be a single file (unchanged) or a directory (batch mode).
    single = not Path(args.video).is_dir()

    def _validator(v):
        ok, reason = validate_openable(v)
        if not ok:
            return False, reason
        tm = args.timing if (single and args.timing) else Path(v).with_suffix(".csv")
        return (True, reason) if Path(tm).exists() else (False, "no timing CSV")

    if single and not Path(args.video).exists():
        ap.error(f"video not found: {args.video}")
    if single:
        _tm = args.timing or Path(args.video).with_suffix(".csv")
        if not Path(_tm).exists():
            ap.error(f"timing CSV not found: {_tm} (pass --timing)")
    videos = resolve_videos(args.video, _validator)
    if not videos:
        ap.error(f"no videos to process for {args.video}")

    n = len(videos)
    t_all = time.perf_counter()
    if n > 1:
        results = _run_batch(videos, args=args, cfg=cfg, single=single)
        dt_all = time.perf_counter() - t_all
        n_ok = sum(1 for r in results if r is not None)
        total_rows = sum(r for r in results if r is not None)
        batch_summary(n_ok, n, total_rows, dt_all)
    else:
        results = [_run_one(videos[0], 1, 1, args=args, cfg=cfg, single=single)]
        n_ok = sum(1 for r in results if r is not None)
        header("done")
    return 0 if n_ok else 1


def _detect_cores() -> int:
    """Logical cores available to this process (respects cgroup/affinity on Linux)."""
    import os
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS/Windows have no sched_getaffinity
        import multiprocessing as mp
        return max(1, mp.cpu_count())


def _batch_concurrency(cfg, n_videos: int) -> int:
    """How many videos to process at once — adapts to the machine, tunable via config.

    ``cfg.batch_concurrency`` (config file / --workers-style knob) forces the value;
    otherwise auto = ``core_budget // n_tracking_workers`` clamped to ``[1, n]``,
    where ``core_budget`` is ``cfg.batch_core_budget`` if set else the detected
    logical cores. Each video still uses its own pool, so K × per-video-workers ≈
    cores. On hybrid-core machines (Apple Silicon P+E) set ``batch_core_budget`` in
    ``config.toml`` to tune how aggressively efficiency cores are used.
    """
    if cfg.batch_concurrency and cfg.batch_concurrency > 0:
        return max(1, min(cfg.batch_concurrency, n_videos))
    cores = cfg.batch_core_budget or _detect_cores()
    per_video = max(1, int(getattr(cfg, "n_tracking_workers", 4) or 4))
    return max(1, min(max(1, cores // per_video), n_videos))


def _run_batch(videos, *, args, cfg, single):
    """Process a batch, up to K videos concurrently, with a multi-line live UI.

    Threads at the outer level are correct: each ``_run_one`` drives its own
    FFmpeg decode + multiprocessing tracking pool, so the outer thread is mostly
    waiting on subprocesses. Results are index-keyed to stay deterministic.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .cli_progress import BatchProgressGroup

    n = len(videos)
    k = _batch_concurrency(cfg, n)
    batch_banner(n, ("fast" if cfg.fast_mode else "legacy") + f" · {k}× concurrent")
    group = BatchProgressGroup(n)
    results = [None] * n
    try:
        with ThreadPoolExecutor(max_workers=k) as ex:
            futs = {ex.submit(_run_one, v, i + 1, n, args=args, cfg=cfg,
                              single=single, group=group): i
                    for i, v in enumerate(videos)}
            for fut, i in futs.items():
                try:
                    results[i] = fut.result()
                except Exception:  # a failed video shouldn't sink the batch
                    results[i] = None
    finally:
        group.close()
    return results


if __name__ == "__main__":
    raise SystemExit(main())
