"""
Runtime benchmarking for anytrack tracking.

Promoted from ``tests/test_performance.py`` so it can be reused from the tests
and from the command line (``python -m anytrack.benchmark --video ...``).
Measures tracking FPS for the legacy and fast paths and writes a TOML report.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

from .models import VideoAsset
from .config import AnyTrackConfig, load_config
from .io import load_video_asset
from .roi import detect_circular_rois
from .background import build_background
from .tracking import track_video
from .tracking_fast import track_video_fast


def ensure_rois(video: VideoAsset, cfg: AnyTrackConfig) -> VideoAsset:
    """Detect circular ROIs on ``video`` in place if none are defined.

    Uses the current GMM background API + Hough ROI detection.
    """
    if video.rois:
        return video
    bg = build_background(
        str(video.video_path),
        n_samples=cfg.gmm_n_samples,
        bic_improvement=cfg.gmm_bic_improvement,
        min_std=cfg.gmm_min_std,
        reg_covar=cfg.gmm_reg_covar,
        lowp=cfg.gmm_lowp,
        arena_detection=cfg.arena_detection_enabled,
        arena_min_area_frac=cfg.arena_min_area_frac,
        arena_blur_sigma=cfg.arena_blur_sigma,
    ).image
    video.rois = detect_circular_rois(
        bg,
        dp=cfg.roi_hough_dp,
        min_dist_ratio=cfg.roi_hough_min_dist_ratio,
        param1=cfg.roi_hough_param1,
        param2=cfg.roi_hough_param2,
        min_radius_ratio=cfg.min_radius_ratio,
        max_radius_ratio=cfg.max_radius_ratio,
    )
    return video


def benchmark_tracking(
    video: VideoAsset,
    cfg: AnyTrackConfig,
    n_frames: int = 1000,
    n_runs: int = 3,
    use_fast_mode: bool = False,
    verbose: bool = True,
) -> dict:
    """Run tracking ``n_runs`` times over ``n_frames`` frames and return FPS stats."""
    if video.timing is not None and len(video.timing) > n_frames:
        limited_timing = video.timing.head(n_frames).copy()
    else:
        limited_timing = video.timing
        n_frames = len(limited_timing) if limited_timing is not None else 0

    limited_video = VideoAsset(
        video_path=video.video_path,
        timing_csv_path=video.timing_csv_path,
        timing=limited_timing,
        fps_nominal=video.fps_nominal,
        frame_count=n_frames,
        width=video.width,
        height=video.height,
        rois=video.rois,
    )

    fps_per_run = []
    times_per_run = []
    mode_label = "FAST" if use_fast_mode else "LEGACY"

    for run_idx in range(n_runs):
        if verbose:
            print(f"  [{mode_label}] Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)

        start = time.perf_counter()
        if use_fast_mode:
            _ = track_video_fast(limited_video, cfg, progress_hook=None, cleanup=True)
        else:
            _ = track_video(limited_video, cfg, progress_hook=None, cancel_event=None)
        elapsed = time.perf_counter() - start

        fps = n_frames / elapsed if elapsed > 0 else 0.0
        fps_per_run.append(fps)
        times_per_run.append(elapsed)
        if verbose:
            print(f"{fps:.1f} fps ({elapsed:.2f}s)")

    fps_mean = mean(fps_per_run) if fps_per_run else 0.0
    fps_std_val = stdev(fps_per_run) if len(fps_per_run) > 1 else 0.0
    ms_per_frame = 1000.0 / fps_mean if fps_mean > 0 else 0.0

    return {
        "n_frames": n_frames,
        "n_runs": n_runs,
        "fps_per_run": fps_per_run,
        "fps_mean": fps_mean,
        "fps_std": fps_std_val,
        "ms_per_frame_mean": ms_per_frame,
        "total_time_s": sum(times_per_run),
    }


def write_benchmark_report(
    results: dict,
    output_path: Path,
    video: VideoAsset,
    cfg: AnyTrackConfig,
    metadata: Optional[dict] = None,
    git_commit: Optional[str] = None,
):
    """Write benchmark results to a TOML file."""
    lines = []

    lines.append("[metadata]")
    lines.append(f'timestamp = "{datetime.now().isoformat()}"')
    if git_commit:
        lines.append(f'git_commit = "{git_commit}"')
    if metadata:
        if "cpu_brand" in metadata:
            lines.append(f'machine = "{metadata["cpu_brand"]}"')
        if "memory_gb" in metadata:
            lines.append(f"memory_gb = {metadata['memory_gb']}")
        lines.append(f'platform = "{metadata.get("platform", "unknown")}"')
        lines.append(f'python_version = "{metadata.get("python_version", "unknown")}"')
        lines.append(f'opencv_version = "{metadata.get("opencv_version", "unknown")}"')
    lines.append("")

    lines.append("[video]")
    lines.append(f'path = "{video.video_path.name}"')
    lines.append(f'resolution = "{video.width}x{video.height}"')
    lines.append(f"n_rois = {len(video.rois)}")
    lines.append(f"total_frames = {video.frame_count}")
    lines.append("")

    lines.append("[benchmark]")
    lines.append(f"n_frames = {results['n_frames']}")
    lines.append(f"n_runs = {results['n_runs']}")
    lines.append("")

    lines.append("[results]")
    fps_list = ", ".join(f"{x:.2f}" for x in results["fps_per_run"])
    lines.append(f"fps_per_run = [{fps_list}]")
    lines.append(f"fps_mean = {results['fps_mean']:.2f}")
    lines.append(f"fps_std = {results['fps_std']:.2f}")
    lines.append(f"ms_per_frame_mean = {results['ms_per_frame_mean']:.2f}")
    lines.append(f"total_time_s = {results['total_time_s']:.2f}")
    lines.append("")

    # Config section (relevant tracking parameters; GMM background API).
    lines.append("[config]")
    lines.append(f"gmm_n_samples = {cfg.gmm_n_samples}")
    lines.append(f'bgdiff_type = "{cfg.bgdiff_type}"')
    lines.append(f'thr_method = "{cfg.thr_method}"')
    lines.append(f"thr_fixed = {cfg.thr_fixed}")
    lines.append(f"morph_open = {cfg.morph_open}")
    lines.append(f"morph_close = {cfg.morph_close}")
    lines.append(f"use_kalman = {str(cfg.use_kalman).lower()}")
    lines.append(f"max_jump_px = {cfg.max_jump_px}")
    lines.append(f"fast_mode = {str(cfg.fast_mode).lower()}")
    lines.append(f"roi_downscale = {cfg.roi_downscale}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def write_stage_report(stages: dict, output_path: Path, video, cfg, git_commit: Optional[str] = None):
    """Write a per-stage benchmark breakdown to TOML (with git_commit metadata)."""
    lines = ["[metadata]", f'timestamp = "{datetime.now().isoformat()}"']
    if git_commit:
        lines.append(f'git_commit = "{git_commit}"')
    lines += [
        "",
        "[video]",
        f'path = "{video.video_path.name}"',
        f'resolution = "{video.width}x{video.height}"',
        f"n_rois = {len(video.rois)}",
        f"total_frames = {video.frame_count}",
        "",
        "[stages]",
    ]
    for k, v in stages.items():
        lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def benchmark_stages(video, cfg, n_frames: int = 2000) -> dict:
    """Per-stage timing for the fast path, measured from the real pipeline.

    Separates the costs the aggregate FPS conflates — FFmpeg preprocessing
    (decode-once), per-ROI background build, and the pure tracking loop — using
    the actual ``track_roi_video`` (via return_timing) rather than a
    re-implementation. Reports the tracking-alone throughput (frames/s for one
    animal) plus parallel full-video estimates. Requires ``video.rois``.
    """
    import time
    from .preprocess import extract_roi_videos, cleanup_roi_videos
    from .tracking_fast import track_roi_video

    if not video.rois:
        raise ValueError("benchmark_stages needs ROIs; call ensure_rois first")

    timing = video.timing
    if timing is not None and len(timing) > n_frames:
        timing = timing.head(n_frames).copy()
    n = len(timing) if timing is not None else 0

    t0 = time.perf_counter()
    pr = extract_roi_videos(
        video.video_path, video.rois,
        downscale=cfg.roi_downscale, use_hw_encode=cfg.use_hw_encode,
    )
    preprocess_s = time.perf_counter() - t0

    bg_builds, track_ss, frames = [], [], []
    try:
        for p in pr.values():
            _, tm = track_roi_video(p.video_path, p, timing, cfg, return_timing=True)
            bg_builds.append(tm["bg_build_s"])
            track_ss.append(tm["track_s"])
            frames.append(tm["n_frames"])
    finally:
        cleanup_roi_videos(pr)

    tot_frames = sum(frames)
    max_track = max(track_ss) if track_ss else 0.0
    # Production runs the ROIs in parallel workers, so the tracking-phase wall
    # time is the slowest worker's (background build + tracking loop).
    parallel_wall = max((b + t for b, t in zip(bg_builds, track_ss)), default=0.0)
    per_roi_fps = mean([f / t for f, t in zip(frames, track_ss) if t > 0]) if track_ss else 0.0

    return {
        "n_rois": len(pr),
        "n_frames": n,
        "preprocess_s": round(preprocess_s, 2),
        "bg_build_s_mean": round(mean(bg_builds), 2) if bg_builds else 0.0,
        "bg_build_s_max": round(max(bg_builds), 2) if bg_builds else 0.0,
        "track_s_mean": round(mean(track_ss), 2) if track_ss else 0.0,
        "track_s_max": round(max_track, 2),
        "tracking_fps_per_roi": round(per_roi_fps, 1),                                    # tracking ALONE (one animal)
        "tracking_crops_per_s": round(tot_frames / max_track, 1) if max_track else 0.0,   # all ROIs, parallel
        "video_fps_excl_preprocess": round(n / parallel_wall, 1) if parallel_wall else 0.0,
        "video_fps_incl_preprocess": round(n / (preprocess_s + parallel_wall), 1) if (preprocess_s + parallel_wall) else 0.0,
    }


def _git_commit() -> Optional[str]:
    """Short HEAD hash (with a ``-dirty`` suffix if the tree has local changes)."""
    try:
        r = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def main(argv=None):
    import dataclasses
    p = argparse.ArgumentParser(description="Benchmark anytrack tracking FPS.")
    p.add_argument("--video", required=True, help="Path to the video file")
    p.add_argument("--timing", default=None, help="Timing CSV (default: video stem + .csv)")
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--fast", action="store_true", help="Benchmark the fast path")
    p.add_argument("--stages", action="store_true",
                   help="Per-stage breakdown (preprocess / bg-build / pure tracking) instead of FPS runs")
    p.add_argument("--no-stage", action="store_true", help="Disable local staging for this run")
    p.add_argument("--report-dir", default="tests/reports")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.fast:
        cfg.fast_mode = True

    video_path = Path(args.video)
    timing_path = Path(args.timing) if args.timing else video_path.with_suffix(".csv")
    video = load_video_asset(video_path, timing_path)

    # Local staging (unless disabled) so reads/seeks hit a fast local disk.
    if not args.no_stage:
        from .staging import stage_video
        import time as _t
        _t0 = _t.perf_counter()
        local, staged = stage_video(video.video_path, cfg)
        if staged:
            print(f"staged -> {local}  ({round(_t.perf_counter() - _t0, 1)}s)")
            video = dataclasses.replace(video, video_path=local)

    ensure_rois(video, cfg)

    if args.stages:
        st = benchmark_stages(video, cfg, n_frames=args.frames)
        print("PER-STAGE (fast path):")
        for k, v in st.items():
            print(f"  {k}: {v}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.report_dir) / f"benchmark_stages_{ts}.toml"
        write_stage_report(st, out, video, cfg, git_commit=_git_commit())
        print(f"report={out}")
        return

    results = benchmark_tracking(
        video, cfg, n_frames=args.frames, n_runs=args.runs, use_fast_mode=args.fast
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "fast" if args.fast else "legacy"
    out = Path(args.report_dir) / f"benchmark_{label}_{ts}.toml"
    write_benchmark_report(results, out, video, cfg, git_commit=_git_commit())
    print(f"fps_mean={results['fps_mean']:.2f} +/- {results['fps_std']:.2f}  report={out}")


if __name__ == "__main__":
    main()
