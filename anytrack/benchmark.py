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
from .background import build_background, build_background_image
from .tracking import track_video
from .tracking_fast import track_video_fast


def ensure_rois(video: VideoAsset, cfg: AnyTrackConfig, background=None) -> VideoAsset:
    """Detect circular ROIs on ``video`` in place if none are defined.

    Uses the current GMM background API + Hough ROI detection. A prebuilt
    ``background`` image may be passed to avoid rebuilding it (e.g. when the
    caller also needs it for QC).
    """
    if video.rois:
        return video
    bg = background if background is not None else build_background_image(video.video_path, cfg)
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


def benchmark_pose(
    video,
    df,
    cfg: AnyTrackConfig,
    *,
    max_frames: int = 150,
    devices=("mps",),
    batch_values=(4,),
    every_n_values=(1,),
    verbose: bool = True,
) -> dict:
    """Time the sleap-nn pose stage across device × batch × every_n (Milestone B6).

    Measures **crops/s** (instances/s) and pose-frames/s on ``max_frames`` pose
    frames per run, plus a **decode+crop floor** (``iter_crop_batches`` with no
    model) so the video-I/O share of the cost is visible. Each run extrapolates
    the full-video pose time at that ``every_n``. Requires sleap-nn + a trained
    model (``cfg.sleap_model_path``).
    """
    import dataclasses
    import time as _t
    from .pose import build_engine, get_skeleton
    from .pose.infer import run_pose_sleap
    from .cropper import iter_crop_batches

    valid = df.dropna(subset=[c for c in ("x", "y") if c in df.columns]) \
        if {"x", "y"} & set(df.columns) else df
    all_frames = sorted(int(f) for f in valid["frame"].unique())
    if not all_frames:
        raise ValueError("no valid frames in tracks for the pose benchmark")

    def capped(every_n):
        sel = [f for f in all_frames if f % every_n == 0][:max_frames]
        return valid[valid["frame"].isin(sel)].copy()

    # decode+crop floor: read + crop the first max_frames frames, no model.
    cdf0 = capped(1)
    cfg0 = dataclasses.replace(cfg, pose_every_n=1)
    n_dec, t0 = 0, _t.perf_counter()
    for _crops, metas in iter_crop_batches(video, cdf0, cfg0):
        n_dec += len(metas)
    dec_s = _t.perf_counter() - t0
    decode_floor = {"n_instances": n_dec, "seconds": round(dec_s, 2),
                    "crops_s": round(n_dec / dec_s, 1) if dec_s > 0 else 0.0}
    if verbose:
        print(f"  [pose] decode+crop floor: {decode_floor['crops_s']} crops/s "
              f"({n_dec} crops, {dec_s:.1f}s)")

    total_frames_full = int(getattr(video, "frame_count", 0) or (all_frames[-1] + 1))
    runs = []
    for dev in devices:
        cfg_dev = dataclasses.replace(cfg, pose_device=dev, pose_backend="sleap-nn")
        try:
            eng = build_engine(cfg_dev)
        except Exception as e:  # noqa: BLE001 - a device may be unavailable
            if verbose:
                print(f"  [pose] device={dev}: engine load failed ({type(e).__name__}); skipping")
            continue
        for bs in batch_values:
            try:
                if getattr(eng, "predictor", None) is not None:
                    eng.predictor.batch_size = int(bs)
            except Exception:
                pass
            for en in every_n_values:
                cdf = capped(en)
                cfg_run = dataclasses.replace(cfg_dev, pose_every_n=1)  # cdf already sub-sampled
                t0 = _t.perf_counter()
                _ = run_pose_sleap(video, cdf, cfg_run, eng)
                secs = _t.perf_counter() - t0
                n_frames = int(cdf["frame"].nunique())
                n_inst = int(len(cdf))
                frames_s = n_frames / secs if secs > 0 else 0.0
                crops_s = n_inst / secs if secs > 0 else 0.0
                full_frames = (total_frames_full + en - 1) // en
                full_s = full_frames / frames_s if frames_s > 0 else 0.0
                run = {"device": dev, "batch": int(bs), "every_n": int(en),
                       "n_frames": n_frames, "n_instances": n_inst, "seconds": round(secs, 2),
                       "frames_s": round(frames_s, 1), "crops_s": round(crops_s, 1),
                       "full_video_pose_s": round(full_s, 1)}
                runs.append(run)
                if verbose:
                    print(f"  [pose] device={dev} batch={bs} every_n={en}: "
                          f"{crops_s:.1f} crops/s ({frames_s:.1f} fps, {secs:.1f}s, {n_frames} frames) "
                          f"→ full video ~{full_s:.0f}s")

    best = max(runs, key=lambda r: r["crops_s"], default=None)
    return {"decode_floor": decode_floor, "runs": runs, "best": best,
            "max_frames": max_frames, "n_keypoints": get_skeleton(cfg).n_nodes,
            "total_frames_full": total_frames_full}


def write_pose_report(results: dict, output_path: Path, video, cfg, git_commit: Optional[str] = None):
    """Write the pose benchmark (decode floor + per-run table) to TOML."""
    rois = getattr(video, "rois", []) or []
    lines = ["[metadata]", f'timestamp = "{datetime.now().isoformat()}"']
    if git_commit:
        lines.append(f'git_commit = "{git_commit}"')
    lines += [
        "", "[video]",
        f'path = "{Path(video.video_path).name}"',
        f'resolution = "{getattr(video, "width", 0)}x{getattr(video, "height", 0)}"',
        f"n_rois = {len(rois)}",
        f"total_frames = {getattr(video, 'frame_count', 0)}",
        "", "[pose_benchmark]",
        f'model = "{cfg.sleap_model_path}"',
        f"max_frames = {results['max_frames']}",
        f"n_keypoints = {results['n_keypoints']}",
        f"decode_floor_crops_s = {results['decode_floor']['crops_s']}",
    ]
    if results.get("best"):
        b = results["best"]
        lines.append(f'best = "{b["crops_s"]} crops/s @ device={b["device"]} batch={b["batch"]}"')
    lines.append("")
    for r in results["runs"]:
        lines.append("[[run]]")
        for k, v in r.items():
            lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


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
    # Pose benchmark (B6)
    p.add_argument("--pose", action="store_true",
                   help="Benchmark the pose stage (crops/s) instead of tracking")
    p.add_argument("--tracks", default=None, help="Tracks table (.parquet/.csv) for --pose")
    p.add_argument("--model", default=None,
                   help="Trained sleap-nn model dir for --pose (else cfg.sleap_model_path)")
    p.add_argument("--pose-max-frames", type=int, default=150,
                   help="Pose frames per run for --pose (default 150)")
    p.add_argument("--devices", default="mps,cpu", help="--pose device sweep (comma list)")
    p.add_argument("--batches", default="4,16", help="--pose batch-size sweep (comma list)")
    p.add_argument("--every-n", default="1,5", help="--pose every_n sweep (comma list)")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.fast:
        cfg.fast_mode = True

    video_path = Path(args.video)
    timing_path = Path(args.timing) if args.timing else video_path.with_suffix(".csv")
    video = load_video_asset(video_path, timing_path)

    if args.pose:
        import pandas as pd
        if not args.tracks or not Path(args.tracks).exists():
            p.error("--pose needs --tracks <parquet/csv>")
        tp = Path(args.tracks)
        df = pd.read_csv(tp) if tp.suffix.lower() == ".csv" else pd.read_parquet(tp)
        if args.model:
            cfg.sleap_model_path = args.model
        if not cfg.sleap_model_path:
            p.error("--pose needs a trained model: pass --model <dir> (or set sleap_model_path)")
        cfg.pose_backend = "sleap-nn"
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        batches = [int(b) for b in args.batches.split(",") if b.strip()]
        every_ns = [int(e) for e in args.every_n.split(",") if e.strip()]
        print("POSE BENCHMARK:")
        res = benchmark_pose(video, df, cfg, max_frames=args.pose_max_frames,
                             devices=devices, batch_values=batches, every_n_values=every_ns)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.report_dir) / f"benchmark_pose_{ts}.toml"
        write_pose_report(res, out, video, cfg, git_commit=_git_commit())
        if res["best"]:
            b = res["best"]
            print(f"best={b['crops_s']} crops/s @ device={b['device']} batch={b['batch']}  report={out}")
        return

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
