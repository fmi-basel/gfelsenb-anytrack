# anytrack

**Modular, ROI-based centroid tracking for behavioural assays.**

`anytrack` turns a recorded multi-arena video (plus a per-frame timing CSV) into
per-fly trajectories and kinematics. It is built for local-search / odor-navigation
assays where several flies are filmed from above, one per circular arena.

!!! info "Active development"
    `anytrack` is at `0.2.x`. The classical tracking pipeline is feature-complete
    and validated on real recordings. **Pose estimation (SLEAP keypoints) is the
    focus of the next release, `0.3.0`.** The `0.x` version line signals that the
    API and CLI may still change between minor versions — see the
    [Roadmap](roadmap.md).

## What it does

Given an AVI/MP4/MOV of several circular arenas and a CSV of per-frame timestamps,
`anytrack`:

1. **Detects each arena** (ROI) as a Hough circle.
2. **Models a per-arena background** (adaptive percentile + de-ghosting).
3. **Detects the fly** each frame: background subtraction → arena-disk mask →
   threshold/morphology → contour → moment centroid + fitted ellipse.
4. **Links** detections into a trajectory with a Kalman-gated centroid tracker.
5. **Computes kinematics** from the *real* per-frame time intervals.
6. **Writes** tracks to Parquet/CSV — optionally with a
   [quality-control report](outputs.md#qc-report) and
   [odor-port distance](pipeline.md#odor-port-detection).

## Why it's fast

The pipeline **decodes each video once** and streams cropped, downscaled arena
frames through a FIFO to parallel tracking workers; on macOS it can use hardware
decode. A whole directory of videos is processed with configurable cross-video
concurrency, with a live multi-line progress display. See
[The pipeline](pipeline.md) for details.

## Get started

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — install with `uv` (needs Python ≥ 3.10 and FFmpeg).
- :material-rocket-launch: **[Quickstart](quickstart.md)** — track one video or a whole folder.
- :material-sitemap: **[The pipeline](pipeline.md)** — how a video becomes trajectories.
- :material-console: **[CLI reference](cli.md)** — every `anytrack` command.
- :material-tune: **[Configuration](configuration.md)** — every tunable setting.
- :material-table: **[Output formats](outputs.md)** — tracks columns, sidecars, QC.

</div>

## License

Released under the [MIT License](https://github.com/fmi-basel/gfelsenb-anytrack/blob/main/LICENSE).
