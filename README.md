# anytrack

**Modular, ROI-based centroid tracking for behavioural assays.** `anytrack` turns a
recorded multi-arena video (plus a per-frame timing CSV) into per-fly trajectories
and kinematics, with a fast streaming pipeline, batch processing, quality-control
reports, and optional odor-port detection.

> **Status:** active development (`0.x`). The classical tracking pipeline is
> feature-complete and validated; **pose estimation (SLEAP) is planned for 0.3.0**.
> The `0.x` line means the API/CLI may still change between minor versions.

## What it does

Given an AVI (or MP4/MOV) of several circular arenas filmed from above and a CSV of
per-frame timestamps, `anytrack`:

1. detects each arena (ROI) as a Hough circle,
2. models a per-arena background,
3. detects the fly each frame (background subtraction → arena-disk mask → threshold
   → contour → centroid),
4. links detections into a trajectory with a Kalman-gated tracker,
5. computes kinematics from the real per-frame time intervals, and
6. writes tracks to Parquet/CSV — optionally with a QC report and odor-port distance.

It decodes each video once and streams cropped frames to parallel workers, and it
can process a whole directory of videos concurrently.

## Install

Requires **Python ≥ 3.10** and **FFmpeg** on your `PATH` (used for decoding).

```bash
# from a clone of the repo, with uv (https://docs.astral.sh/uv/)
uv venv
uv pip install -e .
```

Optional extras: `anytrack[pose]` (SLEAP label/export bridge), `anytrack[sam]`
(Segment-Anything odor-port backend), `anytrack[docs]` (build the docs site).

## Quickstart

```bash
# one video (timing CSV defaults to <video>.csv)
anytrack run --video /data/expt/localsearch_2026-05-28.avi

# a whole directory, 3 videos at a time, with a QC report
anytrack run --video /data/expt/ --concurrency 3 --qc

# preflight only: validate inputs + print the plan, decode nothing
anytrack run --video /data/expt/ --dry-run
```

`anytrack` is a subcommand dispatcher — `anytrack <cmd>` routes to `anytrack-<cmd>`
(e.g. `anytrack run` → `anytrack-run`). Run `anytrack` with no arguments to list
commands, or `anytrack <cmd> --help` for options. The GUI is `anytrack gui`.

Key commands: `run`, `validate`, `qc`, `bg`, `roi`, `crop-export`, `bench`,
`label`, `train`, `gui`, `debug`, `arena-debug`.

## Input format

- **Video:** one file per recording; several arenas per frame is expected.
- **Timing CSV** (`<video>.csv` by default): a `frame` column plus either `dt_s`
  (seconds since the previous frame) or `timestamp_s` (absolute seconds).

## Documentation

Full documentation — installation, the pipeline explained, CLI and configuration
reference, output formats, and the GUIs — is published at:

**https://fmi-basel.github.io/gfelsenb-anytrack/**

Build it locally with `uv pip install -e '.[docs]' && mkdocs serve`.

## Development

- Tests: `uv run python -m pytest` (use `python -m pytest`, not `pytest`).
- Branches: work happens on `dev`; `main` is the released branch. See the
  [development guide](https://fmi-basel.github.io/gfelsenb-anytrack/development/).

## License

[MIT](LICENSE) © 2026 Dennis Goldschmidt, Friedrich Miescher Institute for
Biomedical Research (FMI).
