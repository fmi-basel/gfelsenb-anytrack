# Quickstart

## Input you need

- **A video** of one or more circular arenas filmed from above (`.avi`, `.mp4`,
  `.mov`, `.mkv`).
- **A timing CSV** describing each frame's time. By default `anytrack` looks for
  `<video>.csv` next to the video. It must contain a `frame` column plus **either**:
    - `dt_s` — seconds elapsed since the previous frame, **or**
    - `timestamp_s` — absolute seconds.

    Kinematics use these real intervals, so speeds are correct even with a variable
    frame rate.

## Track one video

```bash
anytrack run --video /data/expt/localsearch_2026-05-28.avi
```

This detects arenas, models backgrounds, tracks each fly, and writes
`localsearch_2026-05-28_tracks.parquet` next to the video (see
[Output formats](outputs.md)).

Point `--timing` at a differently-named CSV, and `--output` at a file or directory:

```bash
anytrack run --video clip.avi --timing clip_times.csv --output /results/clip.parquet
```

## Preflight first (recommended)

Validate inputs and print the effective config + run plan without decoding anything:

```bash
anytrack run --video /data/expt/ --dry-run
```

Or validate inputs on their own (exits non-zero on any bad file):

```bash
anytrack validate --video /data/expt/          # a whole folder
anytrack validate --video clip.avi --deep       # also decode the last frame
```

## Process a whole folder

Passing a directory to `--video` runs every video in it. Control how many run at
once with `--concurrency`:

```bash
anytrack run --video /data/expt/ --concurrency 3
```

## Add a QC report

```bash
anytrack run --video clip.avi --qc
```

This writes an overlay video, diagnostic plots, per-frame flags, and a bundled
`qc_report.html`. See [QC report](outputs.md#qc-report).

## Add odor-port detection

For odor-navigation assays, detect the central delivery port per arena and add
fly→port distance columns:

```bash
anytrack run --video clip.avi --detect-port
```

You can run **only** the port/background stage (no tracking) to check port
placement quickly:

```bash
anytrack run --video /data/expt/ --bg-only --detect-port
```

## Speed vs. accuracy knobs

```bash
# downscale arenas 4×, track every 2nd frame (interpolate the rest), 6 workers
anytrack run --video clip.avi --downscale 4 --stride 2 --workers 6
```

See [Configuration](configuration.md) for what these do and their defaults.

## Common recipes

| Goal | Command |
|------|---------|
| One video, defaults | `anytrack run --video clip.avi` |
| Folder, 3 at a time, with QC | `anytrack run --video /data/ --concurrency 3 --qc` |
| Inputs check only | `anytrack run --video /data/ --dry-run` |
| Backgrounds + ports only | `anytrack run --video /data/ --bg-only --detect-port` |
| One arena only | `anytrack run --video clip.avi --roi arena_02` |
| Export centroid crops | `anytrack run --video clip.avi --crops` |
