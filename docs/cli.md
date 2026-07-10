# CLI reference

`anytrack` is a **subcommand dispatcher**: `anytrack <cmd>` routes to the
`anytrack-<cmd>` console script. Run `anytrack` with no arguments to list commands,
and `anytrack <cmd> --help` for the authoritative, up-to-date options.

| Command | Purpose |
|---------|---------|
| [`run`](#anytrack-run) | The full tracking pipeline (the main entry point). |
| [`validate`](#anytrack-validate) | Preflight-check inputs before a run. |
| [`qc`](#anytrack-qc) | Build QC artifacts from a finished run. |
| [`bg`](#anytrack-bg) | Debug background modelling. |
| [`roi`](#anytrack-roi) | Debug ROI (arena) detection. |
| [`crop-export`](#anytrack-crop-export) | Export centroid-centered crops. |
| [`bench`](#anytrack-bench) | Benchmark tracking FPS. |
| [`label`](#anytrack-label) / [`train`](#anytrack-train) | Pose labeling / training (Milestone B). |
| [`gui`](guis.md) / [`debug`](guis.md) / [`arena-debug`](guis.md) | Graphical tools. |

---

## anytrack run

Run the full pipeline headless and write tracks. Accepts a single video **or a
directory** (batch mode).

```bash
anytrack run --video <path> [options]
```

**Inputs & output**

| Flag | Meaning |
|------|---------|
| `--video PATH` | Source video, or a directory of videos. |
| `--timing PATH` | Timing CSV (default `<video>.csv`). |
| `--output, -o PATH` | Tracks path; a file is used as-is, a directory auto-names `<stem>_tracks.parquet`. |

**Mode & performance**

| Flag | Meaning |
|------|---------|
| `--fast` / `--no-fast` | Force fast streaming mode / legacy single-pass. |
| `--downscale {1,2,4}` | Downscale each arena before detection. |
| `--stride N` | Track every Nth frame, interpolate the rest. |
| `--workers N` | Tracking workers per video. |
| `--concurrency, --jobs N` | Videos processed at once in batch mode (`1` = sequential). |
| `--roi NAME` | Restrict to arena(s) by name (repeatable/comma-separated). |
| `--bg-sample {auto,ffmpeg,reuse}` | How to sample frames for the per-ROI background. |

**Stages & extras**

| Flag | Meaning |
|------|---------|
| `--dry-run` | Preflight only: validate inputs + print config/plan, then stop. |
| `--bg-only` | Build + save backgrounds and stop before tracking. |
| `--detect-port` | Detect the odor port; add `dist_to_port_*`; write `_rois.json` + overlay. |
| `--refine-stuck` | Two-pass repair for flies baked into the background (implies `--detect-port`). |
| `--qc` / `--qc-full` | Write QC artifacts (full = stride 1, downscale 1). |
| `--qc-max-frames`, `--qc-overlay-stride`, `--crop-roi`, `--follow`, `--follow-size`, `--crop-size`, `--trace` | QC overlay controls. |
| `--crops` | Also export centroid-centered crops. |
| `--save-roi-bg`, `--save-frame-diff` | Save per-ROI backgrounds / a motion-map diff PNG. |
| `--pose`, `--pose-model`, `--pose-device`, `--pose-every-n` | Pose stage (Milestone B, preview). |

---

## anytrack validate

Preflight-check that videos (+ timing CSVs) are readable before a run; exits
non-zero if any input fails.

```bash
anytrack validate --video <file|dir> [--deep] [--strict] [--json]
```

- `--deep` also seeks to and decodes the **last** frame (catches truncated files).
- `--json` emits a machine-readable report.

---

## anytrack qc

Generate QC artifacts (overlay, plots, flags, summary → `qc_report.html`) from a
finished run.

```bash
anytrack qc --video <video> --tracks <tracks.parquet> [--out-dir DIR] [options]
```

Options mirror the `run` QC flags (`--overlay-stride`, `--qc-full`, `--crop-roi`,
`--follow`, `--trace`, `--no-overlay`, `--max-frames`, `--pose`).

---

## anytrack bg

Inspect background modelling on a video (methods, sampling, ghost-removal). Useful
for tuning `bg_*` settings.

```bash
anytrack bg <video> [--method ...] [--sampling ...] [...]
```

## anytrack roi

Inspect ROI (arena) detection given a video's background.

```bash
anytrack roi <video> [--gmm-n-samples N] [--gmm-lowp V] [...]
```

## anytrack crop-export

Export centroid-centered crops from a finished session (e.g. to prepare pose
training data).

```bash
anytrack crop-export --video <video> --tracks <tracks> [--out-dir DIR] [--every-n N] [--crop-size N]
```

## anytrack bench

Benchmark tracking FPS (and, with `--pose`, pose throughput). Writes a TOML report.

```bash
anytrack bench --video <video> [--frames N] [--runs N] [--stages] [--pose ...]
```

## anytrack label

Quick keypoint-labeling GUI for pose training (Milestone B).

```bash
anytrack label --video <video> --tracks <tracks> [--labels store.json] [--n N] [--strategy diversity]
```

## anytrack train

Train a centered-instance pose model from labels.

```bash
anytrack train --labels <store.json|.slp> [--out-dir DIR] [--epochs N] [--device auto]
```
