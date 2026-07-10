# Changelog

All notable changes to **anytrack** are documented here. The project is in **active
development** and follows [Semantic Versioning](https://semver.org). The `0.x` line
signals an evolving API/CLI — expect changes between minor versions until `1.0.0`.

## [Unreleased]

### Changed
- **Kalman filtering is now OFF by default** (`use_kalman=false`). Linking gates on
  the last confirmed position (nearest candidate within `max_jump_px`, widened while
  a track is lost) and reports the raw contour centroid. A constant-velocity Kalman
  prediction overshoots on the fast turns of a walking fly, so it is now opt-in
  (`use_kalman=true`); this makes the shipped default match the validated setup.

## [0.2.1] — 2026-07-10

Documentation and packaging polish. No functional changes to the tracking pipeline.

### Added
- MIT `LICENSE` and license metadata.
- A comprehensive documentation site (MkDocs Material), published to GitHub Pages
  at <https://fmi-basel.github.io/gfelsenb-anytrack/> — installation, quickstart,
  the pipeline explained, CLI and configuration reference, output formats, GUIs,
  and a roadmap.

### Changed
- Rewrote the README (previously a stale "skeleton" stub) for the `0.2.x` line.

### Fixed
- Stopped tracking the stale `anytrack.egg-info/` build artifact.
- Removed an unreferenced scratch script (`test_gmm_bg.py`) that shipped inside the
  installed package.

[0.2.1]: https://github.com/fmi-basel/gfelsenb-anytrack/releases/tag/v0.2.1

## [0.2.0] — 2026-07-10

First tagged release. The classical, ROI-based centroid-tracking pipeline is
feature-complete and validated on real local-search assays. **Pose estimation
(SLEAP) is the focus of the next release, `0.3.0`.**

### Pipeline
- ROI (arena) detection → per-ROI background modelling → detection (background
  subtraction, **arena-disk masking**, threshold/morphology, contour + **moment
  centroid**) → **Kalman-gated centroid tracking** → kinematics → Parquet/CSV.
- **Fast path**: decode-once FFmpeg → FIFO raw-frame streaming with multiprocessing
  workers, hardware decode, and cross-video **batch concurrency** with a live
  multi-line progress display (total ETA + per-thread per-stage elapsed/ETA).

### Odor-port detection (`--detect-port`)
- Circular-edge **matched-filter** detector using central + constant-size priors,
  validated against hand-drawn ground truth (centre/radius error ~3 px).
- Adds `dist_to_port_px` / `dist_to_port_mm` columns, and writes a combined
  `<stem>_rois.json` (arena ROI geometry **and** the detected port per arena) plus
  a `<stem>_ports.png` overlay.

### QC
- Per-run HTML report: overlays (arena circles, centroids/trails, annotated
  red-flag rings), distributions, per-ROI + kinematics timeseries, occupancy
  heatmaps, coverage raster, flagged-frame montage, and a JSON summary.

### CLI & GUIs
- `anytrack` subcommand dispatcher; `anytrack-run` (with `--dry-run`,
  `--concurrency/--jobs`), `-validate`, `-bg`, `-roi`, `-qc`, `-bench`,
  `-crop-export`; GUIs `anytrack-gui`, `-debug`, `-arena-debug`, `-label`.

### Robustness
- Arena-disk masking of background subtraction (no candidates outside the arena).
- Moment-centroid candidate position (fixes off-image `fitEllipse` centres on rim
  arcs); tracker reports the raw contour centroid (Kalman used for gating).
- ffmpeg/ffprobe launched with detached stdin (no more terminal corruption /
  `stty sane`).

### Notes
- Requires Python ≥ 3.10. Optional extras `[pose]` (sleap-io) and `[sam]`
  (Segment-Anything port backend) are **not** needed for the classical pipeline.
- Test suite: `uv run python -m pytest` (201 passing).

[0.2.0]: https://github.com/fmi-basel/gfelsenb-anytrack/releases/tag/v0.2.0
