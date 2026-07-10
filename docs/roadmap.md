# Roadmap

`anytrack` is under active development. Versions follow
[Semantic Versioning](https://semver.org); the `0.x` line means the API/CLI may
change between minor versions until `1.0.0`.

## 0.2.x — classical tracking pipeline *(current)*

Feature-complete and validated:

- ROI detection → per-ROI background modelling → detection → Kalman-gated tracking
  → kinematics → Parquet/CSV.
- Fast decode-once streaming path, hardware decode, and cross-video batch
  concurrency with a live progress display.
- Odor-port detection (circular-edge matched filter) + fly→port distance.
- QC reports, input validation, and interactive debug GUIs.

Patch releases (`0.2.1`, …) carry bug fixes and refinements.

## 0.3.0 — pose estimation *(next)*

The focus of the next minor release is **per-fly keypoint pose estimation** via
SLEAP, turning centroid tracks into body-part trajectories (head, thorax,
abdomen tip, wings). The scaffolding is already in place and will be finished and
integrated end-to-end:

- **Label** — `anytrack label` samples frames and seeds keypoints from the tracked
  ellipse; saves a resumable store, exportable to `.slp`.
- **Train** — `anytrack train` fits a centered-instance model
  (`sleap-nn` + `torch`).
- **Infer** — `anytrack run --pose` runs keypoint inference on tracked crops
  (`pose_*` settings: `pose_backend`, `sleap_model_path`, `pose_device`,
  `pose_every_n`, `pose_batch_size`, `pose_conf_min`, …).

The pose backend is an optional dependency (see [Installation](installation.md)),
so the classical pipeline stays dependency-light.

## Beyond

`1.0.0` will mark a stabilized API/CLI once pose is integrated and the interfaces
have settled. Multi-fly-per-arena assignment and additional assay readouts are
candidates for later `0.x` releases.

*Have a request? Open an issue on
[GitHub](https://github.com/fmi-basel/gfelsenb-anytrack/issues).*
