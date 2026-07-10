# The pipeline

A run of `anytrack run` takes a video + timing CSV through the stages below. Steps
1–3 are the "preflight" that `--dry-run` stops after; steps 4–11 do the work.

```
resolve inputs → validate → print config
   → ROI detection → per-ROI background → detection → tracking
   → kinematics → (odor port) → (QC) → write tracks
```

## 1–3. Resolve, validate, config

The input path is resolved (a single file, or every video in a directory), each
video is paired with its timing CSV, and the inputs are validated (readable video,
decodable first frame, well-formed CSV). The **effective configuration** is printed
so every run is self-documenting. `--dry-run` stops here.

## 4. ROI (arena) detection

A model background image is built for the whole frame and **Hough circles** locate
the arenas (`roi_hough_*`, `min_radius_ratio`, `max_radius_ratio`). Detected ROIs
are numbered in reading order (`arena_01`, `arena_02`, …). Restrict a run to
specific arenas with `--roi arena_02` (repeatable/comma-separated).

## 5. Per-ROI background modelling

Each arena gets its **own** background so subtraction is local and robust. The
default `adaptive` model (`bg_model`):

1. a fast per-pixel percentile floor (`bg_model_percentile`, default p90),
2. a **track-free de-ghost** pass that lifts a fly baked into the floor at a
   long-dwell spot back to arena brightness (`bg_deghost*`), and
3. optional foreground-excluded / stationary-fill refinement.

Frames for the background are sampled either by decoding once through the same
crop→scale filter used for tracking (`--bg-sample ffmpeg`) or by reusing the
full-frame samples already taken for ROI detection (`--bg-sample reuse`, ~4× faster,
used automatically for `--bg-only`). A legacy per-pixel GMM model is available
(`bg_model=gmm`).

!!! tip "Inspect / save backgrounds"
    `--save-roi-bg` writes each arena's background as a PNG (plus a fly-mask
    overlay); `--bg-only` builds and saves backgrounds and stops before tracking.

## 6. Detection

Per arena, per frame:

1. **background subtraction** (`bgdiff_type`, default `dark` = fly darker than floor),
2. **arena-disk mask** so nothing outside the arena is ever a candidate
   (`detect_arena_mask`),
3. **threshold** (`thr_method`/`thr_fixed`) and **morphology** (`morph_open`,
   `morph_close`),
4. **contours** → area filter (`expected_fly_area_min/max`) → keep the top
   `detect_max_candidates`,
5. each candidate's **position is the contour moment-centroid**; a fitted ellipse
   supplies orientation/major/minor. (Using the moment centroid avoids a failure
   mode where `cv2.fitEllipse` returns an off-image centre on thin rim arcs.)

## 7. Tracking (linking)

A per-arena `CentroidTracker` links detections into a trajectory:

- each frame the fly is matched to the **nearest candidate** within an **acceptance
  gate** centered on the last confirmed position (`max_jump_px`),
- the gate **widens while a track is lost** so a brief fast move is re-acquired
  within a couple of frames,
- short gaps are tolerated (`miss_tolerance`).

The **reported position is the raw contour centroid**. By default the Kalman filter
is **off** (`use_kalman=false`) — on the fast turns typical of a walking fly, a
constant-velocity Kalman prediction tends to overshoot, so gating on the last
confirmed position tracks more faithfully. Set `use_kalman=true` to gate on the
Kalman prediction instead, and `kalman_smoothing=true` to additionally report the
smoothed posterior rather than the raw centroid.

## 8. Kinematics

From the trajectory and the **real per-frame time intervals** (timing CSV),
`anytrack` computes speed (`speed_mm_s`), angular speed (`ang_speed_deg_s`), and
unwrapped heading — using each arena's px→mm scale (`arena_diameter_mm`).

## 9. Odor-port detection (optional, `--detect-port`) { #odor-port-detection }

The central odor-delivery port is a low-contrast **dark circular fixture** near the
arena centre. A **circular-edge matched filter** (strongest mean outward radial
gradient) locates it within a central window (`port_max_center_frac`) and a
constant-size radius band (`port_radius_frac_min/max`), then a shared median radius
is applied across arenas (`port_shared_radius`). Detection adds
`dist_to_port_px`/`dist_to_port_mm` to the tracks and writes a
[`<stem>_rois.json`](outputs.md#roisjson) sidecar (arena geometry + port) plus a
`<stem>_ports.png` overlay. An opt-in `port_backend=sam` uses Segment-Anything.

## 10. QC (optional, `--qc`)

Generates an overlay video, distribution/kinematics/occupancy plots, a
flagged-frame montage, per-frame flags and a JSON summary, bundled into
`qc_report.html`. See [QC report](outputs.md#qc-report).

## 11. Write tracks

Tracks are written to Parquet (default) or CSV — see
[Output formats](outputs.md). `--crops` additionally exports centroid-centered
image crops for downstream pose work.

## Performance model

- **Decode once, stream:** FFmpeg decodes the video a single time and pipes raw
  grayscale arena crops through a FIFO to parallel workers
  (`roi_stream_decode`, `n_tracking_workers`); each worker pins OpenCV to one thread
  (`cv_threads_per_worker`).
- **Hardware decode:** probed and used when available (`use_hw_decode`,
  `hw_decode_backend`), with a silent software fallback.
- **Downscale & stride:** `--downscale {1,2,4}` shrinks each arena before detection
  (thresholds are auto-rescaled from `detect_params_ref_downscale`); `--stride N`
  tracks every Nth frame and interpolates the rest.
- **Batch concurrency:** a directory is processed with several videos at once
  (`--concurrency` / `batch_concurrency`, auto-derived from cores by default) with a
  live per-video progress display.
