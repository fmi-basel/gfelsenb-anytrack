# Configuration

Every run uses an `AnyTrackConfig`. Values come from a user config file, and many
can be overridden per-run by CLI flags.

## Where settings live

`anytrack` reads a **scalar-only `config.toml`** from your platform's user config
directory (created with defaults on first run):

| OS | Path |
|----|------|
| macOS | `~/Library/Application Support/anytrack/config.toml` |
| Linux | `~/.config/anytrack/config.toml` |
| Windows | `%LOCALAPPDATA%\anytrack\config.toml` |

Unknown keys are tolerated (and old keys migrated), so it's safe to hand-edit. CLI
flags on `anytrack run` (e.g. `--downscale`, `--workers`, `--stride`,
`--concurrency`, `--detect-port`) override the file for that run. Use `--dry-run`
to print the **effective** config before committing to a full run.

The tables below list the most useful settings with their defaults. Run
`anytrack run --dry-run` to see the complete, current set.

## ROI (arena) detection

| Key | Default | Meaning |
|-----|---------|---------|
| `roi_hough_dp` | `1.2` | Hough accumulator resolution. |
| `roi_hough_min_dist_ratio` | `3` | Min centre spacing = `min_dim / ratio`. |
| `min_radius_ratio` / `max_radius_ratio` | `0.18` / `0.30` | Arena radius bounds as a fraction of the frame's short side. |
| `arena_detection_enabled` | `true` | Detect arenas during the background build. |

## Detection

| Key | Default | Meaning |
|-----|---------|---------|
| `bgdiff_type` | `"dark"` | Fly polarity vs. floor: `dark` / `bright` / `absolute`. |
| `thr_method` / `thr_fixed` | `"fixed"` / `15` | Threshold method and fixed level. |
| `morph_open` / `morph_close` | `3` / `5` | Morphology kernel sizes. |
| `detect_arena_mask` | `true` | Mask subtraction to the arena disk (no out-of-arena candidates). |
| `expected_fly_area_min` / `max` | `10` / `1500` | Candidate area filter (px). |
| `detect_max_candidates` | `5` | Keep the N largest candidates per arena. |
| `max_centroids_per_roi` | `1` | Expected flies per arena. |
| `detect_params_ref_downscale` | `2` | Downscale the area/morph thresholds are tuned for (auto-rescaled at other downscales). |

## Tracking & kinematics

| Key | Default | Meaning |
|-----|---------|---------|
| `max_jump_px` | `40.0` | Acceptance-gate radius (widened while a track is lost). |
| `miss_tolerance` | `15` | Frames a track survives without a detection. |
| `use_kalman` | `false` | Off (default): gate on the last confirmed position. On: gate on a constant-velocity Kalman prediction (overshoots on fast turns). |
| `kalman_smoothing` | `false` | Report the Kalman-smoothed posterior instead of the raw centroid (only with `use_kalman=true`). |
| `arena_diameter_mm` | `75.0` | Arena diameter → px→mm scale for kinematics. |

## Per-ROI background model

| Key | Default | Meaning |
|-----|---------|---------|
| `bg_model` | `"adaptive"` | `adaptive` (percentile → de-ghost → fill) or `gmm` (legacy). |
| `bg_model_percentile` | `90` | Baseline percentile floor. |
| `bg_deghost` | `true` | Lift a baked-in dwelling fly back to arena brightness. |
| `bg_deghost_percentile` / `bg_deghost_margin` | `98` / `15` | De-ghost near-max and lift threshold. |
| `bg_reuse_for_roi` | `true` | Reuse full-frame samples for per-ROI backgrounds. |
| `bg_refine_stuck` | `false` | Two-pass repair for baked-in flies (opt-in). |

## Performance / fast mode

| Key | Default | Meaning |
|-----|---------|---------|
| `fast_mode` | `true` | Streaming decode-once pipeline. |
| `roi_downscale` | `2` | Arena downscale (`1`/`2`/`4`). |
| `n_tracking_workers` | `4` | Parallel workers per video. |
| `roi_stream_decode` | `true` | Decode once → FIFO raw crops (vs. temp ROI mp4s). |
| `cv_threads_per_worker` | `1` | OpenCV threads per worker. |
| `track_stride` | `1` | Track every Nth frame; interpolate the rest. |
| `use_hw_decode` / `hw_decode_backend` | `true` / `"auto"` | Hardware decode (probed). |
| `use_hw_encode` | `true` | Hardware encode where applicable. |
| `batch_concurrency` | `4` | Videos at once in batch (`0` = auto from cores, `1` = sequential). |
| `batch_core_budget` | `0` | Cores to target for batch (`0` = auto). |

## Odor-port detection

| Key | Default | Meaning |
|-----|---------|---------|
| `port_detect_enabled` | `true` | Run odor-port detection (default ON; set `false` to skip, or force per-run with `--detect-port`). |
| `port_backend` | `"classical"` | `classical` (edge matched filter) or `sam`. |
| `port_edge_min` | `4.0` | Min mean radial-gradient response to accept a port. |
| `port_max_center_frac` | `0.30` | Port centre must be within this fraction of R from the arena centre. |
| `port_radius_frac_min` / `max` | `0.04` / `0.14` | Port radius search band (fraction of R). |
| `port_shared_radius` | `true` | Enforce one shared (median) port size across arenas. |

## Output & crops

| Key | Default | Meaning |
|-----|---------|---------|
| `output_format` | `"parquet"` | `parquet` or `csv`. |
| `output_dir` | `""` | Default output directory. |
| `crop_size` | `96` | Centroid-crop size (px). |
| `crop_pad_mode` | `"background"` | Crop padding at edges. |

## QC

| Key | Default | Meaning |
|-----|---------|---------|
| `qc_overlay_downscale` | `2` | Overlay video downscale. |
| `qc_overlay_stride` | `5` | Render every Nth frame in the overlay. |
| `qc_overlay_crf` | `23` | libx264 quality (lower = better/bigger). |
| `qc_overlay_trace` | `-1` | Trail length in frames (`-1` = full). |
| `qc_min_contrast` | `10.0` | Flag low-contrast detections below this. |
| `qc_edge_pinned_frac` / `qc_low_activity_mm_s` | `0.9` / `0.05` | Edge-pinned / low-activity flags. |

## Local staging

| Key | Default | Meaning |
|-----|---------|---------|
| `stage_video_locally` | `"never"` | Copy off slow/network storage: `never` / `auto` / `always`. |
| `stage_dir` | `""` | Local scratch dir (default: platform cache). |
| `cleanup_staged` | `false` | Delete the staged copy after each run. |

!!! note "Pose settings"
    Pose (`pose_*`, `sleap_model_path`, …) drives the preview pose stage and is
    documented with the [Roadmap](roadmap.md) — it is the focus of `0.3.0`.
