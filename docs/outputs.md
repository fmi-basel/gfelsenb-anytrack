# Output formats

## Tracks table

A run writes one tracks table. By default it is named `<video_stem>_tracks.parquet`
(or `.csv` with `output_format="csv"` / a `.csv` `--output`), next to the video or
in `output_dir`. One row per tracked frame per fly.

**Core columns**

| Column | Meaning |
|--------|---------|
| `roi` | Arena name (`arena_01`, …). |
| `track_id` | Track identity within the arena. |
| `frame` | Frame index. |
| `t_s` | Timestamp (s), from the timing CSV. |
| `x`, `y` | Centroid in full-resolution frame pixels. |
| `angle_deg`, `cv_angle_deg` | Body orientation (unwrapped / raw OpenCV ellipse angle). |
| `major`, `minor` | Fitted-ellipse axis lengths (px). |
| `area` | Contour area (px). |
| `contour_n` | Number of contour points. |

**Kinematics columns** (added from the real per-frame intervals)

| Column | Meaning |
|--------|---------|
| `speed_mm_s` | Translational speed (mm/s). |
| `ang_speed_deg_s` | Angular speed (deg/s). |
| `angle_unwrapped_deg` | Continuously unwrapped heading. |

**Conditional columns**

| Column | When | Meaning |
|--------|------|---------|
| `x_roi`, `y_roi` | always in fast mode | Centroid in arena-local coordinates. |
| `dist_to_port_px`, `dist_to_port_mm` | `--detect-port` | Distance from the fly to the detected odor port. |
| `bg_drift` | `bg_drift_correction` | Estimated additive background-brightness drift. |

Read it back with pandas:

```python
import pandas as pd
df = pd.read_parquet("clip_tracks.parquet")
```

## `_rois.json` { #roisjson }

When ROIs are detected, `anytrack` writes `<stem>_rois.json` — the arena geometry
**and** the detected odor port per arena (keyed by arena name). The `port` field is
`null` unless `--detect-port` found one.

```json
{
  "video": "clip.avi",
  "image_size": [2176, 2176],
  "arena_diameter_mm": 75.0,
  "arenas": {
    "arena_01": {
      "roi": { "cx": 522.6, "cy": 573.0, "r": 473.2 },
      "port": { "cx": 531.0, "cy": 572.0, "r": 36.6,
                "conf": 0.95, "backend": "classical", "offset_frac": 0.018 }
    }
  }
}
```

## `_ports.png`

With `--detect-port`, a companion overlay is written: the model background with each
arena's ROI (green circle + centre) and the detected port (magenta circle +
crosshair + `<arena> r=.. c=..` label).

## Per-ROI backgrounds & motion diff

- `--save-roi-bg` → `<stem>_roi_bg/<arena>.png` per arena (plus a
  `<arena>_flymask.png` overlay), the backgrounds that actually drove tracking.
- `--save-frame-diff` (automatic with `--bg-only`) → `<stem>_framediff.png`, a
  whole-video motion map (`|last − first|`; near-black means nothing moved).

## Crops

`--crops` (or `anytrack crop-export`) writes centroid-centered image crops
(`crop_size`, default 96 px) plus a manifest — the input for pose labeling/training.

## QC report

`--qc` (or `anytrack qc`) bundles everything into `qc_report.html`, alongside:

- an **overlay video** (arena circles, per-fly centroid + trail, red flag rings
  annotated with which check fired: area / jump / out-of-bounds / multi / low-contrast),
- **distribution** and **per-ROI timeseries** plots (area, candidates, contrast,
  ellipse axes, angle),
- **kinematics** timeseries (speed, angular speed; a fly→port distance panel with
  `--detect-port`),
- **occupancy heatmaps** and a **coverage raster**,
- a **flagged-frame montage**, and
- per-frame flags + a JSON summary.

Crop or follow the overlay with `--crop-roi arena_02` or `--follow arena_02:1`.
