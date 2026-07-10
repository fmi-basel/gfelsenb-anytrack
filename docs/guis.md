# Graphical tools

`anytrack` ships several Tk/ttkbootstrap GUIs. They open on your display, so run
them locally (not over a plain SSH session without X forwarding).

## `anytrack gui`

The main application. Browse videos, ROIs and tracks in a tree, preview frames on a
canvas with a frame slider, and inspect results in a table.

```bash
anytrack gui
```

!!! note
    `anytrack` with no arguments no longer launches the GUI — it lists subcommands.
    Use `anytrack gui` (or the `anytrack-gui` script).

## `anytrack debug`

Interactive per-frame **detection/tracking inspector**. Opens on one video, builds
each arena's background, tracks it, then lets you scrub frame-by-frame in two synced,
zoomable panes — each showing a chosen stage:

`raw · background · subtracted · threshold · mask · candidates · final(track)`

Live sliders (threshold, morphology, area, max-jump) re-render the current frame
instantly, and **Re-track** re-runs linking so you can see how a parameter changes
the tracker's decisions. What you see mirrors the production detector/tracker.

```bash
anytrack debug --video clip.avi
```

## `anytrack arena-debug`

A stripped-down **single-arena** inspector: pick one arena, choose a stage
(`raw / background / subtracted / threshold / mask`), and toggle colour-customizable
overlays — candidate blobs, the tracked centroid, the Kalman prediction, its
acceptance gate, and the trail. Scrub with the slider or ←/→.

```bash
anytrack arena-debug --video clip.avi --roi arena_04
```

## `anytrack label`

Keyboard-first keypoint-labeling GUI for pose training (Milestone B): samples
frames from a finished run, seeds keypoints from the tracked ellipse, and saves to a
resumable JSON store (exportable to `.slp`).

```bash
anytrack label --video clip.avi --tracks clip_tracks.parquet --labels labels.json
```

See the [Roadmap](roadmap.md) for where pose is heading.
