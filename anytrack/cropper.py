"""
Dynamic centroid-centered crop extraction (for pose / SLEAP labeling).

This is distinct from the *static* per-arena crops in ``preprocess.py`` (which
feed the tracker). Here we extract a small ``size x size`` window centered on
the tracked object each frame, padding at frame edges. Because the tracker runs
on downscaled ROI sub-videos, crops are taken in a **second pass over the
original full-resolution video** using the finished centroid dataframe, so the
crops keep full resolution and can be batched across ROIs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2
import pandas as pd


def extract_crop(
    frame: np.ndarray,
    cx: float,
    cy: float,
    size: int,
    pad: str = "background",
    pad_value: Optional[float] = None,
) -> Tuple[np.ndarray, int, int, bool]:
    """Extract a ``size x size`` crop centered on ``(cx, cy)``.

    Returns ``(crop, crop_x0, crop_y0, out_of_bounds)`` where ``crop_x0/y0`` are
    the full-frame coordinates of the crop's top-left corner and
    ``out_of_bounds`` is True when any padding was needed.

    ``pad="background"`` fills the outside region with ``pad_value`` (the frame
    median if None); ``pad="edge"`` replicates the border.
    """
    h, w = frame.shape[:2]
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size

    # Source region intersected with the frame (clamped to [0, w]/[0, h] so the
    # bounds stay ordered even when the crop is entirely outside the frame).
    sx0, sy0 = min(max(0, x0), w), min(max(0, y0), h)
    sx1, sy1 = min(max(0, x1), w), min(max(0, y1), h)
    out_of_bounds = x0 < 0 or y0 < 0 or x1 > w or y1 > h

    if not out_of_bounds:
        return frame[y0:y1, x0:x1].copy(), x0, y0, False

    canvas_shape = (size, size) if frame.ndim == 2 else (size, size, frame.shape[2])
    clipped = frame[sy0:sy1, sx0:sx1]

    if pad == "edge" and clipped.size > 0:
        top, bottom = sy0 - y0, y1 - sy1
        left, right = sx0 - x0, x1 - sx1
        crop = cv2.copyMakeBorder(clipped, top, bottom, left, right, cv2.BORDER_REPLICATE)
        return crop, x0, y0, True

    # background fill (also the fallback when the crop is fully outside).
    if pad_value is None:
        pad_value = float(np.median(frame)) if frame.size else 0.0
    crop = np.full(canvas_shape, pad_value, dtype=frame.dtype)
    if clipped.size > 0:
        oy, ox = sy0 - y0, sx0 - x0
        crop[oy:oy + (sy1 - sy0), ox:ox + (sx1 - sx0)] = clipped
    return crop, x0, y0, True


class CropBatcher:
    """Accumulates crops + metadata and yields fixed-size batches.

    Used by the pose stage (Milestone B) to run one inference call over many
    ROIs/frames instead of one per crop.
    """

    def __init__(self, batch_size: int):
        self.batch_size = max(1, int(batch_size))
        self._crops: List[np.ndarray] = []
        self._meta: List[Dict[str, Any]] = []

    def add(self, crop: np.ndarray, meta: Dict[str, Any]) -> None:
        self._crops.append(crop)
        self._meta.append(meta)

    def ready(self) -> bool:
        return len(self._crops) >= self.batch_size

    def __len__(self) -> int:
        return len(self._crops)

    def pop(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Return ``(stacked_crops[N,H,W(,C)], metas)`` and clear the buffer."""
        batch = np.stack(self._crops, axis=0)
        metas = self._meta
        self._crops, self._meta = [], []
        return batch, metas


def export_crops(
    video,
    df: pd.DataFrame,
    cfg,
    out_dir: Optional[Path] = None,
    every_n: int = 1,
    roi_col: str = "roi",
    frame_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Export centroid-centered crops from the original full-res video.

    Reads the video once (sequential decode), and for every frame present in
    ``df`` (optionally sub-sampled by ``every_n``) writes one PNG per object to
    ``out_dir/{roi}/{frame:06d}.png``. Returns a manifest DataFrame (also
    written as ``out_dir/manifest.parquet``) suitable for SLEAP labeling.
    """
    size = int(getattr(cfg, "crop_size", 128))
    pad = getattr(cfg, "crop_pad_mode", "background")
    out_dir = Path(out_dir) if out_dir is not None else Path(getattr(cfg, "crop_export_dir", "") or "crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return pd.DataFrame()

    # Group needed rows by frame index.
    needed: Dict[int, List[Tuple[str, float, float]]] = {}
    for _, row in df.iterrows():
        fidx = int(row[frame_col])
        if every_n > 1 and (fidx % every_n != 0):
            continue
        needed.setdefault(fidx, []).append((str(row[roi_col]), float(row[x_col]), float(row[y_col])))

    cap = cv2.VideoCapture(str(video.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video.video_path}")

    records: List[Dict[str, Any]] = []
    fidx = 0
    max_needed = max(needed) if needed else -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in needed:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pad_value = float(np.median(gray)) if pad == "background" else None
            for roi_name, x, y in needed[fidx]:
                crop, cx0, cy0, oob = extract_crop(gray, x, y, size, pad=pad, pad_value=pad_value)
                roi_dir = out_dir / roi_name
                roi_dir.mkdir(parents=True, exist_ok=True)
                rel = f"{roi_name}/{fidx:06d}.png"
                cv2.imwrite(str(out_dir / rel), crop)
                records.append({
                    "roi": roi_name,
                    "frame": fidx,
                    "path": rel,
                    "crop_size": size,
                    "crop_x0_full": cx0,
                    "crop_y0_full": cy0,
                    "x_full": x,
                    "y_full": y,
                    "x_crop": x - cx0,
                    "y_crop": y - cy0,
                    "out_of_bounds": oob,
                })
        fidx += 1
        if fidx > max_needed:
            break
    cap.release()

    manifest = pd.DataFrame.from_records(records)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    return manifest


def _load_tracks(path: Path) -> pd.DataFrame:
    """Read a finished session's tracks table (parquet or CSV)."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: export centroid-centered crops from a finished session (A5/A7).

    Runs a second full-resolution pass over the original video using an
    existing tracks table, writing ``{roi}/{frame:06d}.png`` + a manifest for
    SLEAP labeling. Independent of pose deps.
    """
    import argparse
    from types import SimpleNamespace
    from .config import load_config

    ap = argparse.ArgumentParser(
        prog="anytrack-crop-export",
        description="Export centroid-centered crops from a finished tracking session.",
    )
    ap.add_argument("--video", required=True, type=Path, help="Original full-resolution video.")
    ap.add_argument("--tracks", required=True, type=Path,
                    help="Tracks table from the session (.parquet or .csv) with roi/frame/x/y columns.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: cfg.crop_export_dir or ./crops).")
    ap.add_argument("--every-n", type=int, default=1, help="Export every Nth frame (default 1).")
    ap.add_argument("--crop-size", type=int, default=None, help="Override crop_size (px).")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.crop_size is not None:
        cfg.crop_size = args.crop_size

    if not args.video.exists():
        ap.error(f"video not found: {args.video}")
    if not args.tracks.exists():
        ap.error(f"tracks not found: {args.tracks}")

    df = _load_tracks(args.tracks)
    # export_crops only needs the pixel source, so a minimal video object with
    # .video_path is sufficient — no timing CSV required for crop export.
    video = SimpleNamespace(video_path=args.video)

    manifest = export_crops(video, df, cfg, out_dir=args.out_dir, every_n=args.every_n)
    out_dir = args.out_dir or Path(getattr(cfg, "crop_export_dir", "") or "crops")
    print(f"exported {len(manifest)} crops -> {out_dir}")
    print(f"manifest: {out_dir / 'manifest.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
