from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import cv2

from .models import VideoAsset

def load_timing_csv(path: Path) -> pd.DataFrame:
    """
    Load a timing CSV file and return a DataFrame with standardized columns:
    renaming columns and ensuring 'frame' and 'dt_s' are present.
    The CSV must contain either 'dt_s' or 'timestamp_s' column.
    Args:
        path (Path): Path to the timing CSV file.
    Returns:
        pd.DataFrame: DataFrame with columns 'frame', 't_s', and 'dt_s'.
    Raises:
        ValueError: If required columns are missing.    
    """
    df = pd.read_csv(path)
    #cols = [c.lower() for c in df.columns]
    df.rename(columns={ 'Item2.Value.Index': 'frame',
                        'Item2.Interval.TotalMilliseconds': 'dt_s',}, inplace=True)
    if "frame" not in df.columns:
        raise ValueError("Timing CSV must contain a 'frame' column.")
    if "timestamp_s" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
        df["t_s"] = df["timestamp_s"].astype(float)
        df["dt_s"] = df["t_s"].diff().fillna(0.0)
        return df[["frame", "t_s", "dt_s"]]
    if "dt_s" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
        df["dt_s"] = df["dt_s"].astype(float)
        df["t_s"] = df["dt_s"].cumsum()
        return df[["frame", "t_s", "dt_s"]]
    raise ValueError("Timing CSV must contain either 'dt_s' or 'timestamp_s' column.")

def probe_video(path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    cap.release()
    return fps, n, w, h

def load_video_asset(video_path: Path, timing_csv_path: Path, name: str = "") -> VideoAsset:
    fps, n, w, h = probe_video(video_path)
    timing = load_timing_csv(timing_csv_path)
    return VideoAsset(
        video_path=video_path,
        timing_csv_path=timing_csv_path,
        name=name,
        fps_nominal=fps,
        frame_count=n,
        width=w,
        height=h,
        timing=timing,
    )
