from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

@dataclass
class CircleROI:
    name: str
    cx: float
    cy: float
    r: float
    # optional: expected number of flies in this ROI
    n_targets: int = 1

    def bbox(self) -> Tuple[int, int, int, int]:
        x0 = int(max(0, self.cx - self.r))
        y0 = int(max(0, self.cy - self.r))
        x1 = int(self.cx + self.r)
        y1 = int(self.cy + self.r)
        return x0, y0, x1, y1

@dataclass
class VideoAsset:
    video_path: Path
    timing_csv_path: Path
    name: str = ""
    fps_nominal: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    timing: Optional[pd.DataFrame] = None  # frame timing table
    rois: List[CircleROI] = field(default_factory=list)

    def label(self) -> str:
        return self.name or self.video_path.name

@dataclass
class EllipseObservation:
    frame: int
    t_s: float
    x: float
    y: float
    angle_deg: float
    cv_angle_deg: float
    major: float
    minor: float
    area: float
    contour_n: int
    # Additive QC diagnostics (Tier-2). Default to a single clean detection so
    # existing constructors and the reference trajectory stay valid.
    n_candidates: int = 1
    contrast: float = float("nan")
    bg_drift: float = float("nan")  # background brightness drift (only when bg_drift_correction on)

@dataclass
class FlyTrack:
    roi_name: str
    track_id: int
    observations: List[EllipseObservation] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.observations:
            return pd.DataFrame()
        rows = [obs.__dict__ for obs in self.observations]
        return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)

@dataclass
class TrackingResult:
    video: VideoAsset
    tracks: List[FlyTrack]
    background: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        dfs = []
        for tr in self.tracks:
            df = tr.to_dataframe()
            if df.empty:
                continue
            df.insert(0, "roi", tr.roi_name)
            df.insert(1, "track_id", tr.track_id)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def as_hierarchy(self) -> Dict[str, Any]:
        # Convenient structure for GUI tree
        return {
            "video": self.video.label(),
            "rois": {roi.name: {"tracks": []} for roi in self.video.rois},
        }
