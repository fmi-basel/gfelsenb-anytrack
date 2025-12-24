from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .models import VideoAsset, TrackingResult
from .config import AnyTrackConfig
from .tracking import track_video
from .tracking_fast import track_video_fast
from .trajectory import interpolate_missing, enforce_single_per_frame
from .kinematics import add_kinematics

@dataclass
class TrackingSession:
    cfg: AnyTrackConfig
    video: VideoAsset
    result: Optional[TrackingResult] = None
    dataframe: Optional[pd.DataFrame] = None

    def run(self, progress_hook=None, cancel_event=None, progress_every: int = 1) -> pd.DataFrame:
        try:
            if self.cfg.fast_mode:
                # Fast mode: FFmpeg preprocessing + parallel ROI tracking
                self.result = track_video_fast(
                    self.video,
                    self.cfg,
                    progress_hook=progress_hook,
                    cleanup=self.cfg.cleanup_roi_videos,
                )
            else:
                # Legacy mode: single-pass tracking
                self.result = track_video(
                    self.video,
                    self.cfg,
                    progress_hook=progress_hook,
                    cancel_event=cancel_event,
                    progress_every=progress_every,
                )
        except Exception as e:
            if progress_hook is not None:
                try:
                    progress_hook("error", {"exc": repr(e)})
                except Exception:
                    pass
            raise

        # If the user requested cancellation, return empty DataFrame and emit cancelled.
        try:
            if cancel_event is not None and cancel_event.is_set():
                if progress_hook is not None:
                    try:
                        progress_hook("cancelled", {"frame": None})
                    except Exception:
                        pass
                return pd.DataFrame()
        except Exception:
            pass

        # Post-process tracking result into dataframe and kinematics.
        df = self.result.to_dataframe()
        df = enforce_single_per_frame(df)
        df = interpolate_missing(df)

        # scaling per ROI: arena diameter in pixels => px_per_mm
        roi_centers = {r.name: (r.cx, r.cy) for r in self.video.rois}
        px_per_mm = {r.name: (2.0 * r.r) / float(self.cfg.arena_diameter_mm) for r in self.video.rois}
        df = add_kinematics(df, roi_centers=roi_centers, px_per_mm_by_roi=px_per_mm)

        self.dataframe = df

        if progress_hook is not None:
            try:
                progress_hook("done", {"session": self, "dataframe": df})
            except Exception:
                pass

        return df
