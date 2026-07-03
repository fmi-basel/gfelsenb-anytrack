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
from .coordinates import add_roi_local_columns

@dataclass
class TrackingSession:
    cfg: AnyTrackConfig
    video: VideoAsset
    result: Optional[TrackingResult] = None
    dataframe: Optional[pd.DataFrame] = None
    pose_dataframe: Optional[pd.DataFrame] = None

    def run(self, progress_hook=None, cancel_event=None, progress_every: int = 1,
            roi_backgrounds=None) -> pd.DataFrame:
        import dataclasses
        from .staging import stage_video, unstage

        # Stage the source off slow/network storage to a local disk if warranted.
        staged_path, staged = stage_video(self.video.video_path, self.cfg)
        run_video = (
            dataclasses.replace(self.video, video_path=staged_path) if staged else self.video
        )
        try:
            if self.cfg.fast_mode:
                # Fast mode: FFmpeg preprocessing + parallel ROI tracking
                self.result = track_video_fast(
                    run_video,
                    self.cfg,
                    progress_hook=progress_hook,
                    cleanup=self.cfg.cleanup_roi_videos,
                    roi_backgrounds=roi_backgrounds,
                )
            else:
                # Legacy mode: single-pass tracking
                self.result = track_video(
                    run_video,
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
        finally:
            if staged:
                unstage(staged_path, self.cfg)

        # Keep the original asset on the result (not the transient staged copy).
        if self.result is not None:
            self.result.video = self.video

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

        # Additive ROI-local coordinates (x_roi, y_roi). Full-frame x/y stay
        # canonical so kinematics and the GUI are unaffected.
        df = add_roi_local_columns(df, self.video.rois)

        self.dataframe = df

        # Pose stage (Milestone B5): optional, runs the trained keypoint model on
        # the finished centroids. Uses the original video path (not a staged copy,
        # which may already be cleaned up). Failures are non-fatal: tracking output
        # is still returned.
        self.pose_dataframe = None
        if getattr(self.cfg, "pose_enabled", False):
            if progress_hook is not None:
                try:
                    progress_hook("pose", {"status": "start"})
                except Exception:
                    pass
            try:
                from .pose.pipeline import run_pose_stage
                self.pose_dataframe = run_pose_stage(self.video, df, self.cfg, show_progress=True)
                if progress_hook is not None:
                    progress_hook("pose_done", {"pose": self.pose_dataframe})
            except Exception as e:
                if progress_hook is not None:
                    try:
                        progress_hook("pose_error", {"exc": repr(e)})
                    except Exception:
                        pass

        if progress_hook is not None:
            try:
                progress_hook("done", {"session": self, "dataframe": df})
            except Exception:
                pass

        return df
