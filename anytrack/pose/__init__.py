"""Pose estimation (Milestone B): skeleton, engines, and the inference pipeline.

Heavy backends (sleap-nn / torch) are imported lazily inside their concrete
engine modules, so importing this package stays dependency-free.
"""
from .skeleton import Skeleton, DEFAULT_SKELETON, load_skeleton, get_skeleton
from .engine import PoseEngine, MockPoseEngine, heatmaps_to_keypoints, build_engine
from .labeling import (
    LabelStore, FrameLabel, Keypoint, sample_frames, seed_from_ellipse,
    parse_node_colors, resolve_node_colors, contrast_fg, DEFAULT_NODE_COLORS,
)

__all__ = [
    "Skeleton",
    "DEFAULT_SKELETON",
    "load_skeleton",
    "get_skeleton",
    "PoseEngine",
    "MockPoseEngine",
    "heatmaps_to_keypoints",
    "build_engine",
    "LabelStore",
    "FrameLabel",
    "Keypoint",
    "sample_frames",
    "seed_from_ellipse",
    "parse_node_colors",
    "resolve_node_colors",
    "contrast_fg",
    "DEFAULT_NODE_COLORS",
]
