"""Pose estimation (Milestone B): skeleton, engines, and the inference pipeline.

Heavy backends (sleap-nn / torch) are imported lazily inside their concrete
engine modules, so importing this package stays dependency-free.
"""
from .skeleton import Skeleton, DEFAULT_SKELETON, load_skeleton, get_skeleton
from .engine import PoseEngine, MockPoseEngine

__all__ = [
    "Skeleton",
    "DEFAULT_SKELETON",
    "load_skeleton",
    "get_skeleton",
    "PoseEngine",
    "MockPoseEngine",
]
