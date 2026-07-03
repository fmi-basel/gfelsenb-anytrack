"""Pose engine interface + a dependency-free mock for testing the plumbing.

A ``PoseEngine`` maps a batch of centroid crops to per-keypoint coordinates in
**crop-local pixels** plus confidence scores. Keeping this interface small and
backend-agnostic lets the whole pose pipeline (crop → infer → merge → write) be
built and tested with :class:`MockPoseEngine`, so the real sleap-nn engine
(Milestone B4) drops in as an isolated adapter.
"""
from __future__ import annotations

from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class PoseEngine(Protocol):
    """Backend-agnostic keypoint predictor.

    ``keypoint_names`` gives the K keypoint order (must match the skeleton's
    node order). ``infer_batch`` takes ``crops`` shaped ``(N, S, S)`` (or
    ``(N, S, S, C)``) uint8 and returns ``(keypoints, scores)`` where
    ``keypoints`` is ``(N, K, 2)`` crop-local pixels and ``scores`` is
    ``(N, K)`` in ``[0, 1]``.
    """
    keypoint_names: List[str]

    def infer_batch(self, crops: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...


class MockPoseEngine:
    """Deterministic synthetic keypoints for testing (no heavy deps).

    Places each keypoint at a fixed fraction-of-crop offset from the crop
    center, so tests can assert exact crop→full→ROI coordinates. Scores are 1.
    """

    # Offsets as a fraction of crop size, relative to the crop center.
    _FRAC = {
        "head": (0.0, -0.28),
        "thorax": (0.0, 0.0),
        "abdomen_tip": (0.0, 0.28),
        "wingL": (-0.20, 0.12),
        "wingR": (0.20, 0.12),
    }

    def __init__(self, skeleton):
        self.keypoint_names = list(skeleton.nodes)

    def infer_batch(self, crops: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(crops.shape[0])
        s = int(crops.shape[1])
        k = len(self.keypoint_names)
        center = s / 2.0
        kps = np.empty((n, k, 2), dtype=np.float32)
        for j, name in enumerate(self.keypoint_names):
            fx, fy = self._FRAC.get(name, (0.0, 0.0))
            kps[:, j, 0] = center + fx * s
            kps[:, j, 1] = center + fy * s
        scores = np.ones((n, k), dtype=np.float32)
        return kps, scores
