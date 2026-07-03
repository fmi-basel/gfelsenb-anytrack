"""Pose engine interface + a dependency-free mock for testing the plumbing.

A ``PoseEngine`` maps a batch of centroid crops to per-keypoint coordinates in
**crop-local pixels** plus confidence scores. Keeping this interface small and
backend-agnostic lets the whole pose pipeline (crop → infer → merge → write) be
built and tested with :class:`MockPoseEngine`, so the real sleap-nn engine
(Milestone B4) drops in as an isolated adapter.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .skeleton import Skeleton, get_skeleton


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


# --- shared peak extraction (used by real heatmap-based engines) -------------

def heatmaps_to_keypoints(
    heatmaps: np.ndarray,
    crop_size: int,
    method: str = "argmax",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert confidence maps to crop-local keypoints + scores.

    ``heatmaps`` is ``(N, K, h, w)``; the heatmap grid may be smaller than the
    crop (stride > 1), so peak coordinates are rescaled to crop-local pixels by
    ``crop_size / w`` (x) and ``crop_size / h`` (y), with a +0.5 pixel-center
    offset. Returns ``(keypoints[N,K,2] crop-local, scores[N,K])`` where the
    score is the peak confidence value.

    ``method="argmax"`` takes the single peak pixel; ``method="centroid"`` is a
    soft-argmax (confidence-weighted mean over the grid) for sub-pixel accuracy.
    This is pure numpy so it is fully testable without a model or torch.
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    if hm.ndim != 4:
        raise ValueError(f"heatmaps must be (N,K,h,w), got shape {hm.shape}")
    n, k, h, w = hm.shape
    sx, sy = crop_size / w, crop_size / h
    kps = np.empty((n, k, 2), dtype=np.float32)

    if method == "argmax":
        flat = hm.reshape(n, k, h * w)
        idx = flat.argmax(axis=-1)
        scores = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
        yy, xx = np.divmod(idx, w)
        kps[..., 0] = (xx + 0.5) * sx
        kps[..., 1] = (yy + 0.5) * sy
        return kps, scores.astype(np.float32)

    if method == "centroid":
        pos = np.clip(hm, 0.0, None)
        mass = pos.reshape(n, k, h * w).sum(axis=-1)               # (N,K)
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        wx = (pos * gx).reshape(n, k, h * w).sum(axis=-1)
        wy = (pos * gy).reshape(n, k, h * w).sum(axis=-1)
        safe = np.where(mass > 0, mass, 1.0)
        cx = np.where(mass > 0, wx / safe, (w - 1) / 2.0)
        cy = np.where(mass > 0, wy / safe, (h - 1) / 2.0)
        kps[..., 0] = (cx + 0.5) * sx
        kps[..., 1] = (cy + 0.5) * sy
        scores = hm.reshape(n, k, h * w).max(axis=-1)
        return kps, scores.astype(np.float32)

    raise ValueError(f"unknown peak method {method!r} (use 'argmax' or 'centroid')")


# --- engine factory ----------------------------------------------------------

def build_engine(cfg, skeleton: Optional[Skeleton] = None) -> PoseEngine:
    """Resolve the configured :class:`PoseEngine`.

    Falls back to :class:`MockPoseEngine` when no trained model is available
    (``sleap_model_path`` empty) or the backend is ``"mock"``, so the pipeline
    always has a working engine. A ``"sleap-nn"`` backend with a model path
    lazily constructs :class:`~anytrack.pose.engine_sleap.SleapNNEngine`
    (raising a clear error if torch/sleap-nn are not installed).
    """
    skeleton = skeleton or get_skeleton(cfg)
    backend = str(getattr(cfg, "pose_backend", "") or "").lower()
    model_path = str(getattr(cfg, "sleap_model_path", "") or "")

    if backend in ("", "mock") or not model_path:
        return MockPoseEngine(skeleton)
    if backend in ("sleap-nn", "sleap_nn", "sleapnn"):
        from .engine_sleap import SleapNNEngine
        return SleapNNEngine(model_path, skeleton,
                             device=str(getattr(cfg, "pose_device", "auto") or "auto"))
    raise ValueError(f"unknown pose_backend {backend!r} (use 'sleap-nn' or 'mock')")
