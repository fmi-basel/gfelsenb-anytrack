"""
Centroid linking shared by the tracking pipelines.

Owns the single implementation of the constant-velocity Kalman filter, the
greedy nearest-neighbor assignment, and :class:`CentroidTracker`, which
encapsulates the per-ROI predict -> assign -> update -> miss/re-init loop.

The tracker is **coordinate-system agnostic**: it operates in whatever frame
the caller supplies candidates and the fallback center in (ROI-local, scaled,
or full-frame). The caller is responsible for converting the returned position
to full-frame coordinates.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import cv2

from .detector import EllipseCandidate


class Kalman2D:
    """Constant-velocity Kalman filter (state [x, y, vx, vy])."""

    def __init__(self, x: float, y: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)

    def predict(self) -> Tuple[float, float]:
        s = self.kf.predict()
        return float(s[0, 0]), float(s[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        m = np.array([[x], [y]], np.float32)
        s = self.kf.correct(m)
        return float(s[0, 0]), float(s[1, 0])


def greedy_assign(
    pred: Tuple[float, float],
    candidates: List[EllipseCandidate],
    max_jump: float,
) -> Optional[EllipseCandidate]:
    """Return the candidate closest to ``pred`` within ``max_jump``, else None."""
    if not candidates:
        return None
    px, py = pred
    best = None
    best_d = float("inf")
    for c in candidates:
        d = float(np.hypot(c.x - px, c.y - py))
        if d < best_d:
            best_d = d
            best = c
    if best is not None and best_d <= max_jump:
        return best
    return None


class CentroidTracker:
    """Per-ROI single-object tracker.

    Holds the Kalman filter, the consecutive-miss counter, and the fallback
    center used to (re-)initialize. All coordinates are in the caller's frame.
    """

    def __init__(
        self,
        center_xy: Tuple[float, float],
        max_jump: float,
        miss_tolerance: int,
        use_kalman: bool = True,
    ):
        self.cx, self.cy = center_xy
        self.max_jump = max_jump
        self.miss_tolerance = miss_tolerance
        self.use_kalman = use_kalman
        self.kf: Optional[Kalman2D] = None
        self.misses = 0

    def step(
        self, candidates: List[EllipseCandidate]
    ) -> Tuple[Optional[EllipseCandidate], Optional[float], Optional[float]]:
        """Advance one frame.

        Returns ``(chosen, x, y)`` where ``x, y`` is the (Kalman-smoothed if
        enabled) position, or ``(None, None, None)`` when no candidate is
        assigned this frame.
        """
        # Lazy init on first frame: seed at the best candidate, else the center.
        if self.kf is None:
            if candidates:
                self.kf = Kalman2D(candidates[0].x, candidates[0].y)
            else:
                self.kf = Kalman2D(self.cx, self.cy)

        px, py = self.kf.predict() if self.use_kalman else (self.cx, self.cy)
        chosen = greedy_assign((px, py), candidates, self.max_jump)

        if chosen is None:
            self.misses += 1
            if self.misses > self.miss_tolerance:
                # Lost for too long. Re-acquire on the strongest candidate (the
                # single object in the arena) rather than the ROI center: a fly
                # that lives away from center is otherwise never recovered,
                # because the centered prediction gates the real candidate out
                # every frame. Fall back to the center only when there is
                # nothing to lock onto. Adopt the seed as this frame's fix so we
                # don't also waste the re-acquisition frame.
                if candidates:
                    seed = max(candidates, key=lambda c: c.area)
                    # Seed the filter at the object. Don't call update() here:
                    # cv2's correct() reads statePre from a preceding predict(),
                    # which a freshly-built Kalman2D has not run, so it would
                    # corrupt the state. The seed IS the position this frame.
                    self.kf = Kalman2D(seed.x, seed.y)
                    self.misses = 0
                    return seed, seed.x, seed.y
                self.kf = Kalman2D(self.cx, self.cy)
                self.misses = 0
            return None, None, None

        self.misses = 0
        if self.use_kalman:
            x_f, y_f = self.kf.update(chosen.x, chosen.y)
        else:
            x_f, y_f = chosen.x, chosen.y
        return chosen, x_f, y_f
