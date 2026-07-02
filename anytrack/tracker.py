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
        self.last: Tuple[float, float] = center_xy  # last confirmed position

    def _seed(self, x: float, y: float):
        """(Re-)initialize the filter at a known position and clear the miss streak."""
        self.kf = Kalman2D(x, y)
        self.last = (x, y)
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
            seed = max(candidates, key=lambda c: c.area) if candidates else None
            self._seed(seed.x, seed.y) if seed else self._seed(self.cx, self.cy)

        if self.use_kalman and self.misses == 0:
            px, py = self.kf.predict()
        else:
            # While lost (or without Kalman), anchor the search at the last
            # confirmed position, NOT the constant-velocity extrapolation, which
            # drifts away during a miss streak and keeps gating the fly out.
            px, py = self.last

        # Widen the acceptance gate the longer we've been lost, so a fast jump
        # (fly briefly moving faster than max_jump) is re-acquired within a
        # couple of frames instead of waiting out miss_tolerance.
        eff_jump = self.max_jump * (1 + self.misses)
        chosen = greedy_assign((px, py), candidates, eff_jump)

        if chosen is None:
            self.misses += 1
            if self.misses > self.miss_tolerance:
                # Give up on continuity: re-acquire on the strongest candidate
                # (the single object in the arena), else fall back to center.
                if candidates:
                    seed = max(candidates, key=lambda c: c.area)
                    self._seed(seed.x, seed.y)
                    return seed, seed.x, seed.y
                self._seed(self.cx, self.cy)
            return None, None, None

        recovering = self.misses > 0
        self.misses = 0
        self.last = (chosen.x, chosen.y)
        if recovering:
            # Re-seed the filter on the object after a loss instead of correcting
            # the drifted filter (which would leave a bad velocity and re-lose it
            # next frame). Don't call update() on a fresh Kalman2D: cv2's
            # correct() reads statePre from a preceding predict().
            self.kf = Kalman2D(chosen.x, chosen.y)
            return chosen, chosen.x, chosen.y
        if self.use_kalman:
            x_f, y_f = self.kf.update(chosen.x, chosen.y)
        else:
            x_f, y_f = chosen.x, chosen.y
        return chosen, x_f, y_f
