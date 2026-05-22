"""Constant-velocity Kalman bounding-box state, following the SORT formulation.

State x (7-dim): [u, v, s, r, u', v', s']
    u, v   — bbox centroid pixel coords
    s      — bbox scale (area = w*h)
    r      — bbox aspect ratio (w/h), assumed constant (no derivative)
    u', v', s' — velocities

Measurement z (4-dim): [u, v, s, r]
"""

from __future__ import annotations

import math

import numpy as np

from litter_detector.tracker.types import BBox

# 7x7 state-transition: identity + velocity-coupling for u, v, s.
_F: np.ndarray = np.eye(7, dtype=np.float64)
_F[0, 4] = 1.0
_F[1, 5] = 1.0
_F[2, 6] = 1.0

# 4x7 measurement matrix: observe [u, v, s, r] from the state.
_H: np.ndarray = np.zeros((4, 7), dtype=np.float64)
_H[0, 0] = 1.0
_H[1, 1] = 1.0
_H[2, 2] = 1.0
_H[3, 3] = 1.0

# Measurement noise — area is noisier than centroid in mask-derived boxes.
_R: np.ndarray = np.diag([1.0, 1.0, 10.0, 10.0])

# Process noise — small for positions/scale, tiny for aspect-ratio drift, smaller for velocities.
_Q: np.ndarray = np.eye(7, dtype=np.float64)
_Q[4:, 4:] *= 0.01
_Q[6, 6] *= 0.01


def bbox_to_z(bbox: BBox) -> np.ndarray:
    w = float(bbox.w)
    h = float(bbox.h)
    s = w * h
    r = w / h if h > 0 else 1.0
    return np.array([bbox.cx, bbox.cy, s, r], dtype=np.float64)


def z_to_bbox(z: np.ndarray) -> BBox:
    cx = float(z[0])
    cy = float(z[1])
    s = max(float(z[2]), 1.0)
    r = max(float(z[3]), 1e-3)
    w = math.sqrt(s * r)
    h = math.sqrt(s / r)
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return BBox(x=x, y=y, w=max(int(round(w)), 1), h=max(int(round(h)), 1))


class KalmanBoxState:
    """Per-track Kalman state. SortTracker owns lifecycle; this owns just the math."""

    def __init__(self, bbox: BBox) -> None:
        self.x: np.ndarray = np.zeros(7, dtype=np.float64)
        self.x[:4] = bbox_to_z(bbox)
        # Initial covariance: large for unobserved velocities so first updates dominate.
        self.P: np.ndarray = np.eye(7, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0

    def predict(self) -> None:
        # Clamp area-velocity so predicted area can't go non-positive.
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q

    def update(self, bbox: BBox) -> None:
        z = bbox_to_z(bbox)
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ _H) @ self.P

    @property
    def bbox(self) -> BBox:
        return z_to_bbox(self.x[:4])
