from __future__ import annotations

import numpy as np

from litter_detector.tracker.types import BBox


def bbox_iou(a: BBox, b: BBox) -> float:
    """IoU between two boxes. Returns 0.0 when either box has zero area."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_matrix(detections: list[BBox], predictions: list[BBox]) -> np.ndarray:
    """Pairwise IoU matrix shaped (n_detections, n_predictions)."""
    if not detections or not predictions:
        return np.zeros((len(detections), len(predictions)), dtype=np.float32)

    m = np.zeros((len(detections), len(predictions)), dtype=np.float32)
    for i, d in enumerate(detections):
        for j, p in enumerate(predictions):
            m[i, j] = bbox_iou(d, p)
    return m
