from __future__ import annotations

import numpy as np

from litter_detector.tracker.iou import bbox_iou, iou_matrix
from litter_detector.tracker.types import BBox


def test_iou_identical_boxes_is_one():
    a = BBox(0, 0, 10, 10)
    assert bbox_iou(a, a) == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = BBox(0, 0, 10, 10)
    b = BBox(20, 20, 10, 10)
    assert bbox_iou(a, b) == 0.0


def test_iou_half_overlap():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 0, 10, 10)  # shifted right by 5 → 50% horizontal overlap
    # intersection = 5*10=50, union = 100+100-50=150 → 50/150 = 1/3
    assert bbox_iou(a, b) == 50 / 150


def test_iou_zero_area_box_returns_zero():
    a = BBox(0, 0, 0, 0)
    b = BBox(0, 0, 10, 10)
    assert bbox_iou(a, b) == 0.0


def test_iou_matrix_shape_with_empties():
    assert iou_matrix([], []).shape == (0, 0)
    assert iou_matrix([BBox(0, 0, 1, 1)], []).shape == (1, 0)
    assert iou_matrix([], [BBox(0, 0, 1, 1)]).shape == (0, 1)


def test_iou_matrix_populates_correctly():
    dets = [BBox(0, 0, 10, 10), BBox(100, 100, 10, 10)]
    preds = [BBox(0, 0, 10, 10), BBox(100, 100, 10, 10)]
    m = iou_matrix(dets, preds)
    assert np.allclose(m, np.eye(2, dtype=np.float32))
