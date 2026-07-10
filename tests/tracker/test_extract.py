from __future__ import annotations

import numpy as np

from litter_detector.tracker.extract import clean_mask, mask_to_detections


def test_empty_mask_returns_no_detections():
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert mask_to_detections(mask) == []


def test_two_disjoint_blobs_yields_two_detections():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255   # 20x20 blob
    mask[60:90, 60:80] = 255   # 30x20 blob
    dets = mask_to_detections(mask, min_area_px=10)
    assert len(dets) == 2
    # sorted by area descending — the 30x20 = 600 should come before 20x20 = 400
    assert dets[0].area_px == 600
    assert dets[1].area_px == 400


def test_min_area_filter_drops_small_blobs():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:12, 10:12] = 255   # 2x2 = 4 pixels — should be filtered
    mask[60:90, 60:80] = 255   # 30x20 = 600 — should pass
    dets = mask_to_detections(mask, min_area_px=50)
    assert len(dets) == 1
    assert dets[0].area_px == 600


def test_bbox_matches_blob_extent():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    dets = mask_to_detections(mask, min_area_px=1)
    assert len(dets) == 1
    bbox = dets[0].bbox
    assert (bbox.x, bbox.y, bbox.w, bbox.h) == (10, 10, 20, 20)


def test_rejects_non_2d_mask():
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        mask_to_detections(mask)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 3D mask")


def test_detection_confidence_defaults_to_one_without_probs():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    dets = mask_to_detections(mask, min_area_px=10)
    assert len(dets) == 1
    assert dets[0].confidence == 1.0


def test_detection_confidence_is_mean_prob_over_blob():
    mask = np.zeros((50, 50), dtype=np.uint8)
    probs = np.zeros((50, 50), dtype=np.float32)
    mask[10:20, 10:20] = 255   # 100-px blob
    probs[10:20, 10:20] = 0.8  # uniform 0.8 inside the blob
    dets = mask_to_detections(mask, min_area_px=10, probs=probs)
    assert len(dets) == 1
    assert abs(dets[0].confidence - 0.8) < 1e-5


def test_min_confidence_drops_low_prob_blobs():
    mask = np.zeros((50, 50), dtype=np.uint8)
    probs = np.zeros((50, 50), dtype=np.float32)
    mask[10:20, 10:20] = 255   # high-conf blob
    probs[10:20, 10:20] = 0.9
    mask[30:40, 30:40] = 255   # low-conf blob (borderline noise)
    probs[30:40, 30:40] = 0.55
    dets = mask_to_detections(mask, min_area_px=10, probs=probs, min_confidence=0.7)
    assert len(dets) == 1
    assert dets[0].confidence > 0.7


def test_mask_probs_shape_mismatch_raises():
    mask = np.zeros((50, 50), dtype=np.uint8)
    probs = np.zeros((40, 40), dtype=np.float32)
    try:
        mask_to_detections(mask, probs=probs)
    except ValueError:
        return
    raise AssertionError("expected ValueError for shape mismatch")


def test_clean_mask_no_op_when_kernel_zero():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    out = clean_mask(mask, erode_kernel=0)
    assert np.array_equal(out, mask)


def test_clean_mask_erosion_splits_thin_bridge():
    # Two 10x10 squares connected by a 1-px-wide bridge — connected components
    # would treat this as one blob without erosion. A 3x3 erosion eats the bridge.
    mask = np.zeros((30, 60), dtype=np.uint8)
    mask[10:20, 5:15] = 255
    mask[10:20, 45:55] = 255
    mask[14:15, 15:45] = 255   # 1-px bridge

    before = mask_to_detections(mask, min_area_px=10)
    assert len(before) == 1   # bridge makes them one blob

    eroded = clean_mask(mask, erode_kernel=3)
    after = mask_to_detections(eroded, min_area_px=10)
    assert len(after) == 2   # bridge removed → two blobs
