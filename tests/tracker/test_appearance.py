from __future__ import annotations

import numpy as np

from litter_detector.tracker.bytetrack import ByteTracker
from litter_detector.tracker.extract import mask_to_detections
from litter_detector.tracker.types import BBox, Detection


def _det(x: int, y: int, w: int, h: int, conf: float = 1.0,
         appearance: np.ndarray | None = None) -> Detection:
    return Detection(bbox=BBox(x, y, w, h), area_px=w * h,
                     confidence=conf, appearance=appearance)


def _hist(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(8 * 8 * 8).astype(np.float32)
    return v / np.linalg.norm(v)


def test_extract_attaches_appearance_when_frame_supplied():
    """`mask_to_detections` must compute appearance only when given a frame."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:15, 5:15] = 255           # one blob
    frame = np.full((40, 40, 3), 128, dtype=np.uint8)

    no_app = mask_to_detections(mask, min_area_px=10)
    with_app = mask_to_detections(mask, min_area_px=10, frame_bgr=frame)

    assert no_app[0].appearance is None
    assert with_app[0].appearance is not None
    assert with_app[0].appearance.shape == (8 * 8 * 8,)
    # L2-normalised → unit vector (within float32 tolerance).
    assert np.isclose(np.linalg.norm(with_app[0].appearance), 1.0, atol=1e-5)


def test_appearance_breaks_tie_between_two_swapped_objects():
    """Two objects of similar size that pass close to each other:
    pure-IoU matching can swap their IDs; appearance must prevent that."""
    tracker = ByteTracker(
        min_hits=2, max_age=10, det_high_thresh=0.5,
        appearance_weight=0.5,
    )
    red = _hist(seed=1)
    blue = _hist(seed=2)

    # Establish two distinct-appearance tracks at distant positions.
    for ts in range(3):
        tracker.update(
            [_det(10, 10, 20, 20, appearance=red),
             _det(100, 10, 20, 20, appearance=blue)],
            ts_ns=ts * 1_000,
        )
    assert tracker.active_track_count == 2
    red_id = next(t.id for t in tracker._tracks if np.allclose(t.appearance, red, atol=1e-4))
    blue_id = next(t.id for t in tracker._tracks if np.allclose(t.appearance, blue, atol=1e-4))

    # Same boxes, but feed the detections in REVERSED order — pure IoU would
    # bind blue_id to the red detection by accident in some assignment orders.
    # With appearance tiebreaking, identities should remain correct.
    for ts in range(3, 6):
        tracks = tracker.update(
            [_det(100, 10, 20, 20, appearance=blue),
             _det(10, 10, 20, 20, appearance=red)],
            ts_ns=ts * 1_000,
        )
    by_pos = {t.bbox.x: t.id for t in tracks}
    assert by_pos[10] == red_id
    assert by_pos[100] == blue_id


def test_appearance_weight_zero_is_equivalent_to_pure_iou():
    """Sanity: with appearance_weight=0, the appearance field is ignored."""
    t_pure = ByteTracker(min_hits=2, max_age=10, det_high_thresh=0.5,
                         appearance_weight=0.0)
    t_app = ByteTracker(min_hits=2, max_age=10, det_high_thresh=0.5,
                        appearance_weight=0.0)
    red = _hist(seed=42)
    for ts in range(4):
        a = t_pure.update([_det(10, 10, 20, 20)], ts_ns=ts * 1_000)
        b = t_app.update([_det(10, 10, 20, 20, appearance=red)], ts_ns=ts * 1_000)
    assert [t.id for t in a] == [t.id for t in b]
