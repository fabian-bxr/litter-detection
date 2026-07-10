from __future__ import annotations

from litter_detector.tracker.kalman import KalmanBoxState, bbox_to_z, z_to_bbox
from litter_detector.tracker.types import BBox


def test_bbox_z_roundtrip_preserves_box():
    bbox = BBox(10, 20, 30, 40)
    z = bbox_to_z(bbox)
    recovered = z_to_bbox(z)
    assert recovered == bbox


def test_initial_state_matches_observation():
    bbox = BBox(10, 20, 30, 40)
    kf = KalmanBoxState(bbox)
    assert kf.bbox == bbox


def test_predict_with_no_velocity_keeps_box_stable():
    bbox = BBox(10, 20, 30, 40)
    kf = KalmanBoxState(bbox)
    # First predict moves only by velocity (zero) so the box should be ~unchanged.
    kf.predict()
    after = kf.bbox
    assert abs(after.cx - bbox.cx) < 1e-6
    assert abs(after.cy - bbox.cy) < 1e-6


def test_kalman_learns_constant_velocity():
    # Object moving +5px/frame in x, stationary in y, constant size.
    # Observation centroid at step k is (5*k + 10, 10) — the BBox top-left walks
    # by 5px while size stays 20×20.
    kf = KalmanBoxState(BBox(0, 0, 20, 20))
    for step in range(1, 11):
        kf.predict()
        kf.update(BBox(5 * step, 0, 20, 20))
    # Last observation centroid was (60, 10). One more predict with learned
    # velocity ≈ 5 should land near (65, 10).
    kf.predict()
    pred = kf.bbox
    assert abs(pred.cx - 65) < 2.0, f"expected ~65, got {pred.cx}"
    assert abs(pred.cy - 10) < 2.0, f"expected ~10, got {pred.cy}"


def test_predict_clamps_negative_scale():
    # Inject a scale-velocity that would drive predicted area negative.
    kf = KalmanBoxState(BBox(0, 0, 20, 20))
    kf.x[2] = 100.0  # tiny scale
    kf.x[6] = -1000.0  # huge negative scale-velocity
    kf.predict()
    # If clamping worked, scale stayed at 100 (velocity was zeroed) and bbox is still well-formed.
    bbox = kf.bbox
    assert bbox.w > 0 and bbox.h > 0
