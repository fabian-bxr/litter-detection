from __future__ import annotations

from litter_detector.tracker.bytetrack import ByteTracker
from litter_detector.tracker.types import BBox, Detection


def _det(x: int, y: int, w: int, h: int, conf: float = 1.0, area: int = 100) -> Detection:
    return Detection(bbox=BBox(x, y, w, h), area_px=area, confidence=conf)


def test_high_conf_detection_spawns_track():
    tracker = ByteTracker(min_hits=2, det_high_thresh=0.7)
    for ts in range(4):
        tracks = tracker.update([_det(10, 10, 20, 20, conf=0.9)], ts_ns=ts * 1_000)
    assert len(tracks) == 1
    assert tracks[0].id == 1


def test_low_conf_alone_does_not_spawn_track():
    tracker = ByteTracker(min_hits=2, det_high_thresh=0.7)
    # Feed only low-confidence detections — no track should ever spawn.
    for ts in range(8):
        tracks = tracker.update([_det(10, 10, 20, 20, conf=0.5)], ts_ns=ts * 1_000)
    assert tracker.active_track_count == 0
    assert tracks == []


def test_low_conf_rescues_existing_track():
    """A confirmed track should survive on low-conf detections — that is the
    whole point of ByteTrack vs. SORT."""
    tracker = ByteTracker(min_hits=2, max_age=10, det_high_thresh=0.7)
    # Establish a confirmed track with high-conf detections.
    for ts in range(3):
        tracker.update([_det(10, 10, 20, 20, conf=0.9)], ts_ns=ts * 1_000)
    assert tracker.active_track_count == 1
    track_id = tracker._tracks[0].id

    # Now switch to low-conf detections at the same place — track must stay
    # alive AND keep its ID.
    for ts in range(3, 8):
        tracks = tracker.update([_det(10, 10, 20, 20, conf=0.5)], ts_ns=ts * 1_000)

    assert tracker.active_track_count == 1
    assert tracker._tracks[0].id == track_id
    assert len(tracks) == 1
    assert tracks[0].id == track_id


def test_id_stable_for_smooth_motion_with_mixed_confidence():
    tracker = ByteTracker(min_hits=2, max_age=10, det_high_thresh=0.7)
    confs = [0.95, 0.9, 0.55, 0.95, 0.6, 0.5, 0.9, 0.85]
    ids: set[int] = set()
    for step, c in enumerate(confs):
        tracks = tracker.update(
            [_det(10 + 5 * step, 10, 20, 20, conf=c)],
            ts_ns=step * 1_000,
        )
        for t in tracks:
            ids.add(t.id)
    assert ids == {1}, f"expected single stable ID, got {ids}"


def test_two_distinct_objects_get_distinct_ids():
    tracker = ByteTracker(min_hits=2, max_age=10, det_high_thresh=0.7)
    for step in range(5):
        tracks = tracker.update(
            [_det(10, 10, 20, 20, conf=0.9),
             _det(200, 200, 20, 20, conf=0.9)],
            ts_ns=step * 1_000,
        )
    ids = sorted(t.id for t in tracks)
    assert ids == [1, 2]


def test_track_dies_after_max_age_misses():
    tracker = ByteTracker(min_hits=2, max_age=3, det_high_thresh=0.7)
    for step in range(3):
        tracker.update([_det(10, 10, 20, 20, conf=0.9)], ts_ns=step * 1_000)
    assert tracker.active_track_count == 1
    for step in range(3, 8):
        tracker.update([], ts_ns=step * 1_000)
    assert tracker.active_track_count == 0


def test_low_conf_unmatched_detections_are_discarded():
    """A spurious low-conf detection in empty space must not create a track."""
    tracker = ByteTracker(min_hits=2, max_age=10, det_high_thresh=0.7)
    # Establish a real track far away.
    for ts in range(3):
        tracker.update([_det(10, 10, 20, 20, conf=0.9)], ts_ns=ts * 1_000)
    assert tracker.active_track_count == 1
    # Now feed the real track + a spurious low-conf "ghost" detection elsewhere.
    for ts in range(3, 6):
        tracker.update(
            [_det(10, 10, 20, 20, conf=0.9),
             _det(500, 500, 20, 20, conf=0.5)],
            ts_ns=ts * 1_000,
        )
    # Only the original high-conf track exists; the ghost was dropped.
    assert tracker.active_track_count == 1
