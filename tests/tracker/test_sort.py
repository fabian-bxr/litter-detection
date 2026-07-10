from __future__ import annotations

from litter_detector.tracker.sort import SortTracker
from litter_detector.tracker.types import BBox, Detection


def _det(x: int, y: int, w: int, h: int, area: int = 100) -> Detection:
    return Detection(bbox=BBox(x, y, w, h), area_px=area)


def test_first_frame_emits_no_confirmed_tracks_below_min_hits():
    tracker = SortTracker(min_hits=3, max_age=10)
    tracks = tracker.update([_det(10, 10, 20, 20)], ts_ns=1_000)
    # min_hits=3 means a brand-new track is still tentative on frame 1.
    assert tracks == []


def test_track_becomes_confirmed_after_min_hits():
    tracker = SortTracker(min_hits=3, max_age=10)
    for ts in range(1, 5):
        tracks = tracker.update([_det(10, 10, 20, 20)], ts_ns=ts * 1_000)
    assert len(tracks) == 1
    assert tracks[0].id == 1
    assert tracks[0].first_seen_ns == 1_000


def test_id_stable_for_object_moving_smoothly():
    tracker = SortTracker(min_hits=2, max_age=10)
    ids: set[int] = set()
    # Object drifts +5px/frame in x.
    for step in range(0, 8):
        tracks = tracker.update([_det(10 + 5 * step, 10, 20, 20)], ts_ns=step * 1_000)
        for t in tracks:
            ids.add(t.id)
    assert ids == {1}, f"expected stable single ID, got {ids}"


def test_two_distinct_objects_get_distinct_ids():
    tracker = SortTracker(min_hits=2, max_age=10)
    for step in range(0, 5):
        tracks = tracker.update(
            [_det(10, 10, 20, 20), _det(200, 200, 20, 20)],
            ts_ns=step * 1_000,
        )
    ids = sorted(t.id for t in tracks)
    assert ids == [1, 2]


def test_track_dies_after_max_age_misses():
    tracker = SortTracker(min_hits=2, max_age=3)
    # Confirm a track with 3 observations.
    for step in range(0, 3):
        tracker.update([_det(10, 10, 20, 20)], ts_ns=step * 1_000)
    assert tracker.active_track_count == 1
    # Now feed 5 frames with no detections — should exceed max_age.
    for step in range(3, 8):
        tracker.update([], ts_ns=step * 1_000)
    assert tracker.active_track_count == 0


def test_n_observations_increments_per_match():
    tracker = SortTracker(min_hits=2, max_age=10)
    for step in range(0, 5):
        tracks = tracker.update([_det(10, 10, 20, 20)], ts_ns=step * 1_000)
    assert tracks[0].n_observations == 5


def test_last_seen_ns_updates_each_match():
    tracker = SortTracker(min_hits=2, max_age=10)
    for step in range(0, 4):
        tracks = tracker.update([_det(10, 10, 20, 20)], ts_ns=step * 1_000)
    assert tracks[0].last_seen_ns == 3_000


def test_unmatched_detection_gets_new_id():
    tracker = SortTracker(min_hits=2, max_age=10)
    # Confirm object A.
    for step in range(0, 3):
        tracker.update([_det(10, 10, 20, 20)], ts_ns=step * 1_000)
    # Now feed an entirely new object far away.
    for step in range(3, 6):
        tracks = tracker.update(
            [_det(10, 10, 20, 20), _det(500, 500, 20, 20)],
            ts_ns=step * 1_000,
        )
    ids = sorted(t.id for t in tracks)
    assert ids == [1, 2]
