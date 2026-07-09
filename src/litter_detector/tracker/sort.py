"""SORT-style multi-object tracker.

Per frame: predict all tracks, associate detections via IoU + Hungarian, update
matched tracks, spawn new tentative tracks for unmatched detections, kill stale
tracks. Track IDs are stable across frames as long as the object is matched
within `max_age` frames.

This implementation deliberately omits an appearance model (no DeepSORT). For
re-identifying the same physical object across long disappearances, you'd need
either re-ID embeddings or world-frame coordinates from odometry — see
docs/tracking.md for the multi-agent extension path.
"""

from __future__ import annotations

from scipy.optimize import linear_sum_assignment

from litter_detector.tracker.iou import iou_matrix
from litter_detector.tracker.kalman import KalmanBoxState
from litter_detector.tracker.types import Detection, Track


class _ActiveTrack:
    """Internal lifecycle wrapper around KalmanBoxState."""

    def __init__(self, track_id: int, detection: Detection, ts_ns: int) -> None:
        self.id = track_id
        self.kf = KalmanBoxState(detection.bbox)
        self.first_seen_ns = ts_ns
        self.last_seen_ns = ts_ns
        self.n_observations = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.last_area_px = detection.area_px

    def update(self, detection: Detection, ts_ns: int) -> None:
        self.kf.update(detection.bbox)
        self.last_seen_ns = ts_ns
        self.n_observations += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_area_px = detection.area_px

    def predict(self) -> None:
        self.kf.predict()
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0


class SortTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        """
        Args:
            iou_threshold: minimum IoU for a (detection, prediction) pair to count as a match.
            max_age: kill a track that goes unmatched for this many frames.
            min_hits: a track must accumulate this many observations before it's emitted as confirmed.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_ActiveTrack] = []
        self._next_id = 1

    def update(self, detections: list[Detection], ts_ns: int) -> list[Track]:
        # 1. Predict every current track forward one frame.
        for t in self._tracks:
            t.predict()

        # 2. Associate detections to predicted tracks by IoU + Hungarian.
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # 3. Update matched tracks; spawn new tentatives; keep coasting tracks alive until max_age.
        for det_idx, track_idx in matched:
            self._tracks[track_idx].update(detections[det_idx], ts_ns)

        for det_idx in unmatched_dets:
            self._tracks.append(_ActiveTrack(self._next_id, detections[det_idx], ts_ns))
            self._next_id += 1

        # 4. Kill stale tracks.
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # 5. Return confirmed tracks that were updated this frame.
        confirmed: list[Track] = []
        for t in self._tracks:
            if t.time_since_update == 0 and t.n_observations >= self.min_hits:
                bbox = t.kf.bbox
                confirmed.append(
                    Track(
                        id=t.id,
                        bbox=bbox,
                        area_px=t.last_area_px,
                        first_seen_ns=t.first_seen_ns,
                        last_seen_ns=t.last_seen_ns,
                        n_observations=t.n_observations,
                    )
                )
        return confirmed

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self._tracks)))

        det_boxes = [d.bbox for d in detections]
        track_boxes = [t.kf.bbox for t in self._tracks]
        iou = iou_matrix(det_boxes, track_boxes)

        # Hungarian wants a cost matrix; we want to maximise IoU, so negate.
        det_idx_arr, track_idx_arr = linear_sum_assignment(-iou)

        matched: list[tuple[int, int]] = []
        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        for d_i, t_i in zip(det_idx_arr, track_idx_arr):
            if iou[d_i, t_i] < self.iou_threshold:
                continue
            matched.append((int(d_i), int(t_i)))
            matched_dets.add(int(d_i))
            matched_tracks.add(int(t_i))

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(self._tracks)) if i not in matched_tracks]
        return matched, unmatched_dets, unmatched_tracks

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
