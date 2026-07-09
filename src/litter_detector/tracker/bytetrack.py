"""ByteTrack-style multi-object tracker.

Two-pass association per frame (vs. SORT's single pass):

  1. **High-confidence pass**: detections with confidence >= `det_high_thresh`
     are matched against ALL tracks via IoU + Hungarian. This is the "normal"
     SORT pass.
  2. **Low-confidence pass**: detections with confidence < `det_high_thresh`
     (but still above the upstream noise floor) are matched against tracks
     that were NOT matched in pass 1. These weak detections cannot spawn new
     tracks — they exist purely to rescue an existing track when the model
     only half-saw the object this frame (motion blur, partial occlusion).

Net effect: fewer ID switches across short detection gaps without the cost
of an appearance model. Reference: https://arxiv.org/abs/2110.06864.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from litter_detector.tracker.iou import iou_matrix
from litter_detector.tracker.kalman import KalmanBoxState
from litter_detector.tracker.types import Detection, Track


class _ActiveTrack:
    """Internal lifecycle wrapper around KalmanBoxState (identical to SORT's)."""

    def __init__(self, track_id: int, detection: Detection, ts_ns: int) -> None:
        self.id = track_id
        self.kf = KalmanBoxState(detection.bbox)
        self.first_seen_ns = ts_ns
        self.last_seen_ns = ts_ns
        self.n_observations = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.last_area_px = detection.area_px
        # EMA-averaged appearance descriptor. Stays None when the detector
        # doesn't supply appearance features.
        self.appearance: np.ndarray | None = (
            detection.appearance.copy() if detection.appearance is not None else None
        )

    def update(self, detection: Detection, ts_ns: int, appearance_alpha: float = 0.9) -> None:
        self.kf.update(detection.bbox)
        self.last_seen_ns = ts_ns
        self.n_observations += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_area_px = detection.area_px
        if detection.appearance is not None:
            if self.appearance is None:
                self.appearance = detection.appearance.copy()
            else:
                # EMA: a*old + (1-a)*new, then re-normalise to keep cosine sim well-defined.
                blended = appearance_alpha * self.appearance + (1.0 - appearance_alpha) * detection.appearance
                n = float(np.linalg.norm(blended))
                self.appearance = (blended / n).astype(np.float32) if n > 0 else blended.astype(np.float32)

    def predict(self) -> None:
        self.kf.predict()
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0


def _appearance_sim_matrix(
    detections: list[Detection], tracks: list[_ActiveTrack]
) -> np.ndarray | None:
    """Cosine similarity matrix between detection and track appearance vectors.

    Returns None if any side lacks appearance features — caller falls back to
    plain IoU. Both sides are assumed L2-normalised, so cosine sim is a dot
    product.
    """
    if not detections or not tracks:
        return None
    if any(d.appearance is None for d in detections):
        return None
    if any(t.appearance is None for t in tracks):
        return None
    D = np.stack([d.appearance for d in detections]).astype(np.float32)   # type: ignore[arg-type]
    T = np.stack([t.appearance for t in tracks]).astype(np.float32)        # type: ignore[arg-type]
    return D @ T.T  # (n_det, n_track)


def _match_iou(
    detections: list[Detection],
    tracks: list[_ActiveTrack],
    iou_threshold: float,
    appearance_weight: float = 0.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One Hungarian assignment pass.

    Cost matrix: `(1 - appearance_weight) * (1 - IoU) + appearance_weight * (1 - cosine_sim)`.
    The IoU gate (`iou < iou_threshold`) still rejects matches with no spatial
    overlap regardless of how similar the appearances are — appearance is a
    tiebreaker, not a teleporter.

    Returns (matches, unmatched_dets, unmatched_tracks) with indices into the
    input lists.
    """
    if not detections or not tracks:
        return [], list(range(len(detections))), list(range(len(tracks)))

    det_boxes = [d.bbox for d in detections]
    track_boxes = [t.kf.bbox for t in tracks]
    iou = iou_matrix(det_boxes, track_boxes)

    if appearance_weight > 0.0:
        sim = _appearance_sim_matrix(detections, tracks)
    else:
        sim = None

    if sim is not None and appearance_weight > 0.0:
        cost = (1.0 - appearance_weight) * (1.0 - iou) + appearance_weight * (1.0 - sim)
    else:
        cost = 1.0 - iou

    d_idx_arr, t_idx_arr = linear_sum_assignment(cost)
    matched: list[tuple[int, int]] = []
    matched_dets: set[int] = set()
    matched_tracks: set[int] = set()
    for d_i, t_i in zip(d_idx_arr, t_idx_arr):
        # IoU gate is non-negotiable: no overlap → no match, even with perfect
        # appearance similarity. Prevents identity teleport across the frame.
        if iou[d_i, t_i] < iou_threshold:
            continue
        matched.append((int(d_i), int(t_i)))
        matched_dets.add(int(d_i))
        matched_tracks.add(int(t_i))
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    return matched, unmatched_dets, unmatched_tracks


class ByteTracker:
    """Two-pass IoU tracker (ByteTrack). API-compatible with SortTracker."""

    def __init__(
        self,
        iou_threshold: float = 0.2,
        iou_threshold_low: float = 0.2,
        det_high_thresh: float = 0.7,
        max_age: int = 75,
        min_hits: int = 3,
        appearance_weight: float = 0.0,
        appearance_alpha: float = 0.9,
    ) -> None:
        """
        Args:
            iou_threshold: minimum IoU for the first (high-conf) pass.
            iou_threshold_low: minimum IoU for the second (low-conf) pass.
                Often equal to `iou_threshold`; can be set slightly lower since
                geometry is the only signal left for these matches.
            det_high_thresh: confidence split. Detections at or above this go
                into pass 1; below go into pass 2 (and can never spawn a new
                track).
            max_age: kill a track that goes unmatched for this many frames.
            min_hits: a track must accumulate this many observations before
                it is emitted as confirmed.
            appearance_weight: blend factor for appearance similarity in the
                matching cost. 0.0 disables it entirely (pure IoU, original
                ByteTrack); 0.2–0.4 helps disambiguate nearby objects.
            appearance_alpha: EMA decay for the per-track appearance
                descriptor (high = remember the past, low = react fast).
        """
        self.iou_threshold = iou_threshold
        self.iou_threshold_low = iou_threshold_low
        self.det_high_thresh = det_high_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight
        self.appearance_alpha = appearance_alpha
        self._tracks: list[_ActiveTrack] = []
        self._next_id = 1

    def update(self, detections: list[Detection], ts_ns: int) -> list[Track]:
        # 1. Predict every current track forward one frame.
        for t in self._tracks:
            t.predict()

        # 2. Split detections by confidence.
        high_dets: list[Detection] = []
        low_dets: list[Detection] = []
        for d in detections:
            (high_dets if d.confidence >= self.det_high_thresh else low_dets).append(d)

        # 3. First pass — high-conf dets ↔ all tracks.
        matched1, unmatched_high, unmatched_track_idx = _match_iou(
            high_dets, self._tracks, self.iou_threshold,
            appearance_weight=self.appearance_weight,
        )
        for d_i, t_i in matched1:
            self._tracks[t_i].update(high_dets[d_i], ts_ns, appearance_alpha=self.appearance_alpha)

        # 4. Second pass — low-conf dets ↔ tracks still unmatched after pass 1.
        remaining_tracks = [self._tracks[i] for i in unmatched_track_idx]
        matched2, _, _ = _match_iou(
            low_dets, remaining_tracks, self.iou_threshold_low,
            appearance_weight=self.appearance_weight,
        )
        for d_i, rem_t_i in matched2:
            global_t_i = unmatched_track_idx[rem_t_i]
            self._tracks[global_t_i].update(low_dets[d_i], ts_ns, appearance_alpha=self.appearance_alpha)

        # 5. Spawn new tracks — only from unmatched HIGH-conf dets.
        # Low-conf unmatched dets are discarded (the ByteTrack invariant).
        for d_i in unmatched_high:
            self._tracks.append(_ActiveTrack(self._next_id, high_dets[d_i], ts_ns))
            self._next_id += 1

        # 6. Kill stale tracks.
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # 7. Emit confirmed tracks that were updated this frame.
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

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
