# Object Tracking & Registry

This document covers the tracking layer that sits on top of the segmentation
detector — frame-to-frame association, stable IDs, persistent registry, and the
extension path toward a multi-agent system.

## Why this exists

The CNN detector publishes a binary mask per frame (`litter/mask`) and JSON
coverage stats (`litter/detection`). That tells you *whether* litter is in
view, but not *which* piece of litter is which between frames. Without an ID
layer, the robot can't:

- count distinct objects over time,
- avoid double-collecting the same trash bag from two angles,
- record per-object timestamps for "first seen / last seen",
- hand a clean object handle to a future planner.

The tracker module fills that gap.

## Component overview

```
src/litter_detector/tracker/
├── types.py       BBox / Detection / Track dataclasses
├── extract.py     mask → list[Detection] via cv2.connectedComponentsWithStats
├── iou.py         pairwise IoU + IoU matrix
├── kalman.py      Constant-velocity Kalman state for a single bbox (SORT formulation)
├── sort.py        SortTracker: predict + Hungarian assoc + lifecycle
└── registry.py    SQLite persistence of confirmed tracks
```

## Data flow per frame

```
                                   +---------------------+
camera/frame  ───►  detector  ───► |  segmentation mask  | ──┐
                                   +---------------------+   │
                                                             ▼
                                              +---------------------------+
                                              | mask_to_detections        |
                                              | (connectedComponentsStats)|
                                              +---------------------------+
                                                             │ list[Detection]
                                                             ▼
+-------------+     predict     +---------------------+     associate (IoU + Hungarian)
| KalmanState |  ────────────► | SortTracker.update  |  ──────────────────────────────┐
+-------------+                +---------------------+                                │
        ▲                              │                                              │
        │ update on match              │ list[Track]                                  ▼
        │                              ▼                                    +-----------------+
+-------------+               +-----------------+                            | spawn / kill /  |
| _ActiveTrack|◄──────────────| upsert_all()    | ──────────►  SQLite        | promote tracks  |
| (lifecycle) |               | ObjectRegistry  |              objects.db    +-----------------+
+-------------+               +-----------------+
                                       │
                                       ▼
                              Zenoh: litter/tracked  (JSON)
```

## Pipeline (in `detector/main.py`)

The `LitterDetector._process` method now runs:

1. **decode** — JPEG bytes → BGR frame.
2. **preprocess** — resize/normalize to model input.
3. **inference** — ONNX or PyTorch forward pass → logits.
4. **postprocess** — sigmoid → binary mask at original resolution.
5. **track** — extract bboxes from mask, predict + associate, update registry, draw IDs.
6. **publish** — frame, mask, overlay (with IDs drawn), detection stats, *and the new tracked topic*.

## Track lifecycle

```
                  detection without match
                  ─────────────────────────►  TENTATIVE  ──────────────────────────┐
                                              │                                     │
                                              │ matched on n_observations < min_hits │
                                              ▼                                     │
                                              TENTATIVE                             │
                                              │                                     │
                                              │ matched, n_observations ≥ min_hits  │
                                              ▼                                     │
                                              CONFIRMED  ◄───────────────┐          │
                                              │                          │ matched  │
                                              │ unmatched this frame     │          │
                                              ▼                          │          │
                                              COASTING (predicted only)  │          │
                                              │                          │          │
                                              │ time_since_update > max_age          │
                                              ▼                                     │
                                              DEAD ◄──────────────────────────────┘
```

- `min_hits` (default 3): observations required before a track is *confirmed*
  and emitted/persisted. Stops single-frame mask noise from creating fake IDs.
- `max_age` (default 30 frames): how many frames a track may go unmatched
  before it's killed. Allows brief occlusions (object hidden behind a leg or
  another pedestrian) without forcing a new ID on re-detection.
- `iou_threshold` (default 0.3): a Hungarian-assignment pair is accepted only
  if their IoU clears this bar. Below the bar, treat as unmatched.

Configurable via `Settings` env vars (`TRACKER_MIN_HITS`, `TRACKER_MAX_AGE`,
`TRACKER_IOU_THRESHOLD`, `TRACKER_MIN_AREA_PX`) — see `config.py`.

## Kalman bbox state

7-dim state, 4-dim measurement, constant velocity in centroid + scale, constant
aspect ratio. Same formulation as the original SORT paper:

| Symbol | Meaning |
| --- | --- |
| `u`, `v` | bbox centroid (px) |
| `s` | bbox scale (area = w·h) |
| `r` | bbox aspect ratio (w/h), assumed constant |
| `u'`, `v'`, `s'` | velocities |

Each frame: `predict()` advances state by F, then `update(bbox)` folds in the
new observation through the standard Kalman gain. We clamp negative scale
velocities so a fast-shrinking object can't predict a non-positive area.

## Object registry (SQLite)

```sql
CREATE TABLE objects (
  id              INTEGER PRIMARY KEY,
  first_seen_ns   INTEGER NOT NULL,
  last_seen_ns    INTEGER NOT NULL,
  n_observations  INTEGER NOT NULL,
  last_bbox_x     INTEGER NOT NULL,
  last_bbox_y     INTEGER NOT NULL,
  last_bbox_w     INTEGER NOT NULL,
  last_bbox_h     INTEGER NOT NULL,
  last_area_px    INTEGER NOT NULL
);
CREATE INDEX idx_objects_last_seen ON objects(last_seen_ns);
```

`ObjectRegistry.upsert_all(tracks)` is called every frame for confirmed tracks;
`first_seen_ns` is preserved on conflict so the row records the genuine first
sighting. Default location: `runs/objects.db` (overridable via
`REGISTRY_DB_PATH`). The DB outlives the detector process — kill it, restart
it, the history is still there.

## Published topic

`litter/tracked` — JSON payload, one message per processed frame:

```json
{
  "timestamp_ns": 1747929600000000000,
  "tracks": [
    {
      "id": 7,
      "bbox": [320, 180, 60, 90],
      "area_px": 4231,
      "first_seen_ns": 1747929598300000000,
      "last_seen_ns":  1747929600000000000,
      "n_observations": 14
    }
  ]
}
```

The masked-frame topic (`litter/masked_frame`) now also carries the bbox + ID
overlay, so any existing subscriber that already renders that topic will pick
up tracking IDs for free.

## Multi-agent extension path

Two pieces are deliberately *out of scope* for this PR but the seams are clean:

1. **World-frame deduplication.** The current tracker dedups within a single
   stream of frames from a single robot. Two robots seeing the same object —
   or one robot looping back to an old spot after the track was killed — will
   produce different IDs. Fix: subscribe to the Go2 odometry stream, project
   bbox centroids to a 2D ground-plane location using camera intrinsics +
   extrinsics, and add a second-pass spatial dedup keyed by `(x_world,
   y_world)`. Add columns `world_x`, `world_y`, `agent_id` to the registry.

2. **Shared object database.** The current `ObjectRegistry` is local SQLite.
   For coordinated collection across multiple robots, swap the SQLite handle
   for a shared store (Postgres, or a Zenoh-backed key-value store) and use
   `agent_id`-prefixed track IDs (`(agent_id, local_id) → global_id`) to avoid
   collisions. The `Track` dataclass and the publishing format don't have to
   change.

## Limitations

- **No re-identification.** A track that goes >`max_age` frames without a
  match (long occlusion, exiting and re-entering frame) gets a new ID on
  re-detection. DeepSORT-style appearance embeddings would fix this but add
  another inference cost.
- **Mask-merging.** Two pieces of touching litter come out of connected
  components as a single blob and so a single track. Either tighten the mask
  (morphological erosion before extraction) or move to a real instance
  segmentation head.
- **Frame-rate sensitivity.** `max_age` and the Kalman process noise are
  measured in frames, not seconds. Big swings in detector FPS (CPU vs GPU
  inference, queue backpressure) can shift effective tracking behaviour.
