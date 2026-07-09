---
name: robodog-digipro-wire-contracts
description: Sibling repo ~/PycharmProjects/robodog-digipro — Go2 control stack Zenoh topics and message schemas the litter-agents system talks to
metadata: 
  node_type: memory
  type: reference
  originSessionId: 06a21a74-8d84-4079-8467-34623b94d0ce
---

Repo: `/home/fabian/PycharmProjects/robodog-digipro` (reference-only — copy pydantic models into this repo, never import or modify). Schemas live in `src/interfaces/` (`navigation.py`, `robot.py`, `occupancy.py`, `topics/topics.py`).

- `robodog/localization/pose` — `OdometryState {x,y,z, quaternion[qx,qy,qz,qw], timestamp}` (meters, world frame; `quaternion_to_yaw` helper in navigation.py). Currently odometry passthrough.
- `nav/request` — `NavigationRequest {request_id, segments:[{target:{x,y,theta}, max_speed, allowed_deviation=0.15, must_stop, orientation_at_target, ...}], lookahead_segments}`. Straight-line pure-pursuit execution, **no obstacle avoidance**; a new request preempts the running one.
- `nav/status` (~2 Hz) — `{state: idle|following|arrived_segment|arrived_final|blocked|failed, current_pose, distance_to_target, request_id, ...}`. BLOCKED after ~5 s of <5 cm/<5° progress; executor retreats toward last waypoint and stays BLOCKED until a new request arrives.
- `robodog/map/occupancy` — `OccupancyGrid {width, height, resolution, origin_x/y, frame_id, data: base64 row-major int8}` with -1 unknown / 0 free / 100 occupied; has `world_to_grid`/`grid_to_world`.
- Go2 camera frames: `robodog/sensors/go2_camera` (JPEG); RealSense depth topics exist (`robodog/sensors/realsense/*`) — future hook for true litter positions.

Same Zenoh router convention as this repo: `tcp/127.0.0.1:7447`, env `ZENOH_ROUTER_ENDPOINT`. robodog pins eclipse-zenoh 1.7.1 (this repo: ≥1.9.0). Used by [[litter-agents-plan-decisions]].
