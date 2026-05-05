# Overview
For a university project we are building a multi-agent robotic system. Our robot is a Unitree Go2 Robot dog. The main goal of the system should be to autonomously walk through an area and detect litter on the ground using its camera.

# Current state:
These are the current systems already developed
- Go2 Controller:
    - Accepts Movement commands, emote commands, publishes lidar-data, camera-stream, odometry and other robot data (Using the unitree-webrtc-connect python package)
- Eclipse-Zenoh Middleware for communicating with the robodog and receiving data streams
    - Note: no ROS2 is used or should be used
- Occupancy-map generation using go2's lidar-data (Blocked, free, unknown)
- A* Path Planner and path follower (Publish target coordinates, robot walks there using A* path planning)
- Inference system with a trained litter detection model, accepts go2 webcam input, output masks of recognized trash (can run in real-time)

# Goal:
The main goal is a multi-agent system that autonomously searches for litter based on the users' input. The following bullet points should be considered:
- The system is written in Python, using Pydantic-AI for the agents.
- It should be possible to start / interact with the agent through a chat window, voice-interface, etc.
- The agents need to plan the movement of the robot, occupancy-map, odometry and camera-feed are available.
  - To start the search and limit the search area, the users prompt should be used
  - For example User: "Search within 10m for litter" -> Agent evaluates message and returns a structured response for the search are around the robots current position.
  - Or Search 5m ahead of me -> Agent takes current robot position, returns response with the area marked but offset from the current position
  - Coordinates are in the frame of the robots' odometry.
- The detection from the trained litter-model might need to be checked / classified by a different system
  - For example, use Gemma4 + segmentation to check for false-positives and classify the detected trash
  - Detections currently do not have any ID, so they might be reported once per frame.
  - Possibly needs to be fixed inside the detector
- Detected trash should be saved to a database with its position
- The hardware the system will be running on is an Nvidia Jetson Orin AGX, should have plenty of performance of some local inference.
- Using cloud-requests is valid (through ollama etc.)
- Only one robot, no need to orchestrate several at the same time.
- Use MLflow tracing for agents
- Most important task is the implementation how the navigation should be done. The robot system already has a navigation tool implemented that can walk the robot to a specified coordinate, but we have to generate these coordinates first. Research into this topic online what approach would work the best. 

# Your Task
Your task is to help us find a good initial, but simple architecture for these multi-agent systems and some starting points how to start with this project. Research into this field and then discuss a proposed architecture with us. If our goals are too ambitious, propose a simplified, but still multi-agent system. Ask any open questions if anything needs to be further specified. Use LLMs where it makes sense to use or where it would be an interesting approach, keep to traditional algorithms where useful or necessary. 


# Robodog Zenoh Interface Summary
Below are all messages that are of interest for you from / to the robot.
All payloads are JSON (Pydantic `model_dump_json()` on send, `model_validate_json()` on receive). Topics are plain Zenoh key expressions.

## Robot Odometry

- **Topic:** `robodog/system_state/odometry`
- **Direction:** published by robodog/sim bridge
- **Payload:** `OdometryState`

```jsonc
{
  "x": 0.0, "y": 0.0, "z": 0.0,           // position in world frame, meters
  "quaternion": [qx, qy, qz, qw],         // orientation
  "timestamp": "2026-05-05T12:00:00Z"     // ISO-8601 UTC
}
```

Robot pose from onboard SLAM/odometry source. Published continuously while the bridge is connected.

## MovementVector Commands

- **Topic:** `robodog/command/motion/move`
- **Direction:** subscribed by robodog/sim bridge
- **Payload:** `MovementCommand`

```jsonc
{
  "x": 0.0,                    // forward/backward velocity, m/s
  "y": 0.0,                    // lateral velocity, m/s
  "z_deg": 0.0,                // yaw rate, deg/s
  "source": "controller",      // "controller" | "autonomous" | "planner"
  "timestamp": "..."           // ISO-8601 UTC
}
```

Velocity command. Stale messages (older than `movement-max-delay-ms`) are dropped — publish at a steady rate (~20 Hz) while moving.

## NavigationRequest

- **Topic:** `nav/request`
- **Direction:** subscribed by the nav manager
- **Payload:** `NavigationRequest`

```jsonc
{
  "request_id": "uuid-string",
  "lookahead_segments": 1,
  "segments": [
    {
      "target": { "x": 1.0, "y": 2.0, "theta": 0.0 },
      "max_speed": 0.5,                          // m/s, null = default
      "corridor": { "left_width": 0.5, "right_width": 0.5 },  // optional
      "allowed_deviation": 0.15,                 // meters
      "allowed_orientation_deviation": 0.1,      // radians
      "must_stop": true,
      "orientation_at_target": null,             // radians, null = don't care
      "rotation_allowed_on_segment": true
    }
  ]
}
```

A goal (single segment) or sequenced path (multiple segments, e.g. from a VDA5050 edge list). Correlate replies via `request_id`.

## NavStatus

- **Topic:** `nav/status`
- **Direction:** published by the nav manager
- **Payload:** `NavigationStatus`

```jsonc
{
  "timestamp": "...",
  "state": "following",        // idle | following | arrived_segment | arrived_final | blocked | failed
  "current_pose": { "x": 0.0, "y": 0.0, "theta": 0.0 },
  "distance_to_target": 0.42,  // meters to current waypoint
  "distance_to_final": 1.21,   // meters to final waypoint
  "current_segment_index": 0,
  "request_id": "uuid-string"  // matches the originating request
}
```

Streamed during navigation. Watch `state` for terminal transitions.

## OccupancyMap

- **Topic:** `robodog/map/occupancy`
- **Direction:** published by the occupancy node
- **Payload:** `OccupancyGrid` (ROS-style, base64 int8 grid)

```jsonc
{
  "width": 200, "height": 200,
  "resolution": 0.05,                // meters per cell
  "origin_x": -5.0, "origin_y": -5.0,// world coords of cell [0][0]
  "frame_id": "world",
  "timestamp": "...",
  "data": "<base64 int8, row-major height×width>"
}
```

Decode `data` with `base64` then `np.frombuffer(..., dtype=np.int8).reshape(height, width)`. Cell values: `-1` unknown, `0` free, `100` occupied.
