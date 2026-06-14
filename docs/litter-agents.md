# Litter Agents — Autonomous Search System

An AI-driven multi-agent pipeline that lets you say:

```
uv run litter-mission "Search 10 m around me for litter"
```

…and have the Unitree Go2 robot autonomously explore the area, detect litter candidates, validate each one with a vision LLM, and return a structured findings report.

---

## Architecture Overview

```
                      ┌────────────────────────────┐
   Natural-language   │   SearchAreaAgent (LLM)    │
   prompt  ──────────▶│   gemma4:31b-cloud / Ollama│
                      │   → AreaSpec (shape + dims) │
                      └────────────┬───────────────┘
                                   │
                      ┌────────────▼───────────────┐
                      │      MissionController      │
                      │  - FileMapProvider (YAML)   │
                      │  - ZenohPoseTracker         │
                      │  - ExplorationPlanner       │
                      │  - ZenohNavClient           │
                      │  - ValidationWorker (async) │
                      └──────────────┬─────────────┘
                                     │ Zenoh
                       ┌─────────────▼──────────────┐
                       │  Go2 robot stack (robodog)  │
                       │  nav/request, nav/status    │
                       │  robodog/localization/pose  │
                       │  litter/frame, litter/tracked│
                       └─────────────────────────────┘
                                     │
                      ┌──────────────▼─────────────┐
                      │  VisionAgent (Pydantic-AI)  │
                      │  gemma4:31b-cloud / Ollama  │
                      │  + FindingsDB (SQLite)      │
                      └────────────────────────────┘
```

### Components

| Module | Purpose |
|---|---|
| `litter_agents.agents.search_area` | LLM agent: parses natural-language → `AreaSpec` |
| `litter_agents.mapping` | ROS map_server loader, GridMap, area rasterizer |
| `litter_agents.hunter` | Raycasting, coverage tracker, info-gain planner |
| `litter_agents.zenoh_bridge` | Thread-safe Zenoh ↔ asyncio bridge |
| `litter_agents.mission.pose_tracker` | Live pose + ring-buffer history from Zenoh |
| `litter_agents.hunter.navigator` | `ZenohNavClient` — sends nav requests, waits for result |
| `litter_agents.mission.orchestrator` | `MissionController` — top-level async mission loop |
| `litter_agents.validation.vision_agent` | LLM vision agent — confirms litter from JPEG frames |
| `litter_agents.validation.worker` | `ValidationWorker` — subscribes, buffers, validates |
| `litter_agents.validation.findings_db` | SQLite findings store (`runs/findings.db`) |
| `litter_agents.mission.reporter` | Formats and saves mission report |

---

## Quick Start

### Prerequisites

1. Zenoh router running at `tcp/127.0.0.1:7447`
2. robodog stack publishing on `robodog/localization/pose`, `nav/request`, `nav/status`
3. Litter detector running (`uv run detector`)
4. Ollama with `gemma4:31b-cloud` available at `http://localhost:11434`
5. Map YAML at repo root (`my_lab_grid.yaml` + PNG)

### Run a mission

```bash
# Natural-language prompt (SearchArea LLM agent interprets it)
uv run litter-mission "Search 10 m around me for litter"

# Explicit circle — no LLM needed for area parsing
uv run litter-mission --area-circle 10

# Explicit rectangle (6 m wide, 4 m deep in front of robot)
uv run litter-mission --area-rect 6 4

# Confirm before moving, save JSON report
uv run litter-mission --area-circle 8 --confirm --save-report runs/reports
```

### Offline simulation (no robot needed)

```bash
# Simulate exploration on the lab map with a 6 m circle
uv run litter-sim --circle 6

# Rectangular area
uv run litter-sim --rect 8 5
```

---

## Configuration

All settings are in `src/litter_agents/config.py` and can be overridden via environment variables (prefix `LITTER_AGENT_`) or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `LITTER_AGENT_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `LITTER_AGENT_OLLAMA_TEXT_MODEL` | `gemma4:31b-cloud` | SearchArea agent model |
| `LITTER_AGENT_OLLAMA_VISION_MODEL` | `gemma4:31b-cloud` | Vision validation model |
| `LITTER_AGENT_MAP_FILE` | `my_lab_grid.yaml` | Map YAML path |
| `LITTER_AGENT_ROBOT_RADIUS_M` | `0.35` | Robot radius for inflation |
| `LITTER_AGENT_SEEN_RANGE_M` | `2.5` | Camera effective range |
| `LITTER_AGENT_FOV_DEG` | `70.0` | Camera field of view |
| `LITTER_AGENT_COVERAGE_THRESHOLD` | `0.95` | Stop when 95% covered |
| `LITTER_AGENT_FINDINGS_DB` | `runs/findings.db` | SQLite findings database |
| `LITTER_AGENT_MISSION_IMAGES_DIR` | `runs/missions` | Saved trigger frames |
| `LITTER_AGENT_ZENOH_ROUTER_ENDPOINT` | `tcp/127.0.0.1:7447` | Zenoh router |

---

## Data Flow

### Exploration (deterministic, no LLM)

1. `FileMapProvider` loads the occupancy grid from `my_lab_grid.yaml`
2. Grid is inflated by `robot_radius_m` to create a safe-navigation grid
3. `rasterize_area` renders the search polygon onto the grid
4. `ExplorationPlanner` picks waypoints using info-gain scoring:
   - 36 candidate directions, sampled every 0.3 m
   - Each candidate: raycasting sees how much unseen area it would expose
   - Score = `w_gain·gain_m² − w_dist·distance − w_turn·|Δheading|`
5. `ZenohNavClient` sends `NavigationRequest` to robodog and awaits terminal status
6. `CoverageTracker` is updated at 5 Hz from the live pose stream
7. Mission ends when coverage ≥ 95%, no candidates remain, or timeout

### Validation (async, parallel with exploration)

1. `ValidationWorker` subscribes to `litter/frame` (JPEG) → ring buffer (60 frames)
2. `ValidationWorker` subscribes to `litter/tracked` (JSON) → queue
3. For each track with `n_observations ≥ 10`, fire-and-forget:
   - Find nearest buffered JPEG to `track.first_seen_ns`
   - Save JPEG to `runs/missions/<mission_id>/track_<id>_<ns>.jpg`
   - Call `VisionAgent.validate(jpeg)` → `LitterValidationResult`
   - Write `FindingRecord` to `runs/findings.db`
4. At mission end, `MissionController` attaches all findings to `MissionResult`

---

## Zenoh Topics

| Topic | Direction | Type | Description |
|---|---|---|---|
| `robodog/localization/pose` | sub | `OdometryState` JSON | Robot pose at ~10 Hz |
| `nav/request` | pub | `NavigationRequest` JSON | Send waypoint to navigator |
| `nav/status` | sub | `NavigationStatus` JSON | Navigator feedback at ~2 Hz |
| `litter/frame` | sub | JPEG bytes | Camera frame (republished by detector) |
| `litter/tracked` | sub | `TrackedMsg` JSON | Active tracks from ByteTracker |

---

## Findings Database

Schema in `runs/findings.db`:

```sql
CREATE TABLE findings (
    id           INTEGER PRIMARY KEY,
    mission_id   TEXT,       -- e.g. "20260613T123456"
    run_ts       TEXT,       -- ISO-8601 mission start
    track_id     INTEGER,    -- ByteTracker ID
    confirmed    INTEGER,    -- 1 = LLM said is_litter
    confidence   REAL,       -- 0.0 – 1.0
    description  TEXT,       -- one-line LLM description
    category     TEXT,       -- "plastic bottle", "can", etc.
    pose_x       REAL,
    pose_y       REAL,
    pose_theta   REAL,
    image_path   TEXT,       -- path to saved JPEG trigger frame
    validated_at TEXT
);
```

Query findings after a mission:

```python
import sqlite3
con = sqlite3.connect("runs/findings.db")
rows = con.execute("SELECT * FROM findings WHERE confirmed=1").fetchall()
```

---

## Testing

```bash
uv run pytest tests/hunter/   # exploration planner (24 tests)
uv run pytest tests/mission/  # validation, reporter, protocol (15 tests)
uv run pytest                 # full suite (82 tests)
```
