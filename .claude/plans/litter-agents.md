# Multi-Agent Litter-Search System (`litter_agents`)

> **Status:** Approved 2026-06-12, implementation not yet started.
> Task description: [prompts/litter-agents.md](../../prompts/litter-agents.md) · Background facts: [.claude/memory/](../memory/MEMORY.md)

## Context

Per [prompts/litter-agents.md](prompts/litter-agents.md): build a multi-agent system where a user prompt like *"Search 10m around me for litter"* drives a Unitree Go2 to autonomously explore a pre-mapped area, validate detected litter with an Ollama Cloud vision model, and store findings in a database. Existing pieces: the detector/tracker pipeline in this repo (publishes `litter/tracked` + `litter/frame` on Zenoh), the robodog-digipro control stack (publishes `robodog/localization/pose`, accepts straight-line `nav/request`, streams `nav/status` incl. `blocked`), and a Zenoh router.

**Decisions agreed with user:**
- Static map `my_lab_grid.png` is MOLA mm2grid output → ROS map_server format (PNG + YAML: `resolution`, `origin [x,y,yaw]` bottom-left, trinary white=free/black=occupied/gray=unknown). Use **placeholder YAML values** for now; abstract behind a `MapProvider` so Zenoh/REST sources plug in later.
- Vision model: **Gemma on Ollama Cloud** (configurable). Pydantic-AI for all agent code.
- Exploration/path-planning loop: **pure deterministic algorithm, no LLM** (rework vs. original sketch — agreed). Agents only for: search-area parsing, detection validation, optional report summary.
- Camera: FoV 70°, effective seen-range **2.5 m**.

**Reworks vs. the original task notes (flagged, agreed or low-risk):**
- Path planning is deterministic; the "agents" are leaf functions composed by plain asyncio code, not agent-as-tool delegation (determinism, cost, offline testability).
- "Several agents vs. tools for detection" question → resolved as: one async **validation worker** (not an LLM agent) that calls one vision agent per stable track. DB writes are plain code, not an LLM tool.
- On Ollama Cloud, structured output must use `PromptedOutput`/`ToolOutput` (NativeOutput unsupported) — default `prompted`, configurable.

## Verified wire contracts (copy, don't import — robodog repo is reference-only)

- `robodog/localization/pose`: `OdometryState {x,y,z, quaternion[qx,qy,qz,qw], timestamp}` (meters, world frame) — robodog `src/interfaces/robot.py`; copy `quaternion_to_yaw` from `navigation.py`.
- `nav/request`: `NavigationRequest {request_id, segments:[{target:{x,y,theta}, max_speed, allowed_deviation=0.15, must_stop, orientation_at_target, ...}], lookahead_segments}` — straight-line pure-pursuit, **no obstacle avoidance**; a new request preempts the current one.
- `nav/status` (~2 Hz): `{state: idle|following|arrived_segment|arrived_final|blocked|failed, current_pose, distance_to_target, request_id, ...}`. BLOCKED after ~5 s no progress; executor retreats toward last waypoint and waits for a new request.
- `robodog/map/occupancy`: `OccupancyGrid {width,height,resolution,origin_x/y,frame_id,data: base64 int8}` (-1 unknown / 0 free / 100 occupied) — canonical in-memory map representation.
- This repo: `litter/tracked` JSON `{timestamp_ns, tracks:[{id, bbox:[x,y,w,h], area_px, first_seen_ns, last_seen_ns, n_observations}]}`; `litter/frame` raw JPEG (no timestamp).

## Package layout (new: `src/litter_agents/`; `litter_detector` untouched except one line)

```
src/litter_agents/
├── config.py                  # AgentSettings(BaseSettings), robodog topic constants
├── interfaces/
│   ├── robodog.py             # copied: Pose2D, quaternion_to_yaw, OdometryState,
│   │                          #   NavigationRequest/Segment/Status/State, OccupancyGrid
│   └── detections.py          # TrackMsg, TrackedMsg (pydantic models of litter/tracked)
├── zenoh_bridge.py            # AsyncZenoh: subscribe_latest/subscribe_queue/publish_json;
│                              #   callbacks → loop.call_soon_threadsafe, no work on zenoh threads
├── mapping/
│   ├── provider.py            # MapProvider ABC; FileMapProvider (map_server YAML+PNG,
│   │                          #   np.flipud, trinary threshold); ZenohMapProvider (subscribes occupancy once)
│   ├── grid.py                # GridMap: int8 numpy + resolution/origin; world↔grid;
│   │                          #   inflate(robot_radius) via cv2.dilate (unknown treated as occupied)
│   └── raster.py              # rasterize_area(spec, robot_pose, grid) -> bool mask (cv2.circle/fillPoly)
├── hunter/                    # deterministic exploration — pure functions, no I/O (sim seam)
│   ├── raycast.py             # visible_cells(): vectorized DDA, annular 70° wedge, obstacles block
│   ├── coverage.py            # CoverageTracker: seen-grid OR-update from live pose @5 Hz,
│   │                          #   denominator = target & free & reachable; fraction()
│   ├── reachability.py        # reachable_mask (scipy.ndimage.label flood-fill), DynamicObstacles, Blacklist
│   ├── scoring.py             # candidate gen + info-gain scoring (see Algorithm)
│   ├── planner.py             # ExplorationPlanner: next_waypoint(pose) -> Candidate|None,
│   │                          #   register_block(), done()
│   └── navigator.py           # NavInterface Protocol; ZenohNavClient (goto/halt, request_id correlation,
│                              #   timeout = max(20s, 4×dist/speed), status-silence → TIMEOUT)
├── validation/
│   ├── crops.py               # decode/crop(+15% pad)/context-annotate helpers (pure cv2)
│   ├── findings.py            # FindingsRepository (sqlite3, mirrors tracker/registry.py style)
│   └── worker.py              # DetectionValidationWorker (async; queue + 2 LLM consumers)
├── agents/
│   ├── models.py              # build_model(): OpenAIChatModel + OllamaProvider(base_url, OLLAMA_API_KEY);
│   │                          #   output_spec(): PromptedOutput (default) | ToolOutput
│   ├── search_area.py         # SearchAreaSpec + parsing agent (text model)
│   ├── validator.py           # LitterValidation + vision agent (crop + boxed-context images
│   │                          #   via BinaryContent, media_type='image/jpeg')
│   └── reporter.py            # MissionReport (built deterministically) + optional LLM summary_text
├── mission/
│   ├── pose_tracker.py        # PoseSource Protocol; ZenohPoseTracker (latest + ring buffer,
│   │                          #   pose_at(ts_ns) nearest-match, distance integrator)
│   ├── orchestrator.py        # MissionController.run(prompt) -> MissionReport
│   └── main.py                # CLI litter-mission "PROMPT" [--map] [--confirm] [--no-llm-summary]
└── sim/
    ├── fake_nav.py            # FakeNav + FakePoseSource (shared clock, straight-line interpolation,
    │                          #   optional blocked_discs to simulate BLOCKED); no zenoh imports
    └── sim_main.py            # CLI litter-sim: full loop offline on my_lab_grid + cv2 viz frames
```

Plus repo root `my_lab_grid.yaml` (placeholder, clearly commented):
```yaml
image: my_lab_grid.png
resolution: 0.05                  # PLACEHOLDER — replace with real mm2grid metadata
origin: [-10.125, -5.625, 0.0]    # PLACEHOLDER — centers the 405x225 map on world origin
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

## One-line `litter_detector` change

[detector/main.py:211](src/litter_detector/detector/main.py:211): `self.frame_pub.put(payload)` → `self.frame_pub.put(payload, attachment=str(frame_ts_ns).encode())` (`frame_ts_ns` already in scope at line 168). Gives exact frame↔track pairing; worker falls back to latest-frame-before-tracked-message when attachment absent.

## Exploration algorithm (hunter/)

- **Coverage update**: from live pose at 5 Hz (skip if moved <2 cm and <1°): DDA-raycast ~90 rays across 70° out to 2.5 m (min range 0.3 m blind spot), blocked by occupied|unknown on the **raw** grid; OR into `seen`.
- **Candidate generation** (per replan): 36 directions × samples every 0.5 m from 0.75 m to min(ray-clear distance on **inflated** grid − robot radius, 8 m). Per direction, gain is cumulative: `cum |= visible_cells(sample_pose, heading=φ)`; `gain_k = (cum & unseen_target).sum()·res²` — counts what's seen *en route plus at arrival*. Travel feasibility on the inflated grid doubles as the collision-free straight-line corridor check.
- **Score** = `w_gain·gain_m² − w_dist·distance − w_turn·|Δheading|` (defaults 1.0 / 0.25 / 0.3; gains in m² so weights survive resolution changes). Reject candidates within `blacklist_radius_m` of failed goals.
- **Reachability**: flood-fill (`scipy.ndimage.label`) on inflated free space from robot cell; unreachable target cells excluded from the coverage denominator. On BLOCKED/FAILED/TIMEOUT: wait 2.5 s (executor retreat), add dynamic obstacle disc at the stall pose, blacklist the goal, re-flood-fill → denominator shrinks (handles "marked free but not actually reachable").
- **Termination**: `seen/denominator ≥ 0.95` (config), or 3 consecutive replans with best gain < 0.15 m², or safety caps (`mission_max_duration_s=1800`, `mission_max_waypoints=200`).
- **Nav**: single-segment `NavigationRequest` per waypoint, `must_stop=True`, `orientation_at_target=None` (arrives facing travel direction φ — exactly what scoring assumed).

## Validation worker

- Readiness gate per track id: `n_observations ≥ 10`, bbox ≥ 32×32 and `area_px ≥ 400`, bbox not touching frame border (4 px), id not already processed/in-flight. One shot per track id per mission (negatives persisted as `rejected` so they don't requeue).
- Job: crop JPEG (+15% pad) + downscaled context frame with bbox drawn + `pose_at(track.last_seen_ns)` + camera bearing `(cx/frame_w − 0.5)·fov`.
- `asyncio.Queue(16)` drop-oldest; 2 consumer coroutines; `asyncio.wait_for(60 s)` per LLM call; pydantic-ai `retries=2` for schema retries + 1 manual retry on timeout/5xx; final failure → `error` row.
- SQLite (`runs/findings.db`, repo-root-relative like the existing registry): `findings` table (mission_id, track_id, status validated|rejected|error, category, confidence, description, robot pose, bearing, bbox, timestamps, image paths, raw_response, `UNIQUE(mission_id, track_id)`) + `missions` table (prompt, area_spec_json, coverage_fraction, distance_m, counts, report_json). Images → `runs/missions/<mission_id>/findings/track_<id>_{crop,ctx}.jpg`. Spatial dedup is reporting-side only (group findings <1 m apart, same category, as "possible duplicates" — no depth, so no true litter position).

## Agent output models

```python
class SearchAreaSpec(BaseModel):      # flat model + model_validator; robot at origin facing +x
    shape: Literal["circle","rectangle","polygon"]
    radius_m: float | None            # circle
    width_m: float | None; depth_m: float | None   # rectangle
    polygon_points: list[tuple[float,float]] | None
    center_dx_m: float = 0.0; center_dy_m: float = 0.0   # offset in robot frame
    rotate_with_robot: bool = True
    rationale: str = ""

class LitterValidation(BaseModel):
    is_litter: bool
    category: Literal["plastic","paper","cardboard","metal","glass","organic","cigarette","textile","other"] | None
    confidence: float                 # 0..1
    description: str

class MissionReport(BaseModel):       # built deterministically from DB + planner stats
    mission_id, prompt, area, coverage_fraction, reachable_target_m2,
    duration_s, distance_traveled_m, n_waypoints, n_blocked,
    findings: list[FindingSummary], n_rejected, n_errors, summary_text  # summary_text = optional LLM call
```

Orchestrator flow: pose `wait_first(10 s)` (fail fast: "is robodog running?"), warn-only probe on `litter/tracked` → load map, inflate, flood-fill → area agent → rasterize (+`--confirm` prints area stats) → start validation worker → exploration loop (concurrent coverage task) → halt nav, drain queue (≤90 s) → report (persist + pretty-print). Ctrl-C → halt, flush, partial report.

## pyproject changes

```toml
[project.scripts]                    # add
litter-mission = "litter_agents.mission.main:main"
litter-sim = "litter_agents.sim.sim_main:main"
# dependencies += "pydantic-ai-slim[openai]>=1.0.0", "pyyaml>=6.0"
# [tool.hatch.build.targets.wheel] packages += "src/litter_agents"
# dev group += "pytest-asyncio>=0.24"
```

## Testing & verification

- `tests/hunter/` (dirs already exist): raycast geometry (wedge area ≈ analytic ±10%, wall shadowing, unknown blocks), coverage denominator (excludes unknown/occupied/unreachable), reachability + dynamic-disc denominator shrink, scoring orderings (gain/distance/turn, no candidates through walls, determinism), raster shapes vs analytic area, and the integration crown jewel: **full loop with FakeNav on the real `my_lab_grid.png`** — terminates ≤60 waypoints at ≥95% coverage; second scenario with a blocking disc exercises BLOCKED→blacklist→shrink→still terminates.
- `tests/agents/`: `TestModel` smoke for all agents; `FunctionModel` scripted end-to-end (area→raster, validator→DB rows/files); worker unit tests with injected fake agent (gating, frame pairing w/ and w/o attachment, overflow, dedup, error rows) on `tmp_path`; one `@pytest.mark.live` Ollama Cloud canary, skipped without `OLLAMA_API_KEY`.
- No zenoh in tests. Per-phase: `uv run ruff check .`, `uv run ty check`, `uv run pytest`.
- Manual: `uv run litter-sim --circle 6` renders coverage/trajectory frames to `runs/sim/` (also the weight-tuning harness); then live bench test against robodog (Phase 3), then full end-to-end with planted litter (Phase 5).

## Implementation phases

1. **Foundations**: config, interface copies, GridMap/FileMapProvider/raster, `my_lab_grid.yaml`, pyproject. Verify: sync/lint/type, raster+grid tests, loaded map free-cell stats sane (~8.5k free px).
2. **Hunter core + sim**: raycast, coverage, reachability, scoring, planner, fake_nav, `litter-sim`. Verify: hunter tests green; sim achieves ≥95% coverage on lab map; tune weights from rendered frames. *De-risks the whole algorithm offline.*
3. **Zenoh integration**: bridge, pose tracker, ZenohNavClient, orchestrator skeleton with `--area-circle R` bypass. Verify live against robodog: 3 m circle mission; deliberately block robot → confirm retreat/blacklist/replan.
4. **Validation worker + vision agent + DB** (+ the 1-line detector attachment change). Verify: worker tests; live `camera`+`detector`+planted litter → rows + crops; run live Ollama canary.
5. **Search-area agent + reporter + full CLI + docs** (CLAUDE.md/README blurb). Verify: full "Search 10m around me for litter" end-to-end; final lint/type/test sweep.

## Risks / follow-ups for user

1. **Map metadata is fabricated** (0.05 m/px, centered origin) — need the real mm2grid YAML and confirmation that robodog's localization frame coincides with the map frame (loader has room for a static transform field; `origin yaw ≠ 0` raises NotImplementedError for now).
2. **Ollama Cloud specifics**: exact Gemma vision model name on the cloud registry, and whether the provider base_url needs a `/v1` suffix in the installed pydantic-ai version — isolated in `agents/models.py`; live test is the canary.
3. **No depth** → findings store robot pose + camera bearing, not litter world position. Future hook: RealSense depth topics in robodog.
4. `camera_min_range_m=0.3` blind-spot guess — check against actual camera mount pitch.
5. In-place rotation sweep at waypoints (±120° on arrival) deferred as highest-value v2 efficiency improvement; scoring already accounts for arrival heading.
