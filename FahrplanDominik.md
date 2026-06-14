# Fahrplan: Multi-Agent Litter-Search System

> Erstellt: 2026-06-13  
> Basis: `prompts/litter-agents.md`, `.claude/plans/litter-agents.md`, bestehender Code in `src/litter_detector/`  
> Branch: `feature/litter-agents`

---

## 1. Wo stehen wir jetzt?

### Was bereits funktioniert

| Komponente | Status | Beschreibung |
|---|---|---|
| **CNN-Training** |  Fertig | Loop 5 abgeschlossen, bestes Modell `val_iou=0.7486`, ONNX-Export |
| **Echtzeit-Detektor** |  Fertig | ONNX-Inferenz, ByteTrack-Tracking, Zenoh-Publikation |
| **Kamera-Pipeline** |  Fertig | WebcamSource + Go2Source, JPEG-Encoding, Zenoh |
| **Objekt-Registry** |  Fertig | SQLite-Persistenz in `runs/objects.db`, Tracker-Lifecycle |
| **Observability** |  Fertig | OpenTelemetry → Grafana LGTM Stack |
| **Multi-Agent System** |  Noch nicht begonnen | Das ist unser Ziel |

### Was wir bauen wollen

Ein autonomes Such-System, bei dem der Nutzer sagt:
```
"Search 10m around me for litter"
```
…und der Unitree Go2 Roboter:
1. Das Gebiet ausleuchtet (deterministisch, Raycasting)
2. Gefundenen Müll per Vision-LLM verifiziert und klassifiziert
3. Alle Funde in einer SQLite-Datenbank speichert
4. Am Ende einen Bericht erstellt

---

## 2. Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                        NUTZER-PROMPT                            │
│             "Search 10m around me for litter"                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              SearchArea-Agent  (Pydantic-AI + LLM)              │
│  Parst den Prompt → SearchAreaSpec (Kreis, 10m, kein Offset)    │
└────────────────────────┬────────────────────────────────────────┘
                         │ SearchAreaSpec
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MissionController                             │
│  Lädt Karte, inflated Grid, Flood-Fill, startet alle Tasks      │
│                                                                  │
│   ┌──────────────────┐     ┌──────────────────────────────┐    │
│   │ ExplorationLoop  │     │  ValidationWorker            │    │
│   │ (reine Python-   │     │  (asyncio Queue, 2 Consumer) │    │
│   │  Logik, kein LLM)│     │  → Vision-Agent (Gemma via   │    │
│   │                  │     │    Ollama Cloud)              │    │
│   │ Zenoh ◄──────────┤     │  → FindingsRepository        │    │
│   │  litter/tracked  │     │    (SQLite)                  │    │
│   │  robodog/pose    │     └──────────────────────────────┘    │
│   │  nav/request ────►                                         │
│   │  nav/status  ◄───►                                         │
│   └──────────────────┘                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │ MissionReport
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Reporter-Agent  (optional, Pydantic-AI)            │
│  Generiert lesbaren Zusammenfassungstext aus den DB-Daten        │
└─────────────────────────────────────────────────────────────────┘
```

### Zentrale Design-Entscheidungen (bereits festgelegt)

- **Pydantic-AI** für alle Agent-Logik (SearchArea, Validator, Reporter)
- **Exploration/Pfadplanung: deterministisch** – kein LLM im Planungsloop (Kosten, Geschwindigkeit, Reproduzierbarkeit)
- **Gemma via Ollama Cloud** für Vision-Validierung (konfigurierbar)
- **Zenoh** für alle Echtzeit-Kommunikation (Pose, Nav, Tracks, Frames)
- **SQLite** für Findings (`runs/findings.db`) – analoges Muster zu `runs/objects.db`
- **Kein Import aus robodog-Repo** – Interfaces werden kopiert

---

## 3. Paketstruktur (neu: `src/litter_agents/`)

```
src/litter_agents/
├── config.py                   # AgentSettings (pydantic-settings)
├── interfaces/
│   ├── robodog.py              # Pose2D, OdometryState, NavigationRequest, OccupancyGrid
│   └── detections.py           # TrackMsg, TrackedMsg (Wire-Format von litter/tracked)
├── zenoh_bridge.py             # AsyncZenoh: Thread→asyncio-Bridge (call_soon_threadsafe)
├── mapping/
│   ├── provider.py             # MapProvider ABC; FileMapProvider; ZenohMapProvider (später)
│   ├── grid.py                 # GridMap: NumPy int8 + resolution/origin, inflate()
│   └── raster.py               # rasterize_area(spec, robot_pose, grid) → bool-Maske
├── hunter/                     # Deterministischer Explorations-Kern (pure functions)
│   ├── raycast.py              # visible_cells(): DDA-Algorithmus, 70° FoV, blocked by obstacles
│   ├── coverage.py             # CoverageTracker: seen-Grid, denominator, fraction()
│   ├── reachability.py         # Flood-Fill, DynamicObstacles, Blacklist
│   ├── scoring.py              # Kandidaten-Generierung + Info-Gain Scoring
│   ├── planner.py              # ExplorationPlanner: next_waypoint(), register_block(), done()
│   └── navigator.py            # NavInterface Protocol; ZenohNavClient
├── validation/
│   ├── crops.py                # Crop + Context-Frame Utilities (pure OpenCV)
│   ├── findings.py             # FindingsRepository (SQLite, analog zu registry.py)
│   └── worker.py               # DetectionValidationWorker (asyncio.Queue, 2 Consumer)
├── agents/
│   ├── models.py               # build_model(): OllamaProvider + OpenAIChatModel
│   ├── search_area.py          # SearchAreaSpec + Parsing-Agent
│   ├── validator.py            # LitterValidation + Vision-Agent
│   └── reporter.py             # MissionReport + optionaler LLM-Summarizer
├── mission/
│   ├── pose_tracker.py         # ZenohPoseTracker: latest pose + pose_at(ts_ns)
│   ├── orchestrator.py         # MissionController.run(prompt) → MissionReport
│   └── main.py                 # CLI: litter-mission "PROMPT"
└── sim/
    ├── fake_nav.py             # FakeNav + FakePoseSource (kein Zenoh, für Tests)
    └── sim_main.py             # CLI: litter-sim --circle 6
```

---

## 4. Schritt-für-Schritt Implementierungsplan

### Phase 1: Fundament (Geschätzter Aufwand: ~1 Tag)

**Ziel:** Alles ist installierbar, lintbar, typ-checkbar. Karte ladbar. Tests grün.

#### 1.1 pyproject.toml erweitern
```toml
[project.scripts]
litter-mission = "litter_agents.mission.main:main"
litter-sim = "litter_agents.sim.sim_main:main"

# Dependencies hinzufügen:
# "pydantic-ai-slim[openai]>=1.0.0"
# "pyyaml>=6.0"
# "scipy>=1.13.0"

[tool.hatch.build.targets.wheel]
packages = ["src/litter_detector", "src/litter_agents"]

[tool.pytest.ini_options]
asyncio_mode = "auto"         # für pytest-asyncio
```

Dev-Dep: `"pytest-asyncio>=0.24"`

#### 1.2 Karten-YAML anlegen (`my_lab_grid.yaml`)
```yaml
# PLACEHOLDER — Echte Werte aus mm2grid YAML eintragen
image: my_lab_grid.png
resolution: 0.05          # Meter pro Pixel
origin: [-10.125, -5.625, 0.0]   # [x, y, yaw] unten-links
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

#### 1.3 `config.py` für litter_agents
```python
from pydantic_settings import BaseSettings

class AgentSettings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = "ollama"
    ollama_text_model: str = "gemma4:31b-cloud"
    ollama_vision_model: str = "gemma4:31b-cloud"
    agent_output_mode: str = "prompted"   # oder "tool"
    
    # Karte
    map_file: str = "my_lab_grid.yaml"
    robot_radius_m: float = 0.35
    
    # Kamera / Exploration
    fov_deg: float = 70.0
    seen_range_m: float = 2.5
    camera_min_range_m: float = 0.3
    
    # Scoring
    w_gain: float = 1.0
    w_dist: float = 0.25
    w_turn: float = 0.3
    coverage_threshold: float = 0.95
    
    # Zenoh (gleiche Env-Var wie litter_detector)
    zenoh_router_endpoint: str = "tcp/127.0.0.1:7447"
    
    # Findings DB
    findings_db: str = "runs/findings.db"
    mission_images_dir: str = "runs/missions"
```

#### 1.4 Interfaces kopieren (aus robodog-Repo, NICHT importieren)
`interfaces/robodog.py`:
- `Pose2D`, `quaternion_to_yaw()`
- `OdometryState`
- `NavigationRequest`, `NavigationSegment`
- `NavigationStatus`, `NavigationState` (Enum: idle/following/arrived_segment/arrived_final/blocked/failed)
- `OccupancyGrid`

`interfaces/detections.py`:
- `TrackMsg` (id, bbox:[x,y,w,h], area_px, first_seen_ns, last_seen_ns, n_observations)
- `TrackedMsg` (timestamp_ns, tracks: list[TrackMsg])

#### 1.5 GridMap + FileMapProvider
- `GridMap`: NumPy `int8` Array + `resolution` (m/px) + `origin_x/y`
  - `world_to_grid(x, y) → (row, col)`
  - `grid_to_world(row, col) → (x, y)`
  - `inflate(robot_radius_m) → GridMap` via `cv2.dilate` (unknown als occupied behandeln)
- `FileMapProvider`: Lädt YAML + PNG
  - `np.flipud()` weil PNG oben-links, ROS unten-links
  - Trinary-Threshold: weiß=frei (0), schwarz=belegt (100), grau=unbekannt (-1)

#### 1.6 `raster.py`
```python
def rasterize_area(spec: SearchAreaSpec, robot_pose: Pose2D, grid: GridMap) -> np.ndarray:
    # Gibt bool-Maske zurück (True = Suchgebiet)
    # circle → cv2.circle
    # rectangle → cv2.fillPoly (rotiert mit Roboter-Heading wenn rotate_with_robot)
    # polygon → cv2.fillPoly
```

**Verifikation Phase 1:**
```bash
uv sync
uv run ruff check .
uv run ty check
uv run pytest tests/hunter/test_grid.py tests/hunter/test_raster.py
# Soll: ~8.500 freie Pixel im Lab-Grid zählen (405×225 px)
```

---

### Phase 2: Hunter-Kern + Offline-Simulation (Geschätzter Aufwand: ~2 Tage)

**Ziel:** Der Explorations-Algorithmus läuft offline auf `my_lab_grid.png` und erreicht ≥95% Coverage. Das ist der wichtigste De-Risking Schritt – hier sehen wir ob der Algorithmus funktioniert ohne echten Roboter.

#### 2.1 Raycasting (`hunter/raycast.py`)

Der **DDA-Algorithmus** (Digital Differential Analyzer) ist die effizienteste Methode für Grid-Raycasting:

```
Grundidee:
  Für jeden Strahl (von 36 Richtungen × 70° FoV):
    1. Startpunkt = Roboter-Position im Grid
    2. Schrittweise entlang des Strahls
    3. Stop wenn: Hindernis (belegt oder unbekannt), Reichweite überschritten
    4. Alle passierten Zellen → "gesehen"
```

```python
def visible_cells(
    pose: Pose2D,
    grid: GridMap,
    fov_deg: float = 70.0,
    range_m: float = 2.5,
    min_range_m: float = 0.3,
    n_rays: int = 90,
) -> np.ndarray:  # bool-Maske, gleiche Form wie grid
```

**Wichtige Details:**
- Annularer Wedge: Strahlen beginnen erst bei `min_range_m` (Kamera sieht nah nichts)
- Blocking: schwarze (belegt) UND graue (unbekannte) Zellen blockieren
- Vektorisierung: alle Strahlen parallel als NumPy-Arrays

#### 2.2 Coverage Tracker (`hunter/coverage.py`)

```python
class CoverageTracker:
    seen: np.ndarray        # bool, True = diese Zelle wurde gesehen
    target_mask: np.ndarray # bool, Suchgebiet (aus raster.py)
    
    def update(self, pose: Pose2D) -> None:
        # Aufgerufen bei ~5 Hz
        # Skip wenn Bewegung < 2 cm und < 1°
        new_cells = visible_cells(pose, self.grid, ...)
        self.seen |= new_cells
    
    def fraction(self) -> float:
        # seen & target & free & reachable / target & free & reachable
```

**Denominator-Berechnung:** Nur Zellen zählen die:
1. Im Suchgebiet liegen (`target_mask`)
2. Frei sind (nicht belegt, nicht unbekannt)
3. Erreichbar sind (Flood-Fill von Roboter-Position)

#### 2.3 Erreichbarkeit (`hunter/reachability.py`)

```python
def reachable_mask(grid: GridMap, start_row: int, start_col: int) -> np.ndarray:
    # scipy.ndimage.label für Flood-Fill auf dem *inflierten* Grid
    # Gibt bool-Maske zurück: True = von Start aus erreichbar
```

**DynamicObstacles:** Wenn der Roboter BLOCKED wird:
- Disc mit Radius ~0.5m um die Stall-Position auf dem Grid markieren
- Flood-Fill neu berechnen → Denominator schrumpft
- Ziel wird auf Blacklist gesetzt (Radius konfigurierbar)

#### 2.4 Scoring (`hunter/scoring.py`)

```
Kandidaten-Generierung:
  Für jede der 36 Richtungen (0°, 10°, 20°, ..., 350°):
    Für jeden Abstand (0.75m, 1.25m, 1.75m, ..., bis min(Strahlende, 8m)):
      Kandidat = (Richtung, Abstand)

Score eines Kandidaten:
  gain_m² = Fläche der neu gesehenen Zellen auf dem Weg + am Ziel
  score = w_gain × gain_m² − w_dist × abstand_m − w_turn × |delta_heading_deg| / 180

  Ablehnen wenn:
    - Kandidat auf der Blacklist (Radius check)
    - Weg durch infliiertes Hindernis (Straight-Line Check)
    - gain_m² == 0 (nichts Neues zu sehen)
```

**Kumulativer Gain:** Beim Sampling entlang einer Richtung wird der Gain kumulativ berechnet – weiter weg bedeutet mehr Gain nur wenn der direkte Weg auch wirklich neue Fläche aufdeckt.

#### 2.5 Planner (`hunter/planner.py`)

```python
class ExplorationPlanner:
    def next_waypoint(self, pose: Pose2D) -> Candidate | None:
        # 1. Reachability neu berechnen (nur wenn nötig)
        # 2. Kandidaten generieren
        # 3. Scoren und sortieren
        # 4. Besten zurückgeben (oder None wenn done())
    
    def register_block(self, stall_pose: Pose2D, goal: Candidate) -> None:
        # DynamicObstacle hinzufügen
        # Blacklist updaten
        # Consecutive-block-counter erhöhen
    
    def done(self) -> bool:
        return (
            self.coverage.fraction() >= self.config.coverage_threshold  # 0.95
            or self.consecutive_low_gain >= 3
            or self.n_waypoints >= 200
            or time.time() - self.start_time > 1800
        )
```

#### 2.6 Offline Simulation (`sim/`)

**`fake_nav.py`:**
```python
class FakeNav:
    # Simuliert straight-line Bewegung mit konfigurierbarer Geschwindigkeit
    # Kann "BLOCKED" simulieren bei vordefinierten blocked_discs

class FakePoseSource:
    # Shared Clock
    # Interpoliert Pose linear zwischen Waypoints
    # pose_at(ts_ns) über Ringpuffer
```

**`sim_main.py`:** CLI-Tool das den vollen Loop offline ausführt:
```bash
uv run litter-sim --circle 6   # 6m Radius, Lab-Grid
uv run litter-sim --rect 8 5   # 8m × 5m Rechteck
```
Gibt Coverage-Frame (PNG) und Trajektorie-Visualisierung nach `runs/sim/` aus.

**Verifikation Phase 2:**
```bash
uv run pytest tests/hunter/ -v
# Muss:
# - Raycasting: Wedge-Fläche ≈ analytisch ±10%
# - Wandschatten korrekt (Zellen hinter Wand nicht gesehen)
# - Coverage-Denominator: unknown/belegt/unerreichbar ausgeschlossen
# - Scoring: Gain > 0 → Wand-Kandidaten abgelehnt, Determinismus
# - Integration: Lab-Grid ≥95% Coverage in ≤60 Waypoints
# - BLOCKED-Szenario: Blacklist → Denominator schrumpft → terminiert trotzdem

uv run litter-sim --circle 6
# Visuell prüfen: Coverage-Frame soll Lab-Fläche gut abdecken
```

---

### Phase 3: Zenoh-Integration (Geschätzter Aufwand: ~1-2 Tage)

**Ziel:** System läuft gegen den echten Roboter. Erste Live-Tests.

#### 3.1 `zenoh_bridge.py`

Das zentrale Problem: Zenoh-Callbacks laufen in einem eigenen Thread, aber unser Code ist async. Lösung: `loop.call_soon_threadsafe`.

```python
class AsyncZenoh:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._session = zenoh.open(config)
    
    def subscribe_latest(self, key: str) -> asyncio.Queue[bytes]:
        # Gibt Queue(1) zurück, älteste Nachricht wird verworfen (LatestSlot-Pattern)
        # Zenoh-Callback: loop.call_soon_threadsafe(queue.put_nowait, payload)
    
    def subscribe_queue(self, key: str, maxsize: int = 0) -> asyncio.Queue[bytes]:
        # FIFO Queue, kein Drop
    
    async def publish_json(self, key: str, msg: BaseModel) -> None:
        # msg.model_dump_json() → Zenoh put
    
    async def close(self) -> None: ...
```

#### 3.2 `mission/pose_tracker.py`

```python
class ZenohPoseTracker:
    # Abonniert robodog/localization/pose
    # Hält latest pose + Ringpuffer (letzte 5 Sekunden)
    # distance_traveled_m: Integrator
    
    async def wait_first(self, timeout_s: float = 10.0) -> Pose2D:
        # Wirft RuntimeError wenn kein Pose ankommt → "Ist der Roboter an?"
    
    def pose_at(self, ts_ns: int) -> Pose2D:
        # Nearest-match im Ringpuffer (für Frame-Pairing im ValidationWorker)
```

#### 3.3 `hunter/navigator.py`

```python
class ZenohNavClient:
    # Publiziert nav/request (NavigationRequest)
    # Abonniert nav/status (~2 Hz)
    
    async def goto(
        self, target: Pose2D, max_speed: float = 0.4, must_stop: bool = True
    ) -> NavigationState:
        # Timeout = max(20s, 4 × abstand / speed)
        # Interpretiert: arrived_final → OK
        #                blocked/failed → BLOCKED
        #                Stille > timeout → TIMEOUT
    
    async def halt(self) -> None: ...
```

#### 3.4 `mission/orchestrator.py` (Skeleton)

```python
class MissionController:
    async def run(self, prompt: str) -> MissionReport:
        # 1. Pose warten (10s Timeout)
        # 2. Karte laden + inflaten + Flood-Fill
        # 3. SearchArea-Agent ausführen
        # 4. Suchgebiet rastern + --confirm anzeigen
        # 5. ValidationWorker starten (asyncio Task)
        # 6. CoverageTracker starten (asyncio Task, 5 Hz Pose-Update)
        # 7. Explorations-Loop:
        #    while not planner.done():
        #        waypoint = planner.next_waypoint(current_pose)
        #        result = await nav_client.goto(waypoint.pose)
        #        if result == BLOCKED:
        #            planner.register_block(stall_pose, waypoint)
        # 8. Nav stoppen, ValidationWorker drainieren (≤90s)
        # 9. Report erstellen, persistieren, ausgeben
```

**Bypass-Option `--area-circle R`:** Überspringt den SearchArea-Agent, nützlich für erste Live-Tests ohne Ollama.

**Verifikation Phase 3:**
```bash
# Live gegen robodog (3m Kreis):
uv run litter-mission "Search 3m around me" --area-circle 3 --no-llm-summary
# Roboter soll Kreis abfahren, Waypoints loggen, terminieren

# BLOCKED simulieren (Roboter manuell blockieren):
# → Erwartung: Retreat + Blacklist + Neuplanung + trotzdem Terminierung
```

---

### Phase 4: Validation Worker + Vision Agent + Datenbank (Geschätzter Aufwand: ~1-2 Tage)

**Ziel:** Erkannter Müll wird per LLM verifiziert und in der DB gespeichert.

#### 4.1 Einzige Änderung in `litter_detector` (`detector/main.py:211`)

```python
# Vorher:
self.frame_pub.put(payload)

# Nachher (frame_ts_ns ist bereits in Scope bei Zeile 168):
self.frame_pub.put(payload, attachment=str(frame_ts_ns).encode())
```

Das ermöglicht exaktes Frame-Track-Pairing im ValidationWorker.

#### 4.2 `validation/crops.py`

```python
def crop_detection(frame: np.ndarray, bbox: list[int], pad: float = 0.15) -> np.ndarray:
    # JPEG-Crop mit 15% Padding, mind. 32×32 px

def context_frame(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    # Herunterskaliertes Gesamt-Frame mit gezeichnetem BBox-Overlay
```

#### 4.3 `agents/validator.py` (Pydantic-AI)

```python
class LitterValidation(BaseModel):
    is_litter: bool
    category: Literal["plastic","paper","cardboard","metal","glass",
                       "organic","cigarette","textile","other"] | None
    confidence: float   # 0..1
    description: str

validation_agent = Agent(
    model=build_model(vision=True),   # Gemma auf Ollama Cloud
    output_type=LitterValidation,
    retries=2,
)

async def validate_detection(
    crop_jpeg: bytes,
    context_jpeg: bytes,
    pose: Pose2D,
    bearing_deg: float,
) -> LitterValidation:
    # BinaryContent für Bilder (media_type='image/jpeg')
    # PromptedOutput oder ToolOutput (konfigurierbar, kein NativeOutput auf Ollama Cloud)
```

#### 4.4 `agents/models.py`

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

def build_model(vision: bool = False) -> OpenAIChatModel:
    model_name = (
        settings.ollama_vision_model if vision else settings.ollama_text_model
    )
    provider = OpenAIProvider(
        base_url=f"{settings.ollama_base_url}/v1",
        api_key=settings.ollama_api_key,
    )
    return OpenAIChatModel(model_name, openai_client=provider)
```

**Modell:** `gemma4:31b-cloud` (bestätigt 2026-06-13 via `ollama run gemma4:31b-cloud`).  
**Wichtig:** Pydantic-AI nutzt den OpenAI-kompatiblen Endpunkt von Ollama (`/v1`). Kein `NativeOutput` – Ollama Cloud unterstützt das nicht.

#### 4.5 `validation/findings.py`

```python
# SQLite Schema:
# Tabelle: missions
CREATE TABLE missions (
    mission_id TEXT PRIMARY KEY,
    prompt TEXT,
    area_spec_json TEXT,
    started_at REAL,
    finished_at REAL,
    coverage_fraction REAL,
    distance_m REAL,
    n_waypoints INTEGER,
    n_blocked INTEGER,
    n_validated INTEGER,
    n_rejected INTEGER,
    n_errors INTEGER,
    report_json TEXT
);

# Tabelle: findings
CREATE TABLE findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT REFERENCES missions(mission_id),
    track_id INTEGER,
    status TEXT,           -- validated | rejected | error
    category TEXT,
    confidence REAL,
    description TEXT,
    robot_x REAL, robot_y REAL, robot_heading REAL,
    bearing_deg REAL,
    bbox_json TEXT,
    crop_path TEXT,
    context_path TEXT,
    first_seen_ns INTEGER,
    last_seen_ns INTEGER,
    n_observations INTEGER,
    raw_response TEXT,
    UNIQUE(mission_id, track_id)
);
```

Bilder werden gespeichert unter:
`runs/missions/<mission_id>/findings/track_<id>_crop.jpg`
`runs/missions/<mission_id>/findings/track_<id>_ctx.jpg`

#### 4.6 `validation/worker.py`

```python
class DetectionValidationWorker:
    # asyncio.Queue(16) mit Drop-Oldest-Strategie
    # 2 Consumer-Coroutinen
    
    def maybe_enqueue(self, track: TrackMsg, frame_jpeg: bytes, pose: Pose2D) -> None:
        # Readiness-Gate:
        # - n_observations >= 10
        # - bbox >= 32×32 und area_px >= 400
        # - bbox nicht am Rand (4px Margin)
        # - track_id noch nicht verarbeitet oder in-flight
        
    async def _consumer(self) -> None:
        # asyncio.wait_for(60s) pro LLM-Aufruf
        # 1 manueller Retry bei Timeout/5xx
        # Speichert Ergebnis (validated/rejected/error) in FindingsRepository
    
    async def drain(self, timeout_s: float = 90.0) -> None:
        # Wartet bis Queue leer oder Timeout
```

**Verifikation Phase 4:**
```bash
uv run pytest tests/agents/ -v
# Worker-Tests mit FakeAgent:
# - Gating (zu wenig Beobachtungen → nicht enqueued)
# - Frame-Pairing (mit und ohne Zeitstempel-Attachment)
# - Queue-Overflow (Drop-Oldest)
# - Dedup (gleicher Track → nicht nochmal)
# - Error-Row bei Timeout

# Live-Test: camera + detector starten, Müll-Objekt hinlegen
# uv run camera &
# uv run detector &
# uv run litter-mission "Search 2m around me" --no-exploration
# → findings.db prüfen
```

---

### Phase 5: SearchArea-Agent + Reporter + Full CLI + Docs (Geschätzter Aufwand: ~1 Tag)

**Ziel:** End-to-End funktioniert mit natürlichsprachlichem Prompt.

#### 5.1 `agents/search_area.py` (Pydantic-AI)

```python
class SearchAreaSpec(BaseModel):
    shape: Literal["circle", "rectangle", "polygon"]
    radius_m: float | None = None
    width_m: float | None = None
    depth_m: float | None = None
    polygon_points: list[tuple[float, float]] | None = None
    center_dx_m: float = 0.0      # Offset vom Roboter im Robot-Frame
    center_dy_m: float = 0.0
    rotate_with_robot: bool = True
    rationale: str = ""
    
    @model_validator(mode='after')
    def check_shape_fields(self) -> Self:
        # circle → radius_m gesetzt
        # rectangle → width_m + depth_m gesetzt
        # polygon → polygon_points gesetzt (min. 3 Punkte)

area_agent = Agent(
    model=build_model(vision=False),
    output_type=SearchAreaSpec,
    system_prompt="""
    You parse a user's search area request into a structured specification.
    The robot is at the origin, facing +x. Positive y is left.
    Examples:
    "Search 10m around me" → circle, radius_m=10
    "Search 5m in front of me" → rectangle, width_m=3, depth_m=5, center_dy_m=0, center_dx_m=2.5
    """,
    retries=2,
)
```

#### 5.2 `agents/reporter.py`

```python
class MissionReport(BaseModel):
    mission_id: str
    prompt: str
    area: SearchAreaSpec
    coverage_fraction: float
    reachable_target_m2: float
    duration_s: float
    distance_traveled_m: float
    n_waypoints: int
    n_blocked: int
    findings: list[FindingSummary]
    n_rejected: int
    n_errors: int
    summary_text: str | None = None   # optionaler LLM-Text

async def generate_summary(report: MissionReport) -> str:
    # Nur wenn --no-llm-summary nicht gesetzt
    # Reporter-Agent generiert lesbaren Zusammenfassungstext
```

#### 5.3 Full CLI (`mission/main.py`)

```bash
# Basis-Aufruf:
uv run litter-mission "Search 10m around me for litter"

# Optionen:
uv run litter-mission "Search 10m around me" \
    --map my_lab_grid.yaml \          # explizite Karte (Default: aus config)
    --confirm \                        # Zeigt Gebiet-Stats, wartet auf Enter
    --no-llm-summary \                 # Kein Reporter-Agent am Ende
    --area-circle 10                   # Überspringt SearchArea-Agent (Testing)
```

#### 5.4 Finale Verifikation

```bash
# Vollständiger Durchlauf:
uv run ruff check .
uv run ty check
uv run pytest -v

# Live End-to-End:
uv run litter-mission "Search 10m around me for litter" --confirm
# → Roboter fährt, verifiziert Müll, generiert Report
```

---

## 5. Offene Fragen / Risiken

### Kritisch

| # | Problem | Impact | Lösung |
|---|---|---|---|
| 1 | **Karten-YAML fehlt** | Phase 1–2 laufen auf Placeholder | Echte mm2grid YAML besorgen und `origin` + `resolution` eintragen |
| 2 | **Ollama Cloud Modell-Name** | Phase 4 schlägt fehl | Genauen Modell-Namen im Ollama Cloud Registry prüfen (z.B. `gemma3:4b` vs `gemma3-4b-it`) |
| 3 | **`/v1` Suffix** | Pydantic-AI kann Ollama nicht erreichen | Wahrscheinlich nötig (OpenAI-compat Endpunkt) — in `agents/models.py` isoliert, leicht zu ändern |

### Mittel

| # | Problem | Impact | Empfehlung |
|---|---|---|---|
| 4 | **Kein Tiefensensor** | Müll-Position ungenau (nur Pose + Bearing) | Erst später: RealSense Depth-Topics aus robodog |
| 5 | **Rotations-Sweep** | Weniger Coverage bei Waypoint-Ankunft | Als v2 Feature: ±120° Rotation an jedem Waypoint |
| 6 | **`camera_min_range_m=0.3`** | Falscher Blind-Spot | Echte Kamera-Montage-Neigung prüfen |

### Gering

| # | Problem | Empfehlung |
|---|---|---|
| 7 | Map-Frame ≠ Lokalisierungs-Frame | Statisches Transform-Feld in GridMap vorbereiten; `origin yaw ≠ 0` → NotImplementedError |
| 8 | Spatial-Dedup | Reporting-seitig: Funde < 1m, gleiche Kategorie → "possible duplicates" |

---

## 6. Pydantic-AI: Warum und Wie

Pydantic-AI wurde explizit vom User festgelegt. Hier das wichtigste Konzept:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Agent mit strukturiertem Output
agent = Agent(
    model=OpenAIChatModel(
        "gemma3:4b",
        openai_client=OpenAIProvider(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
    ),
    output_type=LitterValidation,  # Pydantic BaseModel
    retries=2,
)

# Aufruf
result = await agent.run(
    [
        UserContent([
            TextContent("Is this litter? Classify it."),
            BinaryContent(data=crop_jpeg, media_type="image/jpeg"),
            BinaryContent(data=context_jpeg, media_type="image/jpeg"),
        ])
    ]
)
validation: LitterValidation = result.output
```

**Wichtig für Ollama Cloud:**
- `PromptedOutput` (Standard): LLM wird im System-Prompt angewiesen, JSON zu erzeugen → funktioniert immer
- `ToolOutput`: Nutzt Function Calling → funktioniert nur wenn Ollama Cloud das unterstützt
- `NativeOutput` (z.B. OpenAI structured_output): **Nicht verfügbar auf Ollama Cloud**

Die Einstellung `agent_output_mode: str = "prompted"` in `config.py` steuert das.

---

## 7. Zenoh-Kommunikation: Gesamtbild

```
Zenoh Router (tcp/127.0.0.1:7447)
│
├── robodog/localization/pose    ◄── robodog-digipro (publiziert ~10 Hz)
│   OdometryState JSON
│
├── litter/tracked               ◄── detector (unser Repo, publiziert ~FPS)
│   TrackedMsg JSON
│
├── litter/frame                 ◄── detector (JPEG + timestamp_ns Attachment)
│   raw JPEG bytes
│
├── nav/request                  ──► robodog-digipro (akzeptiert NavigationRequest)
│   NavigationRequest JSON
│
└── nav/status                   ◄── robodog-digipro (publiziert ~2 Hz)
    NavigationStatus JSON
```

**AsyncZenoh Pattern (Thread-Safety):**
```python
# Zenoh callback läuft in Zenoh-Thread → niemals asyncio direkt aufrufen
def _on_sample(sample: zenoh.Sample) -> None:
    data = bytes(sample.payload)
    # RICHTIG: call_soon_threadsafe für asyncio-Loop
    self._loop.call_soon_threadsafe(self._queue.put_nowait, data)
    # FALSCH: await queue.put(data)  ← würde crashen
```

---

## 8. Testing-Strategie

### Ohne Roboter (immer möglich)

```bash
# Phase 1-2: Kern-Algorithmen
uv run pytest tests/hunter/ -v

# Phase 4: Worker + Agents (mit FakeAgent, kein echter LLM)
uv run pytest tests/agents/ -v -k "not live"

# Simulation (visuell):
uv run litter-sim --circle 6
# Prüfe runs/sim/ auf Coverage-Frames
```

### Mit Roboter (ab Phase 3)

```bash
# Live-Canary (braucht OLLAMA_API_KEY):
uv run pytest tests/agents/ -v -m live

# Live-Mission:
uv run litter-mission "Search 3m around me" --area-circle 3 --confirm
```

### Test-Datei-Struktur

```
tests/
├── hunter/
│   ├── test_raycast.py      # Wedge-Fläche, Wandschatten, Unknown blockiert
│   ├── test_coverage.py     # Denominator-Logik, Flood-Fill-Ausschluss
│   ├── test_reachability.py # Flood-Fill, Dynamic-Disc, Denominator-Shrink
│   ├── test_scoring.py      # Gain-Ordering, keine Kandidaten durch Wände
│   ├── test_raster.py       # Circle/Rect/Polygon vs. analytische Fläche
│   └── test_integration.py  # Voll-Loop auf Lab-Grid: ≥95% in ≤60 Waypoints
└── agents/
    ├── test_search_area.py  # TestModel-Smoke + FunctionModel-Szenarien
    ├── test_validator.py    # TestModel-Smoke, BinaryContent wird übergeben
    ├── test_worker.py       # Gating, Frame-Pairing, Drop-Oldest, Dedup, Errors
    └── conftest.py          # tmp_path für findings.db, FakeAgent fixtures
```

---

## 9. Empfohlene Reihenfolge (Zusammenfassung)

```
Woche 1:
  Tag 1:  Phase 1 – pyproject + config + interfaces + GridMap + raster + YAML
  Tag 2:  Phase 2a – raycast + coverage (Tests schreiben + grün bekommen)
  Tag 3:  Phase 2b – reachability + scoring + planner (Tests grün)
  Tag 4:  Phase 2c – fake_nav + sim_main (litter-sim visuell testen)

Woche 2:
  Tag 1:  Phase 3 – zenoh_bridge + pose_tracker + nav_client + orchestrator
  Tag 2:  Phase 3 Live-Test (robodog aktiv)
  Tag 3:  Phase 4 – crops + validator + models + findings + worker
  Tag 4:  Phase 4 Worker-Tests + Live-Canary + detector Änderung

Woche 3:
  Tag 1:  Phase 5 – search_area agent + reporter + full CLI
  Tag 2:  End-to-End Test (gesamt)
```

---

## 10. Schnellstart: Was als nächstes tun?

```bash
# 1. Abhängigkeiten ergänzen (pydantic-ai, pyyaml, scipy)
# 2. my_lab_grid.yaml anlegen
uv run python -c "from PIL import Image; img=Image.open('my_lab_grid.png'); print(img.size)"
# → Gibt (405, 225) → YAML origin = [-10.125, -5.625, 0.0] bei 0.05 m/px

# 3. Paket anlegen
mkdir -p src/litter_agents/{interfaces,mapping,hunter,validation,agents,mission,sim}

# 4. Phase 1 implementieren und verifizieren
uv sync
uv run python -c "from litter_agents.mapping.provider import FileMapProvider; \
  m = FileMapProvider().load('my_lab_grid.yaml'); \
  print(f'Free cells: {(m.data == 0).sum()}')"
# → Soll ~8500 ausgeben
```

Der wichtigste erste Meilenstein ist `uv run litter-sim --circle 6` – sobald der ohne echten Roboter durchläuft und ≥95% Coverage zeigt, ist das algorithmische Kernrisiko eliminiert.