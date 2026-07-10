# Fahrplan: Litter Detection UI

## Architektur-Überblick

```
Browser (React + TypeScript)
  │  WS /ws/camera    → JPEG-Frames live (Zenoh camera/frame)
  │  WS /ws/state     → Pose, Waypoints, Nav-Status live
  │  REST /api/*      → Findings CRUD, Mission starten, Karte
  │
FastAPI Backend  (src/litter_ui/)
  │  ↔ FindingsRepository   (src/litter_agents/validation/findings.py)
  │  ↔ MissionController    (src/litter_agents/mission/orchestrator.py)
  │  ↔ AsyncZenoh           (src/litter_agents/zenoh_bridge.py)
  │
Zenoh Router (tcp/127.0.0.1:7447)
     camera/frame · litter/masked_frame
     robodog/localization/pose · nav/status · nav/request · litter/tracked
```

**Technologie-Stack:**
- Backend: FastAPI + uvicorn (Python, passt in bestehendes uv-Projekt)
- Frontend: Vite + React + TypeScript (im Verzeichnis `ui/`)
- Karte: react-leaflet mit `L.imageOverlay` auf Occupancy-Grid-PNG
- Echtzeit: WebSocket (FastAPI native)
- DB: direkte Nutzung des bestehenden `FindingsRepository` (sqlite3)

---

## Phase 1 — Backend-Scaffold & Findings-API

### Ziel
FastAPI-App mit vollständiger REST-API für Missions und Findings, Map-Auslieferung,
statische UI-Files. Noch kein Zenoh, noch kein WebSocket.

### Prompt

```
Lies vor dem Start folgende Dateien komplett durch:
- CLAUDE.md                                        (Projektübersicht, Commands)
- src/litter_agents/validation/findings.py         (FindingsRepository, FindingRow, Schema)
- src/litter_agents/interfaces/mission.py          (MissionReport, SearchAreaSpec, LitterCategory)
- src/litter_agents/config.py                      (AgentSettings, DB-Pfade)
- pyproject.toml                                   (Entry-Points, Dependencies)

Erstelle das Package src/litter_ui/ mit folgenden Dateien:

1. src/litter_ui/__init__.py  (leer)

2. src/litter_ui/app.py
   - FastAPI-App mit CORS (allow_origins=["*"] für lokale Entwicklung)
   - Einbinden aller Router (findings, missions, map)
   - `/health` GET-Endpunkt → {"status": "ok"}
   - Static-File-Mount: /ui → ui/dist/ (Verzeichnis muss nicht existieren, 
     mount nur wenn vorhanden)
   - main()-Funktion: startet uvicorn auf 0.0.0.0:8080
   - Lifespan-Context-Manager für spätere Zenoh-Initialisierung (noch leer)

3. src/litter_ui/routes/__init__.py  (leer)

4. src/litter_ui/routes/findings.py
   Router-Prefix: /api
   Endpunkte:
   - GET  /api/missions                    → Liste aller Missions (id, prompt, started_ns,
                                             finished_ns, coverage_fraction, distance_m,
                                             n_waypoints, status_counts)
   - GET  /api/missions/{mission_id}/findings?status=validated|rejected|error
                                           → Liste FindingRow als JSON
   - GET  /api/missions/{mission_id}/findings/{track_id}
                                           → einzelnes Finding
   - DELETE /api/findings/{mission_id}/{track_id}
                                           → Löscht Row aus findings-Tabelle
   - PATCH /api/findings/{mission_id}/{track_id}
                                           → Body: {category?, status?}
                                             Aktualisiert nur diese Felder
   - GET  /api/findings/{mission_id}/{track_id}/image?type=crop|context
                                           → Liefert Bilddatei (FileResponse)
                                             404 wenn image_path None oder Datei fehlt
   
   FindingsRepository aus AgentSettings().findings_db_path öffnen.
   Dependency-Injection per FastAPI Depends() für den Repository-Zugriff.

5. src/litter_ui/routes/map.py
   Router-Prefix: /api/map
   - GET /api/map/image   → Liefert das Occupancy-Grid-PNG (FileResponse)
                            Pfad aus AgentSettings().map_yaml_path (YAML lesen → image-Feld)
                            404 wenn Datei nicht gefunden
                            Aktuelle Karte: deb_lab_grid.png (über my_lab_grid.yaml referenziert)
   - GET /api/map/config  → Liefert origin [x,y,theta], resolution, width, height als JSON
                            Parsed aus dem YAML (pyyaml)
                            Aktuelle Werte: 336×594px, resolution=0.05m/px,
                            origin=(-9.6, -13.0) → Karte ist portrait-Format (Höhe > Breite)

6. Eintrag in pyproject.toml:
   litter-ui = "litter_ui.app:main"

Nach der Implementierung:
SELBSTKRITISCHE REVIEW — beantworte jeden Punkt explizit:
[ ] Laufen `uv run ruff check src/litter_ui/` und `uv run ty check` fehlerfrei?
[ ] Gibt DELETE wirklich 404 zurück wenn (mission_id, track_id) nicht existiert?
[ ] Gibt /api/findings/.../image einen klaren 404 zurück wenn image_path NULL in DB ist?
[ ] Öffnet jeder Request einen neuen DB-Handle oder wird er geteilt? 
    (check_same_thread=False ist gesetzt — trotzdem: sicherstellen dass kein paralleler
     Write auf demselben Connection-Objekt landet)
[ ] Sind alle Imports absolut (keine relativen Imports die außerhalb des Package brechen)?
[ ] Fehlt pyyaml in pyproject.toml dependencies? Falls ja: ergänzen.
[ ] Kann `uv run litter-ui` ohne Fehler starten (auch wenn runs/findings.db nicht existiert)?
```

---

## Phase 2 — Camera-WebSocket (Zenoh → Browser)

### Ziel
Live-Kamerabild vom Zenoh-Topic `camera/frame` (oder `litter/masked_frame`)
als JPEG-Stream über WebSocket an den Browser liefern.

### Prompt

```
Lies vor dem Start folgende Dateien komplett durch:
- src/litter_agents/zenoh_bridge.py       (AsyncZenoh, Bridge-Protocol, subscribe_queue)
- src/litter_detector/config.py           (TOPICS: camera.frame, detection.masked_frame)
- src/litter_agents/config.py             (build_zenoh_config)
- src/litter_ui/app.py                    (Lifespan-Context-Manager, bestehende Struktur)

Erweitere src/litter_ui/ um den Camera-WebSocket:

1. src/litter_ui/zenoh_state.py
   - Singleton-Modul, das im Lifespan-Context-Manager initialisiert wird
   - Hält: az: AsyncZenoh | None, camera_queue: asyncio.Queue[bytes] | None
   - Funktion startup(loop) → öffnet Zenoh, subscribed auf TOPICS.camera.frame
     Decode: sample.payload.to_bytes() direkt (sind bereits JPEG-Bytes)
     Queue-Size: 4 (drop-oldest, kein Stau im WS-Puffer)
   - Funktion shutdown() → az.close()

2. src/litter_ui/app.py — Lifespan ergänzen:
   - Ruft zenoh_state.startup() im Lifespan auf
   - Ruft zenoh_state.shutdown() beim Teardown auf
   - Zenoh-Fehler (Router nicht erreichbar) dürfen die App NICHT crashen:
     Exception fangen, loggen, camera_queue bleibt None

3. src/litter_ui/routes/ws.py
   WebSocket-Endpunkt GET /ws/camera:
   - Holt camera_queue aus zenoh_state
   - Wenn None (Zenoh nicht verbunden): schickt einmalig JSON {"error": "zenoh_unavailable"}
     und schließt die Verbindung sauber
   - Wenn verbunden: Loop — queue.get() → ws.send_bytes(jpeg_bytes)
   - Bei WebSocket-Disconnect: Loop beenden, keine Exception nach oben werfen
   - Timeout: wenn 5 s kein Frame kommt, schickt {"keepalive": true} als Text,
     damit der Browser den WS nicht schließt

4. Router in app.py einbinden (kein Prefix für WS-Routes)

SELBSTKRITISCHE REVIEW — beantworte jeden Punkt explizit:
[ ] Was passiert wenn der Zenoh-Router erst nach App-Start hochkommt?
    (startup() darf nicht blockieren — zenoh.open() in try/except wrappen)
[ ] Wird camera_queue korrekt als drop-oldest Queue implementiert?
    Prüfen: subscribe_queue in zenoh_bridge.py hat bereits drop-oldest-Logik →
    einfach subscribe_queue() nutzen statt eigene Queue
[ ] Werden WebSocket-Verbindungen korrekt bereinigt wenn der Client trennt?
    (asyncio.CancelledError fangen)
[ ] Laufen ruff und ty check fehlerfrei?
[ ] Gibt es einen Race-Condition zwischen startup() und einem frühen WS-Request?
    (zenoh_state.camera_queue None-Check ist die Absicherung — verifizieren)
[ ] Ist die Queue-Size 4 sinnvoll für 15fps? (4 Frames = ~270ms Puffer — ok)
```

---

## Phase 3 — State-WebSocket (Pose, Waypoints, Nav-Status)

### Ziel
Einen zweiten WebSocket `/ws/state` aufbauen, der Roboterpose,
Nav-Status und Waypoint-Updates als JSON an den Browser streamt.
Die gefahrene Route wird als wachsende Liste von Koordinaten akkumuliert.

### Prompt

```
Lies vor dem Start folgende Dateien komplett durch:
- src/litter_agents/mission/pose_tracker.py     (ZenohPoseTracker, OdometryState-Decode)
- src/litter_agents/interfaces/robodog.py       (Pose2D, OdometryState, NavigationRequest,
                                                  NavigationSegment, NavigationStatus)
- src/litter_agents/config.py                   (ROBODOG_POSE_TOPIC, NAV_STATUS_TOPIC,
                                                  NAV_REQUEST_TOPIC)
- src/litter_ui/zenoh_state.py                  (bestehende Struktur erweitern)
- src/litter_ui/routes/ws.py                    (Camera-WS als Vorlage)

Hintergrund: Der NBV-Planner (neu seit dem letzten Update) sendet Multi-Leg-Pfade
an den Nav-Stack via `nav/request`. Jede NavigationRequest enthält `segments: list[NavigationSegment]`
mit je einem `target: Pose2D`. Diesen Pfad wollen wir auf der Karte als gestrichelte
Linie (geplante Route) anzeigen. Er kommt nicht vom Roboter-Pose-Stream, sondern
wird vom Missions-Planner gepublished → wir lesen ihn mit auf.

Erweitere den State-WebSocket:

1. src/litter_ui/zenoh_state.py — ergänzen:
   - pose_latest: Pose2D | None            (aktuellste Pose)
   - path_history: list[tuple[float,float]] (alle bisherigen x/y-Positionen, max 10000)
   - nav_status_latest: dict | None         (letzter Nav-Status als dict)
   - planned_path: list[tuple[float,float]] (aktueller Multi-Leg-Plan vom Planner, ersetzt sich
                                             bei jeder neuen NavigationRequest)
   - state_subscribers: set[asyncio.Queue]  (alle aktiven /ws/state Verbindungen)
   
   In startup():
   - Subscribe auf ROBODOG_POSE_TOPIC:
     Decode: OdometryState.model_validate_json(sample.payload.to_bytes())
     Handler: pose_latest setzen, (x,y) an path_history anhängen (max 10000),
              broadcast_state() aufrufen
   - Subscribe auf NAV_STATUS_TOPIC:
     Decode: NavigationStatus.model_validate_json(sample.payload.to_bytes())
     Handler: nav_status_latest = nav_status.model_dump(mode="json"),
              broadcast_state() aufrufen
   - Subscribe auf NAV_REQUEST_TOPIC ("nav/request"):
     Decode: NavigationRequest.model_validate_json(sample.payload.to_bytes())
     Handler: planned_path = [(s.target.x, s.target.y) for s in nav_request.segments]
              broadcast_state() aufrufen

   Broadcast-Funktion: broadcast_state() → baut ein Dict zusammen:
     {"pose": {x,y,theta} | null,
      "path_history": [[x,y], ...],    ← gefahrene Route (Browser rendert als Polyline)
      "planned_path": [[x,y], ...],    ← geplante Multi-Leg-Route (gestrichelt)
      "nav_status": {...} | null}
     Schickt es als JSON in alle Queues in state_subscribers (drop wenn voll)

2. src/litter_ui/routes/ws.py — ergänzen:
   WebSocket GET /ws/state:
   - Erstellt asyncio.Queue(maxsize=8)
   - Fügt sie zu state_subscribers hinzu
   - Schickt sofort den aktuellen State als initialen Snapshot
     (pose_latest, path_history, planned_path, nav_status_latest)
   - Loop: queue.get() → ws.send_text(json)
   - Bei Disconnect oder CancelledError: Queue aus state_subscribers entfernen (finally)

SELBSTKRITISCHE REVIEW — beantworte jeden Punkt explizit:
[ ] Kann path_history unbegrenzt wachsen? → max 10000 Einträge: älteste droppen
[ ] planned_path wird bei jeder neuen NavigationRequest überschrieben — korrekt,
    wir wollen immer nur den aktuellen Plan sehen
[ ] Thread-Safety: alle Handler laufen via call_soon_threadsafe im asyncio-Loop →
    state_subscribers-Modifikation und broadcast_state() laufen single-threaded. ✓
[ ] Was wenn beim Start der App noch keine Pose kam?
    Initialer Snapshot schickt pose=null → Browser muss das tolerieren.
[ ] Laufen ruff und ty check fehlerfrei?
[ ] Werden Queues aus state_subscribers auch bei WebSocket-Fehlern (nicht nur
    normalem Disconnect) entfernt? (finally-Block verwenden)
[ ] NAV_REQUEST_TOPIC: in src/litter_agents/config.py als Konstante prüfen
    (NAV_REQUEST_TOPIC = "nav/request")
[ ] NavigationStatus model_dump(mode="json"): datetime-Felder werden ISO-String
    → das ist gewünscht, Browser kann es parsen
```

---

## Phase 4 — React Frontend: Grundgerüst, Karte & Kamera

### Ziel
Vite + React + TypeScript Projekt im Verzeichnis `ui/` mit dem
Basis-Layout (3-Panel), funktionierendem Kamera-Stream und
interaktiver Leaflet-Karte mit Roboterposition und Litter-Markern.

### Prompt

```
Lies vor dem Start folgende Dateien komplett durch:
- CLAUDE.md                              (Projektübersicht)
- src/litter_ui/routes/findings.py       (API-Endpunkte und Response-Struktur)
- src/litter_ui/routes/map.py            (GET /api/map/config Response-Struktur)
- src/litter_agents/interfaces/mission.py (LitterCategory-Werte)

Erstelle das React-Frontend:

1. Initialisierung (Bash-Kommandos ausführen):
   cd ui && npm create vite@latest . -- --template react-ts
   npm install react-leaflet leaflet @types/leaflet

2. ui/src/main.tsx  — Standard Vite Entry, leaflet CSS importieren:
   import 'leaflet/dist/leaflet.css'

3. ui/src/App.tsx — 3-Panel CSS-Grid-Layout:
   Oben: [CameraPanel | MapPanel | ChatPanel]  (je 1fr, min-height: 400px)
   Unten: [FindingsGallery]                    (volle Breite, scrollable)
   Alle Panels als eigene Komponenten, Props via State-Hooks in App.tsx

4. ui/src/components/CameraPanel.tsx
   - useEffect: WebSocket zu ws://localhost:8080/ws/camera aufbauen
   - onmessage: wenn binary (Blob) → objectURL → <img src={url} />
                wenn text (keepalive/error JSON) → ignorieren
   - Reconnect-Logik: 3 s Timeout dann erneut verbinden
   - Statusanzeige: "Verbunden" / "Kein Signal" / "Zenoh nicht verfügbar"
   - img-Tag: object-fit: cover, volle Panel-Größe

5. ui/src/components/MapPanel.tsx
   - Beim Mount: GET /api/map/config → {origin_x, origin_y, resolution, width_px, height_px}
   - GET /api/map/image → als imageURL
   - MapContainer mit CRS.Simple (pixel-basiert, keine Geo-Koordinaten)
     Bounds in Weltkoordinaten (Meter):
       bottomLeft = [origin_y, origin_x]            ← Leaflet [lat,lng] = [y,x]
       topRight   = [origin_y + height_px*resolution, origin_x + width_px*resolution]
     Hinweis: aktuelle Karte ist portrait (336×594px, 16.8m×29.7m) → Bounds korrekt berechnen
   - ImageOverlay mit dem Kartenimage auf diese Bounds
   - Roboter-Marker (blauer Kreis, zIndex hoch) aus /ws/state pose
   - Gefahrene Route: path_history als grüne Polyline (Opacity 0.7)
   - Geplante Route: planned_path als orange gestrichelte Polyline (dashArray: "6 4")
     Wird bei jeder neuen NavigationRequest überschrieben
   - Litter-Marker: Props findings: FindingRow[] → rote Kreise auf robot_x/robot_y
     Popup: Kategorie, Konfidenz, Beschreibung
   - WebSocket /ws/state (StateMessage): pose, path_history, planned_path, nav_status
   - Proportionale Darstellung: Karte füllt Panel, Leaflet attributionControl=false

6. ui/src/types.ts — TypeScript-Interfaces:
   FindingRow, MissionRow,
   StateMessage: { pose: {x,y,theta}|null, path_history: [number,number][],
                   planned_path: [number,number][], nav_status: object|null }
   exakt passend zu den FastAPI/WS-Response-Modellen

7. ui/vite.config.ts — Dev-Server Proxy:
   /api/* und /ws/* → proxy zu http://localhost:8080
   (damit im Dev-Mode kein CORS-Problem entsteht)

SELBSTKRITISCHE REVIEW — beantworte jeden Punkt explizit:
[ ] CRS.Simple Koordinaten: Leaflet [lat,lng] = [y,x] = [robot_y, robot_x].
    Werden Pose, path_history und planned_path korrekt als [y,x] übergeben?
[ ] Karten-Bounds korrekt? Aktuelle Karte 336×594px, 0.05m/px, origin(-9.6,-13.0):
    bottomLeft=[origin_y, origin_x]=[-13.0,-9.6], topRight=[-13.0+29.7, -9.6+16.8]=[16.7, 7.2]
    → Bounds=[[-13.0,-9.6],[16.7,7.2]]
[ ] Wird der ImageOverlay-URL korrekt erzeugt (kein Memory-Leak durch
    unbegrenzte createObjectURL-Aufrufe)?
[ ] WebSocket-Reconnect: wird der alte WS vor dem Reconnect geschlossen?
[ ] Laufen `npm run build` (in ui/) und `npx tsc --noEmit` fehlerfrei?
[ ] Wenn /api/map/config 404 liefert (Karte nicht konfiguriert): zeigt die
    Karte eine sinnvolle Fehlermeldung statt zu crashen?
[ ] Werden Leaflet-Marker bei State-Updates korrekt neu positioniert
    (nicht dupliziert)?
[ ] Wird das ui/dist/ Verzeichnis nach `npm run build` korrekt von FastAPI
    als statische Files ausgeliefert (mount in app.py prüfen)?
[ ] Zeigt die geplante Route (planned_path) korrekt als gestrichelte Polyline?
    (dashArray: "6 4", Farbe orange, damit sie sich von der gefahrenen Route abhebt)
```

---

## Phase 5 — Findings-Galerie: Bilder, CRUD & Filter

### Ziel
Vollständige Findings-Galerie mit Bildvorschau, Filterfunktionen (Status, Kategorie),
Sortierung, Inline-Edit von Kategorie und Status, und Löschen mit Bestätigung.
Galerie ist mit der Karte verknüpft: Klick auf Marker öffnet das Finding.

### Prompt

```
Lies vor dem Start folgende Dateien komplett durch:
- src/litter_ui/routes/findings.py       (alle Endpunkte, PATCH-Body-Struktur)
- src/litter_agents/interfaces/mission.py (LitterCategory Literal-Werte)
- src/litter_agents/validation/findings.py (FindingRow-Felder, Status-Werte)
- ui/src/types.ts                        (bestehende TypeScript-Interfaces)
- ui/src/App.tsx                         (State-Struktur, wie Props weitergegeben werden)
- ui/src/components/MapPanel.tsx         (Marker-Klick-Callback ergänzen)

Implementiere die Findings-Galerie:

1. ui/src/components/FindingsGallery.tsx
   Props:
   - missionId: string | null
   - onFindingSelect: (trackId: number) => void  (für Karten-Sync)
   - highlightedTrackId: number | null           (von Karte markiert)
   
   State:
   - findings: FindingRow[]
   - filterStatus: 'all' | 'validated' | 'rejected' | 'error'
   - filterCategory: string (LitterCategory | 'all')
   - sortBy: 'confidence' | 'validated_at' | 'area_px'
   - sortDir: 'asc' | 'desc'
   - editingId: {mission_id, track_id} | null
   - deleteConfirm: {mission_id, track_id} | null

   Verhalten:
   - Beim Mount + bei missionId-Änderung: GET /api/missions/{id}/findings laden
   - Filter/Sort rein clientseitig (keine erneuten API-Calls)
   - Grid-Layout: 3–5 Spalten je nach Viewport-Breite (CSS grid auto-fill)
   
   Jede Finding-Card:
   - Thumbnail: <img src="/api/findings/{mission_id}/{track_id}/image?type=crop">
     Bei Klick: Modal mit context-image und allen Metadaten
   - Badge: Kategorie (farbkodiert per LitterCategory), Konfidenz als %
   - Status-Chip: validated=grün, rejected=rot, error=gelb
   - Inline-Edit-Button: öffnet Dropdown für Kategorie und Status-Toggle
     PATCH-Request beim Speichern, lokaler State-Update ohne Re-Fetch
   - Löschen-Button: zeigt inline Bestätigungs-Dialog ("Wirklich löschen?")
     DELETE-Request, Finding aus lokalem State entfernen
   - Highlight: wenn highlightedTrackId matched → Card bekommt Rahmen + scroll into view

2. ui/src/components/MissionSelector.tsx
   - GET /api/missions → Liste der Missions als Dropdown
   - Zeigt: Mission-ID, Datum, Anzahl Findings, Coverage %
   - Neueste Mission automatisch vorausgewählt
   - Platz in der App.tsx-Toolbar oben

3. (Optional) Debug-Frame-Anzeige:
   Der NBV-Planner schreibt während der Mission Render-Frames nach
   runs/missions/<mission_id>/debug/ (overview.png + Intervall-Frames).
   Diese können als eigener Tab "Debug Frames" in der Galerie angezeigt werden:
   - GET /api/missions/{mission_id}/debug → Liste der PNG-Dateien in diesem Verzeichnis
     (neuer Endpunkt in findings.py: listet Dateien via Path.glob("*.png"))
   - Thumbnails in einem eigenen Grid, Klick → Vollbild
   Dieser Punkt ist optional und kann übersprungen werden wenn die Zeit knapp ist.

4. MapPanel.tsx erweitern:
   - Marker-Klick → onFindingSelect(track_id) Callback aufrufen
   - highlightedTrackId-Prop: entsprechender Marker bekommt anderen Icon (gelb)

5. App.tsx verbinden:
   - selectedMissionId-State
   - highlightedTrackId-State  
   - findings-State (gemeinsam für Karte und Galerie)
   - findings nach Mission laden, an MapPanel und FindingsGallery übergeben

SELBSTKRITISCHE REVIEW — beantworte jeden Punkt explizit:
[ ] Wird der PATCH-Request korrekt gebaut? Body: nur die geänderten Felder
    (partial update), nicht das komplette FindingRow-Objekt
[ ] Funktioniert "scroll into view" korrekt wenn der highlighted Track
    durch Filter ausgeblendet ist? (Highlight nur wenn Card sichtbar)
[ ] Memory-Leak: werden img-Tags mit API-URLs korrekt gecacht vom Browser?
    (keine createObjectURL nötig hier, direkte URL ist fine)
[ ] Laufen `npm run build` und `npx tsc --noEmit` fehlerfrei?
[ ] Wird der lokale State nach DELETE korrekt aktualisiert, auch wenn ein
    Filter aktiv ist? (findings-Array filtern, nicht nur re-fetchen)
[ ] Kategorie-Dropdown zeigt genau die LitterCategory-Werte aus 
    mission.py: plastic, paper, cardboard, metal, glass, organic, 
    cigarette, textile, other — keine hardgecodeten anderen Strings
[ ] Ist der Delete-Confirm-Flow klickbar ohne Versehens-Löschung?
    (zwei Klicks erforderlich: Löschen-Button → Bestätigen)
```

---

## Phase 6 — Chat, Mission-Start & Mission-Log

### Ziel
Chat-Panel zum Starten von Missionen per Freitext-Prompt,
Live-Status-Stream während die Mission läuft,
und Missions-Selektor der sich nach Mission-Ende automatisch aktualisiert.

### Prompt

```
Lies vor dem Start folgende Dateien komplett durch:
- src/litter_agents/mission/orchestrator.py  (MissionController.run Signatur, MissionReport)
- src/litter_agents/interfaces/mission.py    (MissionReport-Felder)
- src/litter_agents/config.py               (AgentSettings, findings_db_path)
- src/litter_ui/app.py                      (Lifespan, bestehende Router)
- src/litter_ui/routes/findings.py          (Missions-Endpunkt als Vorlage)
- ui/src/App.tsx                            (selectedMissionId-State)

Implementiere Chat und Mission-Steuerung:

1. src/litter_ui/routes/missions.py — erweitern:
   
   Mission-State im Modul-Scope (kein Datenbankpersistenz nötig):
   _running_mission: asyncio.Task | None = None
   _mission_log: list[str] = []          (max 200 Einträge, älteste droppen)
   _log_subscribers: set[asyncio.Queue]  (SSE-Clients)
   
   POST /api/mission/start
   Body: {"prompt": str, "circle_radius_m": float | None}
   - 409 wenn bereits eine Mission läuft (_running_mission und not done)
   - Erstellt asyncio.Task der MissionController().run(prompt) aufruft
   - MissionController-Logs via loguru-Sink in _mission_log einspeisen +
     an alle _log_subscribers broadcasten
   - Returns: {"mission_id": str, "status": "started"}
   
   POST /api/mission/stop
   - Cancelt _running_mission Task
   - Returns: {"status": "stopped"}
   
   GET /api/mission/status
   - Returns: {"running": bool, "mission_id": str | None, "log_tail": list[str]}
     (letzte 50 Log-Einträge)
   
   GET /api/mission/log  (Server-Sent Events)
   - SSE-Endpunkt: neue Log-Zeilen als "data: {line}\n\n"
   - Client schließt Verbindung wenn Mission endet (schick "data: __END__\n\n")
   - Implementierung: StreamingResponse mit async generator
   
   Wichtig: MissionController.run() ist async und läuft im selben Event-Loop.
   asyncio.create_task() nutzen, NICHT asyncio.run() oder ThreadPoolExecutor.
   
   Hinweis zu Secrets: AgentSettings lädt seit dem letzten Update automatisch aus
   einer .env-Datei im Repo-Root (SettingsConfigDict env_file=".env"). Das bedeutet:
   OLLAMA_API_KEY muss nicht mehr als Env-Var gesetzt werden, sondern kann in .env
   stehen. Das ChatPanel muss keinen API-Key-Input anzeigen — der Key wird serverseitig
   aus .env geladen.

2. ui/src/components/ChatPanel.tsx
   State:
   - prompt: string
   - isRunning: boolean
   - logLines: string[]
   - circleRadius: number (Default: 5)
   
   UI:
   - Textarea für den Mission-Prompt (Placeholder: "Suche 10m um mich nach Müll")
   - Optional: Zahlenfeld "Radius (m)"
   - Start-Button: POST /api/mission/start
     Disabled wenn isRunning
   - Stop-Button: POST /api/mission/stop
     Nur sichtbar wenn isRunning
   - Log-Anzeige: scrollbares div, auto-scroll zu neuestem Eintrag
     EventSource auf /api/mission/log
     Bei "__END__": EventSource schließen, isRunning=false
   
   Nach Mission-Ende: App.tsx informieren → MissionSelector neu laden
   (Callback-Prop: onMissionComplete())

3. App.tsx final verkabeln:
   - onMissionComplete → ruft GET /api/missions neu ab → selectedMissionId 
     auf neue Mission setzen → findings neu laden

SELBSTKRITISCHE REVIEW — beantworte jeden Punkt explizit:
[ ] Läuft MissionController.run() wirklich im asyncio-Event-Loop und blockiert
    den FastAPI-Request-Handler nicht?
    (asyncio.create_task() korrekt, NICHT await im Request-Handler)
[ ] loguru-Sink: wird die Sink nach Mission-Ende wieder entfernt?
    (loguru.logger.remove(sink_id) im Task-Finally-Block)
[ ] SSE-Endpunkt: wird die Queue aus _log_subscribers beim Client-Disconnect
    entfernt? (try/finally um den async-generator)
[ ] 409-Logik: Task.done() prüfen — ein abgeschlossener Task zählt nicht als
    "läuft noch"
[ ] EventSource in React: wird sie bei Component-Unmount geschlossen?
    (useEffect cleanup: eventsource.close())
[ ] Laufen `npm run build`, `npx tsc --noEmit` und
    `uv run ruff check src/litter_ui/` fehlerfrei?
[ ] End-to-End-Test simulieren: Was passiert wenn Zenoh-Router fehlt aber
    /api/mission/start aufgerufen wird? MissionController wartet auf Pose-Timeout
    → Task läuft, Status zeigt "running" → nach Timeout kommt MissionReport
    mit 0 Findings. Kein Crash der App.
[ ] Gibt es einen Entry-Point-Befehl `uv run litter-ui` der sowohl Backend
    startet als auch den ui/dist/ Build ausliefert?
    Falls ui/dist/ nicht existiert: klare Fehlermeldung mit Hinweis auf 
    `cd ui && npm run build`
```

---

## Abschluss-Check (nach Phase 6)

Führe nach Abschluss aller Phasen folgende Checks durch:

```bash
# Type-Check Python
uv run ty check

# Linting Python
uv run ruff check src/litter_ui/

# Type-Check + Build TypeScript
cd ui && npx tsc --noEmit && npm run build

# Smoke-Test Backend (ohne Zenoh)
uv run litter-ui &
curl http://localhost:8080/health
curl http://localhost:8080/api/missions
curl http://localhost:8080/api/map/config
```

**Erwartete Ergebnisse:**
- `/health` → `{"status": "ok"}`
- `/api/missions` → `[]` (leere Liste wenn keine DB existiert, kein 500)
- `/api/map/config` → Karten-Metadaten oder klares 404
- Backend startet ohne Fehler auch wenn Zenoh-Router nicht läuft
