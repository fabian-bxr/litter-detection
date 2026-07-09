# YOLO Litter Detection — Pipeline & Änderungen

Dokumentiert die YOLO-Integration für die Litter-Detection: Trainings-Pipeline,
Live-Detector, Zenoh-Ausgabe (inkl. Tracker), SQLite-Registry, die Daten-
Ingest-Skripte und die Test-Ergebnisse.

Es gibt **zwei Modell-Stränge**:

1. **Ein-Klassen-Segmentierung (`litter`) — aktiver Default im Detector.**
   Liefert die Binärmaske + Tracks für die Go2-Pipeline. Seit dem letzten Stand
   auf einem **deutlich größeren, diverseren Datensatz** neu trainiert.
2. **Multiklassen-Detection (6 Material-Klassen) — Experiment „Ansatz A".**
   Getrennter Datensatz/Trainer (Boxen statt Masken). Untersucht „kategorisieren
   statt nur Maske", ist aber **nicht** in den Live-Detector verdrahtet (siehe
   Fazit unten).

## Architektur / Datenfluss

![YOLO Litter-Detection Pipeline](../docs/diagrams/yolo_pipeline.png)

**Kernidee:** Ein `_YoloSegModel`-Wrapper (`src/litter_detector/detector/model.py`)
kapselt das Ultralytics-YOLO hinter der bestehenden Segmentierungs-Schnittstelle
(`model(tensor) → logits`). Dadurch bleibt `main.py`/`postprocess` als Drop-in
erhalten und liefert dieselbe Binärmaske wie zuvor — zusätzlich werden pro Objekt
Tracks (mit stabilen IDs) exponiert und auf Zenoh sowie in eine SQLite-DB geschrieben.

## Was erledigt wurde

### 1. Umgebung (GPU)
- `pyproject.toml`: PyTorch von **Nightly-cu128 → Stable-cu128** umgestellt,
  `prerelease=allow` entfernt, **`uv.lock`** erzeugt.
- Ergebnis: `torch 2.11.0+cu128`, CUDA aktiv auf der RTX 5070 Ti (vorher CPU).

### 2. YOLO in den Detector eingebaut
- `_YoloSegModel`-Wrapper als Drop-in für das alte U-Net.
- Bugfix: YOLO ist ein `nn.Module` und überschreibt `.train()`; `model.eval()`
  löste versehentlich ein echtes Training aus. Behoben (YOLO nicht als
  nn.Module-Kind registrieren + `train()` neutralisiert).

### 3. Inferenz-Tuning (ohne Retraining)
- Inferenz-Auflösung **1280** (`YOLO_IMGSZ`)
- Confidence **0.25** (`YOLO_CONF`) — präzisionsfreundlicher Default gegen Fehlalarme
  (für maximalen Recall ggf. auf 0.05 senken)
- Flächenfilter **0.0005** (`YOLO_MIN_AREA`) — verwirft winzige Speck-Detektionen
- Masken-Dilatation **0** (`YOLO_MASK_DILATE`)
- TTA verworfen (von Segmentierungsmodellen nicht unterstützt)

### 4. Tracker-Ausgabe für Zenoh
- `yolo.track()` (ByteTrack, persistente IDs) statt `predict()`.
- Neues Topic **`litter/tracked`** mit `TrackedMsg`
  (`{timestamp_ns, tracks:[{id, bbox:[x,y,w,h], area_px, first_seen_ns,
  last_seen_ns, n_observations}]}`) — schema-kompatibel zu
  `litter-agent-V1`/`detection_tracking`.

### 5. SQLite-Object-Registry
- `src/litter_detector/detector/registry.py`, Schema identisch zu
  `detection_tracking` (Tabelle `objects`, Upsert pro Frame).
- Pfad via `registry_db_path` (Default `object_registry.db`, leer = aus).

### 6. False-Positives reduzieren — Daten statt nur Schwellen
Der größte Hebel gegen Fehlalarme (v. a. **Schuhe**) war nicht weiteres
Tuning, sondern **mehr und domänennähere Trainingsdaten** plus **Hard Negatives**:

- **`prepare.py`-Härtung** (`prepare_yolo.py`): degenerierte/winzige Polygone
  (< `MIN_POLY_AREA_FRAC`) werden verworfen; aus den TACO-Bildern werden
  **Background-Tiles ohne Annotation** geerntet und mit leerem Label gespeichert
  (Suffix `_bg`) → „leerer Boden ist kein Müll", ohne neue Bilder.
- **Echte Positiv-Daten zugemischt** (alle als Klasse 0 `litter`, Seg-kompatibel):
  - `ingest_hf_waste.py` → `moondream/waste_detection` (Boden/Umgebung, Suffix `_md`)
  - `ingest_plastopol_seg.py` → PlastOPol (Outdoor/Strand/Fluss, Suffix `_plast`)
  - `merge_roboflow.py` → beliebiges Roboflow-/YOLO-Dataset (Box→Polygon, Suffix `_rf`)
- **Schuh-Hard-Negatives** (`fetch_shoe_negatives.py`): echte Schuh-Bilder aus
  Open Images V7 (Klassen *Footwear/Sandal/Boot/High heels*) mit **leerem Label**
  ins Trainingsset (Suffix `_shoeneg`). Sauberkeitsfilter verwirft Bilder, die
  TACO-ähnliche Confusables (Flasche, Dose, Karton …) mit-annotiert haben — sonst
  würde ein leeres Label dem Modell beibringen, echten Müll zu unterdrücken.
  → bringt dem Modell direkt bei: **„Schuh ≠ Müll"**.
- Datensatz dadurch von **1275/225 → ~5335 train / 927 val** gewachsen.
- **`train_yolo.py`** entsprechend angepasst: imgsz **960**, `copy_paste=0.1`,
  `mosaic=0.5` + `close_mosaic=10`, `dropout=0.1`, `freeze=10` (gezähmte
  Augmentation = konservativeres Modell, weniger FPs bei leichtem Recall-Trade-off).

### 7. Multiklassen-Detection als Alternative untersucht (Ansatz A)
Idee: statt einer Klasse `litter` 6 **Material-Klassen** lernen (Plastik/Glas/
Metall/Papier/Karton/Bio), Boxen statt Masken — ein Schuh passt zu keiner Klasse
und sollte strukturell seltener als Müll erscheinen. Komplett getrennter
Datensatz (`data/yolo_mc`, `dataset_mc.yaml`) und Trainer:

- `prepare_yolo_mc.py` — Anker aus `keremberke/garbage-object-detection` (Studio/
  Förderband, exakt diese 6 Klassen).
- `ingest_realworld_mc.py` — TACO + moondream auf die 6 Klassen gemappt
  (Echtwelt-Domäne; mehrdeutige Kategorien & **Shoe verworfen**).
- `ingest_zerowaste_mc.py` — ZeroWaste-f (Förderband, CVPR 2022), korrelierte
  Video-Frames werden **strided** gedeckelt (Anti-Leakage).
- `balance_mc.py` — die stark überrepräsentierte **bio**-Klasse wird gedeckelt
  (nur Studio-Bilder ausgedünnt, Echtwelt bleibt).
- `train_yolo_mc.py` — `yolov8s.pt` (Detection, nicht -seg), 6 Klassen.

**Fazit:** Die Schuh-FPs lassen sich empirisch **am direktesten über Hard
Negatives im Ein-Klassen-Modell** lösen, nicht über den Multiklassen-Umweg.
Das Multiklassen-Modell bleibt daher ein eigenständiges Experiment und ist
**nicht** im Live-Detector aktiv.

## Test-Ergebnisse

### Ein-Klassen-Seg — Neutraining auf erweitertem Datensatz
(yolov8s-seg @ 960, 30 ep, **5681 train / 981 val**, Val-Metriken `best.pt`)

| Metrik | Box | Mask |
|--------|-----|------|
| Precision | 0.76 | 0.76 |
| Recall | 0.60 | 0.56 |
| mAP50 | **0.653** | 0.60 |
| mAP50-95 | **0.432** | 0.34 |

→ Gegenüber dem alten TACO-Subset (mAP50 ~0.44, „Datendecke") ein klarer Sprung
durch **mehr/domänennähere Daten + Hard Negatives** — der eigentliche Gewinn liegt
in den Daten, nicht im Modell-/Hyperparameter-Tuning.

#### Historie (altes TACO-Subset: 1275 train / 225 val, zur Einordnung)

| Lauf | mAP50 | Recall | mAP50-95 | Anmerkung |
|------|-------|--------|----------|-----------|
| yolov8s@640 (Baseline) | 0.433 | 0.41 | 0.271 | — |

→ Auf dem Subset pendelten alle Modell-/Trainings-Hebel um mAP50 ~0.44 =
**Datendecke** des Subsets.

### Multiklassen-Detection (Ansatz A)
(yolov8s @ 640, 30 ep, **9394 train / 3727 val**, 24 869 Instanzen)

| Klasse | mAP50 | Recall | mAP50-95 |
|--------|-------|--------|----------|
| **all** | **0.576** | 0.525 | **0.393** |
| plastik | 0.551 | 0.516 | 0.357 |
| glas | 0.712 | 0.656 | 0.519 |
| metall | 0.615 | 0.608 | 0.439 |
| papier | 0.556 | 0.495 | 0.407 |
| karton | 0.527 | 0.466 | 0.374 |
| bio | 0.496 | 0.407 | 0.262 |

→ `glas` am stärksten, `bio` (trotz Deckelung dominant & heterogen) am schwächsten.

### Recall-Inferenz (Val-Bilder, erkannte Instanzen)

| Anpassung | Effekt |
|-----------|--------|
| imgsz 384 → 960 (conf 0.15→0.10) | Instanzen 87 → 165 (~+90 %), Treffer 23/25 → 25/25 |
| conf 0.10 → 0.05 @1280 | Instanzen 243 → 401 (+65 %), Coverage 0.139 → 0.169 |
| imgsz 960 → 1280 @conf 0.05 | Instanzen 401 → 430 (+7 %), +2 ms |
| Masken-Dilatation 0/5/9 | Coverage 0.155 / 0.160 / 0.164 |

### Tracker & Registry
- Stabiler Stream (Bild 6× wiederholt): **4 Tracks mit persistenten IDs [1–4]**,
  `n_observations` zählt korrekt auf 6 hoch.
- SQLite: 4 Zeilen persistiert, `first_seen_ns` über Upserts erhalten,
  `n_observations` akkumuliert, Boxen/Flächen korrekt.

### Performance
- Inferenz **~30 ms/Frame** auf der RTX 5070 Ti (yolov8s-seg @ 1280).

## Konfiguration (Env-Vars)

| Variable | Default | Wirkung |
|----------|---------|---------|
| `MLFLOW_MODEL_URI` | `models/best_yolov8s_seg.pt` | aktives Modell (`.pt` = YOLO; `models:/…` = U-Net-Fallback) |
| `YOLO_IMGSZ` | `1280` | Inferenz-Auflösung |
| `YOLO_CONF` | `0.25` | Confidence-Schwelle (niedriger = mehr Recall, höher = weniger Fehlalarme) |
| `YOLO_MIN_AREA` | `0.0005` | Min. Maskenfläche als Bildanteil; verwirft winzige Speck-Fehlalarme (0 = aus) |
| `YOLO_MASK_DILATE` | `0` | Masken-Dilatation in px (0 = aus) — vergrößert Fläche |
| `YOLO_MASK_ERODE` | `0` | Masken-Erosion in px (0 = aus) — verkleinert Fläche |
| `YOLO_MASK_THRESH` | `0.5` | Binarisierungs-Schwelle (höher = enger; bei YOLO-Seg kaum wirksam) |
| `REGISTRY_DB_PATH` | `object_registry.db` | SQLite-Registry (leer = aus) |

## Daten-Ingest-Skripte (`auto-research/`)

| Skript | Ziel-Set | Quelle / Zweck | Suffix |
|--------|----------|----------------|--------|
| `prepare_yolo.py` | `data/yolo` (seg) | TACO → Seg; Polygon-Filter + Background-Tiles | `_bg` |
| `ingest_hf_waste.py` | `data/yolo` (seg) | moondream/waste_detection (Boden/Umgebung) | `_md` |
| `ingest_plastopol_seg.py` | `data/yolo` (seg) | PlastOPol (Outdoor/Strand/Fluss) | `_plast` |
| `merge_roboflow.py` | `data/yolo` (seg) | beliebiges Roboflow-/YOLO-Dataset (Box→Polygon) | `_rf` |
| `fetch_shoe_negatives.py` | `data/yolo` (seg) | Open Images V7 Schuhe als **Hard Negatives** | `_shoeneg` |
| `prepare_yolo_mc.py` | `data/yolo_mc` (mc) | keremberke/garbage-object-detection (6 Klassen) | — |
| `ingest_realworld_mc.py` | `data/yolo_mc` (mc) | TACO + moondream auf 6 Klassen gemappt | `_taco`/`_md` |
| `ingest_zerowaste_mc.py` | `data/yolo_mc` (mc) | ZeroWaste-f (Förderband, strided) | `_zw` |
| `balance_mc.py` | `data/yolo_mc` (mc) | bio-Klasse deckeln (Schieflage glätten) | — |

## Verwendung

### Ein-Klassen-Seg (aktiver Detector)
```bash
# 1) Daten bauen + anreichern (Reihenfolge beliebig nach prepare_yolo.py)
uv run python auto-research/prepare_yolo.py
uv run python auto-research/ingest_hf_waste.py --max-train 1500 --max-val 250
uv run python auto-research/ingest_plastopol_seg.py
uv run python auto-research/fetch_shoe_negatives.py --max 400
# optional: uv run python auto-research/merge_roboflow.py --src <pfad>

# 2) Training (yolov8s-seg @ 960, batch 16, 30 ep — Config in train_yolo.py)
uv run python auto-research/train_yolo.py

# 3) Live-Detector (nutzt models/best_yolov8s_seg.pt als Default)
uv run detector

# Noch weniger Fehlalarme (Recall ↔ Precision)
YOLO_CONF=0.35 uv run detector
```

### Multiklassen-Detection (Experiment)
```bash
uv run python auto-research/prepare_yolo_mc.py
uv run python auto-research/ingest_realworld_mc.py
uv run python auto-research/ingest_zerowaste_mc.py --max-train 1500 --max-val 300
uv run python auto-research/balance_mc.py --target 7000
uv run python auto-research/train_yolo_mc.py   # → runs/yolo/litter-yolov8s-mc-detect-*/
```

## Modelle (`models/`)

| Datei | Modell | Status |
|-------|--------|--------|
| `best_yolov8s_seg.pt` | yolov8s-seg @ 960 (erweiterter Datensatz) | **Default (aktiv)** |
| (Multiklasse) | yolov8s @ 640, 6 Klassen | nur in `runs/yolo/litter-yolov8s-mc-detect-*/weights/best.pt`, nicht im Detector |