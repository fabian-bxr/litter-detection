### Prompt 1:
Das YOLO-Modell soll robuster werden und mehrere Modell-Varianten unterstützen.
- Baue eine Modell-Registry (`detector/registry.py`), über die verschiedene Checkpoints (`best_yolov8s_seg.pt`) konfigurierbar geladen werden können.
- Erweitere die Trainingsdaten über TACO hinaus (zusätzliche Waste-Datasets von HuggingFace/Roboflow ingestieren und mergen).
- Stelle Kompatibilität sicher: `model.py` und `main.py` so anpassen, dass der Detector mit beiden Modellvarianten läuft.

**Follow-Up Prompts:**
- Welche zusätzlichen öffentlichen Litter-Datasets eignen sich, und wie merge ich sie konfliktfrei mit TACO?

| Metric                      | Score                                                  |
|-----------------------------|--------------------------------------------------------|
| **Tool used**               | Claude Code                                            |
| **Error Rate**              | 4                                                      |
| **Code Quality**            | 4.5                                                    |
| **Discrepancy from Prompt** | 4.5                                                    |
| **Notes**                   | Datensatz-Ingestion-Skripte (`ingest_*.py`, `merge_roboflow.py`) |
---

### Prompt 2:
Das Modell erzeugt im Live-Betrieb viele False Positives, vor allem auf Schuhen. Ich möchte das systematisch reduzieren — auf Inferenz-, Trainings- und Datenebene.
- Inferenz: sinnvolle Confidence-/NMS-Defaults, Box- vs. Masken-Ausgabe prüfen.
- Daten: Hard Negatives (Schuhbilder als Hintergrund mit leerem Label) ins Training aufnehmen — Skript zum Beschaffen aus Open Images Footwear.
- Untersuche, ob ein Multiklassen-Detection-Modell (mehrere Material-Klassen) Schuhe strukturell besser abweist als das 1-Klassen-Seg-Modell. Mache einen empirischen Vergleich auf einem Schuh-Testset.

**Follow-Up Prompts:**
- Beim Hinzufügen vieler neuer Positiv-Daten (PlastOPol) steigen die Schuh-FPs wieder — woran liegt das und wie fixe ich es? *(→ Negatives proportional mitskalieren)*

| Metric                      | Score                                                                        |
|-----------------------------|------------------------------------------------------------------------------|
| **Tool used**               | Claude Code (Opus)                                                           |
| **Error Rate**              | 5                                                                            |
| **Code Quality**            | 4.5                                                                          |
| **Discrepancy from Prompt** | 5                                                                            |
| **Notes**                   | Started in Plan Mode; empirisch belegt: entscheidend sind Hard Negatives, nicht Multiklassen |
---