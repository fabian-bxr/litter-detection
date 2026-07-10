"""
train_yolo_mc.py — Multiklassen-Müll-DETECTION-Modell trainieren (Ansatz A).

6 Material-Klassen (Plastik/Glas/Metall/Papier/Karton/Bio), Boxen statt Masken.
Getrennt vom Masken-/Ein-Klassen-Modell (train_yolo.py). Basis: yolov8s.pt
(Detection, NICHT -seg).

Start (erst auf Ansage):
    uv run python auto-research/train_yolo_mc.py
"""

import os
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "yolo-litter-mc")

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")  # Detection (Boxen), nicht Segmentierung

    model.train(
        data=str(REPO_ROOT / "dataset_mc.yaml"),
        epochs=30,
        patience=12,
        imgsz=640,
        batch=16,
        project=str(REPO_ROOT / "runs" / "yolo"),
        name="litter-yolov8s-mc-detect",

        pretrained=True,
        optimizer="AdamW",
        device=0,
        freeze=10,
        cos_lr=True,
        lrf=0.1,
        warmup_epochs=3,
        copy_paste=0.0,   # nur für Seg sinnvoll
        mixup=0.0,
        mosaic=0.5,
        close_mosaic=10,
        weight_decay=0.0005,
        dropout=0.1,
    )