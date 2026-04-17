import os
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "yolo-litter")

if __name__ == "__main__":
    model = YOLO("yolov8s-seg.pt")

    model.train(
        data=str(REPO_ROOT / "dataset.yaml"),
        epochs=20,
        patience=10,
        imgsz=640,
        batch=16,
        project=str(REPO_ROOT / "runs" / "yolo"),
        name="litter-yolov8s-seg",

        pretrained=True,
        optimizer="AdamW",
        device=0,
        freeze=5,
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=3,
        copy_paste=0.3,
        weight_decay=0.0005,
        overlap_mask=True,
        mosaic=1.0,
        dropout=0.1,
    )

