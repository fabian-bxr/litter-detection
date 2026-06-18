import os
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "yolo-litter")

if __name__ == "__main__":
    model = YOLO("yolo11s-seg.pt")

    model.train(
        data=str(REPO_ROOT / "dataset.yaml"),
        epochs=30,
        patience=12,
        imgsz=960,
        batch=16,
        project=str(REPO_ROOT / "runs" / "yolo"),
        name="litter-yolo11s-seg-768-aug",

        pretrained=True,
        optimizer="AdamW",
        device=0,
        freeze=10,
        cos_lr=True,
        lrf=0.1,
        warmup_epochs=3,
        copy_paste=0.3,
        mixup=0.1,
        weight_decay=0.0005,
        overlap_mask=True,
        mosaic=0.7,
        dropout=0.1,
    )

