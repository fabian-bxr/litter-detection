
uv # First Lab KI-Systeme

Over all task is to build a robot that can detect litter and notify its operator.

This project was build with the autoresaerch idea of Andrew Karpathy: https://github.com/karpathy/autoresearch

The overall idea is to critically look at the experiments and progress the AI made, identify improvements and integrate a further improved version into a robot setup.

Other approaches fine-tune a yolo model: e.g. see for https://github.com/jeremy-rico/litter-detection

## 1 Student Task

- [Task Description](docs/student_task.md)
- [Context to this project](docs/explainer.md)

## Example images not in the dataset

| No litter                    | Litter                       |
|------------------------------|------------------------------|
| ![](docs/images/Image2.jpeg) | ![](docs/images/Image3.jpeg) |

## Autoresearch Content

> Note: There is already one good model in this repository. Thus you should be able to investigate the performance using the Analysis Notebook.

- [Analysis Notebook](auto-research/analysis.ipynb)
- [Instructions](auto-research/program.md)

## Setup

Init project:

```bash
uv sync
```

Content:

- There is a [analysis.ipynb](auto-research/analysis.ipynb) notebook to take a first look on the project and test the existing models.
- The project contains a mlflow project that stores the hole experiment and training history.
  Run the following command to launch the mlflow server and ui
  ```bash
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```
  >To upgrade outdated DB: `uv run mlflow db upgrade sqlite:///mlflow.db`

### Run Camera:
```bash
uv run camera [--source webcam/go2] [--id DeviceID]
```

### Run Detector:

By default the detector loads from the MLflow registry (`models:/litter-segmentation/latest`). To run against a shared `.onnx` checkpoint from `models/` instead, pass `--model`:

```bash
# MLflow (default)
uv run detector

# Local ONNX file
uv run detector --model models/best_resnet34.onnx
uv run detector --model models/best_efficientnetb4.onnx
```

You can also set `LITTER_MODEL_URI` to make a choice sticky for the shell. ONNX inference runs on CPU by default; GPU ONNX requires installing `onnxruntime-gpu` with a CUDA version matching the torch wheels.

### Object tracking

The detector also runs a SORT-style multi-object tracker on top of the segmentation mask. Connected components extract per-object bounding boxes from the mask each frame, a Kalman filter predicts where each known object should appear next, and IoU-based Hungarian assignment links new detections to existing tracks. Confirmed tracks are persisted to a local SQLite registry (`runs/objects.db` by default) with first-seen / last-seen timestamps, and republished over Zenoh on `litter/tracked`. Bounding boxes plus IDs are also drawn on `litter/masked_frame`.

Tunable via env vars (Pydantic Settings):

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRACKER_MIN_AREA_PX` | 50 | drop mask blobs below this many pixels (noise filter) |
| `TRACKER_IOU_THRESHOLD` | 0.3 | minimum IoU for a (detection, prediction) match |
| `TRACKER_MIN_HITS` | 3 | observations required before a track is emitted/persisted |
| `TRACKER_MAX_AGE` | 30 | frames a track may go unmatched before it's killed |
| `TRACKER_COUNT_MIN_OBSERVATIONS` | 10 | observations required before a track ticks the unique-objects counter (filters flickery blobs) |
| `TRACKER_MASK_ERODE_KERNEL` | 3 | square erosion kernel applied to the mask before extraction; separates touching blobs into distinct tracks (set 0 to disable) |
| `TRACKER_MIN_CONFIDENCE` | 0.6 | drop detections whose mean sigmoid prob (inside the blob) is below this value |
| `REGISTRY_DB_PATH` | `runs/objects.db` | SQLite path for the persistent object registry |

See [docs/tracking.md](docs/tracking.md) for the full architecture, lifecycle diagram, schema, and the multi-agent extension path.

### Distributing trained models

Training writes both `models/best_model.pth` (state-dict, for resume-training) and `models/best_model.onnx` (self-contained graph + weights, for distribution). To share a named checkpoint across machines, rename the `.onnx` to something descriptive and commit it directly:

```bash
# From an existing .pth:
uv run python scripts/export_onnx.py --arch resnet34 --pth models/best_resnet34.pth

git add models/best_resnet34.onnx
git commit -m "models: add best_resnet34.onnx"
```

The `.onnx` format is architecture-agnostic at load time, so adding a new architecture (EfficientNet, MobileNet, …) needs no changes to the detector. Keep the number of tracked model files small — each update bakes the full binary into git history.

### Run Grafana OTel LGTM 
```
cd docker
docker compose up -d
```


## Additional Content

- [Experiment Tracking](https://mlflow.org/docs/latest/ml/getting-started/deep-learning/)