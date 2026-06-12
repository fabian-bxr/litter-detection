# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous CNN training for litter segmentation on the TACO dataset, designed to run on a Unitree Go2 robot. Uses the "autoresearch" approach (inspired by Karpathy) where an AI agent iterates on training experiments. The project also includes a real-time camera pipeline that publishes frames over Zenoh for litter detection.

## Commands

```bash
# Setup
uv sync

# Data preparation (downloads TACO dataset from HuggingFace, creates data/ directory)
uv run python auto-research/prepare.py

# Training (epoch-limited, logs to MLflow; also exports models/best_model.onnx)
uv run python auto-research/train.py [--run-name NAME] [--epochs N] [--seed N]

# Export an existing .pth state-dict to a single-file .onnx for distribution
uv run python scripts/export_onnx.py --arch resnet34 --pth models/best_resnet34.pth

# MLflow experiment UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Run camera publisher
uv run camera

# Run detector (defaults to MLflow registry; override with --model for a local ONNX file)
uv run detector
uv run detector --model models/best_resnet34.onnx

# Offline exploration sim (no robot/zenoh needed; renders frames to runs/sim/)
uv run litter-sim --circle 5
uv run litter-sim --circle 5 --block -10.7 -0.3 0.5   # simulate an unmapped obstacle

# Full litter-search mission (needs zenoh router, robodog stack, camera + detector)
uv run litter-mission "Search 10m around me for litter"
uv run litter-mission --circle 5 "manual area" --no-llm-summary

# Linting & type checking
uv run ruff check .
uv run ty check

# Tests
uv run pytest
```

## Architecture

**Training pipeline** (`auto-research/prepare.py`, `auto-research/train.py`): `prepare.py` downloads TACO from HuggingFace, converts COCO polygon annotations to binary masks (litter vs background), and writes to `data/` at the repo root. `train.py` trains a U-Net segmentation model with a configurable encoder/decoder, epoch-limited per run (hardware-independent), logging metrics (primary: `val_iou`) to MLflow (`mlruns/` at repo root). Best checkpoint is saved to `models/best_model.pth`. Both scripts resolve paths relative to the repo root via `__file__`, so CWD does not matter. `train.py` is designed to be freely modified by the autoresearch agent; `prepare.py` is not.

**Camera pipeline** (`src/litter_detector/`): A `CameraSource` ABC with two implementations — `WebcamSource` (local webcam via OpenCV/imutils) and `Go2Source` (receives frames from the Go2 robot via Zenoh subscription). `CameraPublisher` captures frames from either source, post-processes them (resize), JPEG-encodes, and publishes to Zenoh.

**Litter-search missions** (`src/litter_agents/`): Multi-agent system that drives a Unitree Go2 to autonomously search an area for litter. `mission/orchestrator.py` composes: a Pydantic-AI agent that parses the user's prompt into a `SearchAreaSpec` (`agents/search_area.py`), a deterministic exploration loop (`hunter/`: FoV raycasting coverage, information-gain waypoint scoring, flood-fill reachability with dynamic obstacle/blacklist handling on BLOCKED), and an async validation worker (`validation/worker.py`) that crops stable tracks from `litter/tracked` + `litter/frame`, has an Ollama Cloud vision agent verify/classify them, and persists findings to SQLite (`runs/findings.db`) with crops under `runs/missions/<id>/`. Talks to the robodog-digipro stack over Zenoh (`robodog/localization/pose`, `nav/request`, `nav/status` — schemas copied into `interfaces/robodog.py`). The static map is ROS map_server format (PNG+YAML from MOLA mm2grid; `my_lab_grid.yaml` has PLACEHOLDER metadata) behind a `MapProvider` abstraction. `sim/` runs the identical exploration loop offline against the map with fake nav/pose (`uv run litter-sim`); the `hunter/` package is pure numpy with no I/O.

**Communication**: All inter-component messaging uses [Zenoh](https://zenoh.io/). Topics are defined in `config.py` under the `TOPICS` constant. Zenoh router endpoint defaults to `tcp/127.0.0.1:7447` (override via `ZENOH_ROUTER_ENDPOINT` env var).

**Configuration**: `Settings` in `config.py` uses `pydantic-settings` for frame dimensions and provides static accessors for Zenoh config and topic definitions.

## Key Details

- Python 3.11, managed with `uv`
- Pre-trained model weights live in `models/`. `.onnx` files (self-contained graph + weights) are the distribution format and are committed directly to the repo; `.pth` state-dicts are kept for resume-training. `best_model.pth` is gitignored; named checkpoints are tracked. Each file must stay under GitHub's 100 MB hard limit — resnet34 and efficientnetb4 are already close (93 MB / 74 MB), so keep the set of checkpoints small.
- The detector (`uv run detector`) loads from MLflow by default (`models:/litter-segmentation/latest`); pass `--model path/to/model.onnx` or set `LITTER_MODEL_URI` to load a local ONNX instead. ONNX inference runs on CPU by default; for GPU ONNX inference install `onnxruntime-gpu` separately (CUDA version must match the torch wheels — beware of cu12/cu13 conflicts).
- The litter-agents LLM calls need `OLLAMA_API_KEY` (Ollama Cloud, OpenAI-compatible endpoint `https://ollama.com/v1`); model names and structured-output mode are configurable via `AgentSettings` in `src/litter_agents/config.py`. Tests marked `live` hit the real endpoint and are skipped without the key.
- The `auto-research/analysis.ipynb` notebook is for evaluating model performance
- Docs (including `explainer.md`, `student_task.md`, diagrams, README hero images) are under `docs/`
- `data/` and `mlruns/` are gitignored — regenerated by `prepare.py` and training runs
- Entry points `camera` and `detector` are registered in `pyproject.toml` under `[project.scripts]`