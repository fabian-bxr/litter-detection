from __future__ import annotations

import os
from pathlib import Path

import cv2
import mlflow.pytorch
import numpy as np
import torch
from torch import nn

INPUT_SIZE = 384
_SEG_INPUT_SIZE = 384
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_URI = str(REPO_ROOT / "models" / "best_yolo11s_seg.pt")
_MLFLOW_SCHEMES = ("models:/", "runs:/", "mlflow://")
_YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.05"))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "1280"))
_YOLO_DILATE = int(os.environ.get("YOLO_MASK_DILATE", "5"))


class _YoloSegModel(nn.Module):
    def __init__(self, weights_path: str, device: torch.device) -> None:
        super().__init__()
        from ultralytics import YOLO

        self._holder = [YOLO(weights_path)]
        self._device_arg = 0 if device.type == "cuda" else "cpu"
        self._last_tracks: list[dict] = []

    @property
    def yolo(self):
        return self._holder[0]

    def train(self, mode: bool = True) -> "_YoloSegModel":
        return self

    def last_tracks(self) -> list[dict]:

        return self._last_tracks

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[-2:]
        chw = tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
        rgb = np.clip((chw * _STD + _MEAN) * 255.0, 0, 255).astype(np.uint8)

        results = self.yolo.track(
            rgb,
            imgsz=YOLO_IMGSZ,
            conf=_YOLO_CONF,
            persist=True,
            device=self._device_arg,
            verbose=False,
        )
        res = results[0]

        mask = np.zeros((h, w), dtype=np.float32)
        if res.masks is not None and len(res.masks.data) > 0:
            md = res.masks.data.detach().cpu().numpy()  # [N, mh, mw], 0..1
            combined = (md.max(axis=0) > 0.5).astype(np.float32)
            if combined.shape != (h, w):
                combined = cv2.resize(
                    combined, (w, h), interpolation=cv2.INTER_NEAREST
                )
            mask = combined

        self._last_tracks = self._extract_tracks(res)

        if _YOLO_DILATE > 0 and mask.any():
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (_YOLO_DILATE, _YOLO_DILATE)
            )
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(np.float32)

        logits = (mask * 2.0 - 1.0) * 20.0
        return torch.from_numpy(logits).reshape(1, 1, h, w).to(tensor.device)

    @staticmethod
    def _extract_tracks(res) -> list[dict]:
        boxes = res.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().cpu().tolist()
        xywhn = boxes.xywhn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        if res.masks is not None and len(res.masks.data) == len(ids):
            md = res.masks.data.detach().cpu().numpy()
            areas = [float((m > 0.5).mean()) for m in md]
        else:
            areas = [float(bw * bh) for (_, _, bw, bh) in xywhn]
        tracks = []
        for i, tid in enumerate(ids):
            cx, cy, bw, bh = (float(v) for v in xywhn[i])
            tracks.append({
                "id": int(tid),
                "bbox_norm": [cx - bw / 2.0, cy - bh / 2.0, bw, bh],
                "conf": float(confs[i]),
                "area_frac": areas[i],
            })
        return tracks


def _is_yolo_uri(uri: str) -> bool:
    return uri.endswith(".pt") and not uri.startswith(_MLFLOW_SCHEMES)


def load_model(device: torch.device) -> tuple[nn.Module, str]:
    global INPUT_SIZE
    uri = os.environ.get("MLFLOW_MODEL_URI", DEFAULT_MODEL_URI)
    if _is_yolo_uri(uri):
        if not Path(uri).exists():
            raise FileNotFoundError(f"YOLO weights not found: {uri}")
        INPUT_SIZE = YOLO_IMGSZ
        model = _YoloSegModel(uri, device)
    else:
        INPUT_SIZE = _SEG_INPUT_SIZE
        model = mlflow.pytorch.load_model(uri, map_location=device)
    model.to(device).eval()
    return model, uri


def preprocess(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return tensor.to(device, non_blocking=True)


def postprocess(logits: torch.Tensor, out_hw: tuple[int, int]) -> np.ndarray:
    probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    mask_small = (probs > 0.5).astype(np.uint8) * 255
    h, w = out_hw
    return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colored = np.zeros_like(frame_bgr)
    colored[mask > 0] = (0, 0, 255)  # red in BGR
    return cv2.addWeighted(frame_bgr, 1.0, colored, 0.5, 0)