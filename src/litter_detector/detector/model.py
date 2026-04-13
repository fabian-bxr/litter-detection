from __future__ import annotations

import os

import cv2
import mlflow.pytorch
import numpy as np
import torch
from torch import nn

INPUT_SIZE = 384
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_MODEL_URI = "models:/litter-segmentation/latest"


def load_model(device: torch.device) -> tuple[nn.Module, str]:
    uri = os.environ.get("MLFLOW_MODEL_URI", DEFAULT_MODEL_URI)
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
