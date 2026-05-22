from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import cv2
import mlflow.pytorch
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

INPUT_SIZE = 384
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_MODEL_URI = "models:/litter-segmentation/latest"
_MLFLOW_SCHEMES = ("models:/", "runs:/", "mlflow://")


class ModelRunner(Protocol):
    def infer(self, tensor: torch.Tensor) -> torch.Tensor: ...


class TorchRunner:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    def infer(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(tensor)


class OnnxRunner:
    def __init__(self, path: str, device: torch.device) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = device

    def infer(self, tensor: torch.Tensor) -> torch.Tensor:
        arr = tensor.detach().cpu().numpy()
        outputs = self.session.run(None, {self.input_name: arr})
        return torch.from_numpy(outputs[0]).to(self.device)


def load_model(uri: str, device: torch.device) -> tuple[ModelRunner, str]:
    if uri.startswith(_MLFLOW_SCHEMES):
        model = mlflow.pytorch.load_model(uri, map_location=device)
        return TorchRunner(model, device), uri
    path = Path(uri[len("file://"):] if uri.startswith("file://") else uri)
    if path.suffix == ".onnx":
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return OnnxRunner(str(path), device), str(path)
    raise ValueError(
        f"Unsupported model URI {uri!r}: expected MLflow URI "
        f"('models:/…', 'runs:/…') or a local '.onnx' file path."
    )


def resolve_default_uri() -> str:
    return os.environ.get("LITTER_MODEL_URI") or os.environ.get("MLFLOW_MODEL_URI") or DEFAULT_MODEL_URI


def preprocess(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return tensor.to(device, non_blocking=True)


def probs_from_logits(logits: torch.Tensor, out_hw: tuple[int, int]) -> np.ndarray:
    """Sigmoid → resize → full-resolution probability map (float32, H×W)."""
    probs_small = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    h, w = out_hw
    return cv2.resize(probs_small, (w, h), interpolation=cv2.INTER_LINEAR)


def binarize(probs: np.ndarray, prob_threshold: float = 0.5) -> np.ndarray:
    """Threshold a probability map to a uint8 binary mask (0 or 255)."""
    return (probs > prob_threshold).astype(np.uint8) * 255


def morph_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphological closing — fills small holes / bridges nearby blobs.

    Set kernel_size <= 1 to skip (returns input unchanged).
    """
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def postprocess(
    logits: torch.Tensor,
    out_hw: tuple[int, int],
    prob_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: probs → binarize. Returns (mask_uint8, probs_float).

    The detector loop does its own temporal smoothing + morphology between the
    two steps, so it bypasses this helper and calls `probs_from_logits` /
    `binarize` / `morph_close` directly.
    """
    probs = probs_from_logits(logits, out_hw)
    return binarize(probs, prob_threshold), probs


def overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colored = np.zeros_like(frame_bgr)
    colored[mask > 0] = (0, 0, 255)  # red in BGR
    return cv2.addWeighted(frame_bgr, 1.0, colored, 0.5, 0)
