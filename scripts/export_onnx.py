"""Export a trained .pth state-dict to a self-contained .onnx file.

Usage:
    uv run python scripts/export_onnx.py --arch resnet34 --pth models/best_resnet34.pth
    uv run python scripts/export_onnx.py --arch efficientnetb4 --pth models/best_efficientnetb4.pth

The produced .onnx is architecture-agnostic at load time, so the detector
doesn't need access to the model class.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "auto-research"))
from train import (  # noqa: E402
    CROP_SIZE,
    DROPOUT,
    EfficientNetB3UNet,
    EfficientNetB4UNet,
    ResNet34UNet,
    ResNet50UNet,
)

ARCHS: dict[str, type[torch.nn.Module]] = {
    "resnet34": ResNet34UNet,
    "resnet50": ResNet50UNet,
    "efficientnetb3": EfficientNetB3UNet,
    "efficientnetb4": EfficientNetB4UNet,
}


def export(arch: str, pth: Path, out: Path | None, opset: int) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARCHS[arch](dropout=DROPOUT).to(device).eval()
    model.load_state_dict(torch.load(pth, map_location=device))

    onnx_path = out or pth.with_suffix(".onnx")
    dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        external_data=False,
    )

    with torch.inference_mode():
        torch_out = model(dummy).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.cpu().numpy()})[0]
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    torch_mask = (1 / (1 + np.exp(-torch_out)) > 0.5)
    onnx_mask = (1 / (1 + np.exp(-onnx_out)) > 0.5)
    mask_agreement = float((torch_mask == onnx_mask).mean())
    print(
        f"Exported {onnx_path} | max |torch - onnx| logits = {max_diff:.2e} | "
        f"mask agreement = {mask_agreement:.6f}"
    )
    if mask_agreement < 0.999:
        raise RuntimeError(
            f"ONNX mask disagrees with PyTorch on {(1-mask_agreement)*100:.2f}% of pixels; "
            "export is not numerically equivalent."
        )
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=sorted(ARCHS))
    parser.add_argument("--pth", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Output .onnx path (default: same name as --pth)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.arch, args.pth, args.out, args.opset)
