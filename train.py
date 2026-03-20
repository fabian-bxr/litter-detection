"""
train.py — CNN semantic segmentation training for litter detection.

This file IS modified by the agent. Everything is fair game:
  - Model architecture (encoder depth, decoder, attention, backbone, etc.)
  - Loss function (BCE, Dice, Focal, combo)
  - Optimizer and LR schedule
  - Data augmentation strategy
  - Batch size, image crop size
  - Any other technique the agent wants to try

Constraint: training stops after TIME_LIMIT seconds so every experiment is
comparable. The primary metric logged to MLflow is val_iou (higher is better).

Usage:
    uv run python train.py [--run-name NAME] [--time-limit SECONDS]
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ── Hyperparameters (edit freely) ─────────────────────────────────────────────

TIME_LIMIT       = 20 * 60   # seconds of training per run
BATCH_SIZE       = 8
CROP_SIZE        = 384        # random-crop spatial resolution during training
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
ENCODER_CHANNELS = [32, 64, 128, 256]   # U-Net encoder stage widths
DECODER_CHANNELS = [128, 64, 32, 16]    # U-Net decoder stage widths
DROPOUT          = 0.1
POS_WEIGHT       = 5.0        # BCEWithLogitsLoss pos_weight (handles class imbalance)
                               # override with value from data/meta.json

# ── Data ──────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR  = DATA_DIR / "masks"


def load_meta() -> dict:
    p = DATA_DIR / "meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


class LitterDataset(Dataset):
    def __init__(self, split: str, crop_size: int = CROP_SIZE, augment: bool = True):
        stems_file = DATA_DIR / f"{split}.txt"
        self.stems = [s.strip() for s in stems_file.read_text().splitlines() if s.strip()]
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(crop_size, crop_size),
                                    scale=(0.4, 1.0), ratio=(0.75, 1.33)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.3, hue=0.05, p=0.7),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=crop_size, width=crop_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        image = np.array(Image.open(IMAGES_DIR / f"{stem}.jpg").convert("RGB"))
        mask  = (np.array(Image.open(MASKS_DIR / f"{stem}.png")) > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)   # (3,H,W), (1,H,W)


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Double conv + BN + ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Vanilla U-Net for binary segmentation.
    Encoder depth and channel widths are controlled by ENCODER_CHANNELS /
    DECODER_CHANNELS — the agent is free to change these.
    """
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: list[int] = ENCODER_CHANNELS,
        decoder_channels: list[int] = DECODER_CHANNELS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert len(encoder_channels) == len(decoder_channels)

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for out_ch in encoder_channels:
            self.encoders.append(ConvBlock(ch, out_ch, dropout))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(ch, ch * 2, dropout)
        ch = ch * 2

        # ── Decoder ───────────────────────────────────────────────────────
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for enc_ch, dec_ch in zip(reversed(encoder_channels), decoder_channels):
            self.upconvs.append(nn.ConvTranspose2d(ch, enc_ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(enc_ch * 2, dec_ch, dropout))
            ch = dec_ch

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # handle odd spatial sizes
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """BCE + Dice loss (equal weight)."""
    def __init__(self, pos_weight: float = POS_WEIGHT):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def dice_loss(self, logits, targets, smooth: float = 1.0):
        probs = torch.sigmoid(logits)
        num   = 2 * (probs * targets).sum() + smooth
        den   = probs.sum() + targets.sum() + smooth
        return 1 - num / den

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice_loss(logits, targets)


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_iou(logits: torch.Tensor, masks: torch.Tensor,
                threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum().item()
    union = (preds + masks - preds * masks).sum().item()
    return inter / max(union, 1.0)


# ── Training loop ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(run_name: str, time_limit: int):
    device = get_device()
    print(f"Device: {device}")

    meta = load_meta()
    pos_weight = meta.get("pos_weight_suggestion", POS_WEIGHT)

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = LitterDataset("train", crop_size=CROP_SIZE, augment=True)
    val_ds   = LitterDataset("val",   crop_size=CROP_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True,
                              persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = UNet(
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer + Schedule ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=9999,          # effectively unlimited; time-budget controls stop
        pct_start=0.05,
    )
    criterion = CombinedLoss(pos_weight=pos_weight).to(device)

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_experiment("litter-segmentation")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size":        BATCH_SIZE,
            "crop_size":         CROP_SIZE,
            "lr":                LR,
            "weight_decay":      WEIGHT_DECAY,
            "encoder_channels":  str(ENCODER_CHANNELS),
            "decoder_channels":  str(DECODER_CHANNELS),
            "dropout":           DROPOUT,
            "pos_weight":        pos_weight,
            "optimizer":         "AdamW",
            "scheduler":         "OneCycleLR",
            "loss":              "BCE+Dice",
            "total_params":      total_params,
            "device":            str(device),
            "time_limit_s":      time_limit,
        })

        t0 = time.time()
        step = 0
        epoch = 0
        best_val_iou = 0.0

        while True:
            epoch += 1
            model.train()
            train_loss = 0.0
            train_iou  = 0.0

            for images, masks in train_loader:
                if time.time() - t0 > time_limit:
                    break

                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss   = criterion(logits, masks)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_iou  += compute_iou(logits, masks)
                step += 1

            else:
                # ── Validation ────────────────────────────────────────────
                model.eval()
                val_loss = 0.0
                val_iou  = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(device, non_blocking=True)
                        masks  = masks.to(device,  non_blocking=True)
                        logits = model(images)
                        val_loss += criterion(logits, masks).item()
                        val_iou  += compute_iou(logits, masks)

                n_train = len(train_loader)
                n_val   = len(val_loader)
                elapsed = time.time() - t0

                metrics = {
                    "train_loss": train_loss / max(n_train, 1),
                    "train_iou":  train_iou  / max(n_train, 1),
                    "val_loss":   val_loss   / max(n_val,   1),
                    "val_iou":    val_iou    / max(n_val,   1),
                    "epoch":      epoch,
                    "elapsed_s":  elapsed,
                    "lr":         scheduler.get_last_lr()[0],
                }
                mlflow.log_metrics(metrics, step=step)

                if val_iou / max(n_val, 1) > best_val_iou:
                    best_val_iou = val_iou / max(n_val, 1)
                    torch.save(model.state_dict(), "best_model.pth")
                    mlflow.log_artifact("best_model.pth")

                print(
                    f"epoch {epoch:3d}  "
                    f"train_loss={metrics['train_loss']:.4f}  "
                    f"train_iou={metrics['train_iou']:.4f}  "
                    f"val_loss={metrics['val_loss']:.4f}  "
                    f"val_iou={metrics['val_iou']:.4f}  "
                    f"[{elapsed:.0f}s]"
                )
                continue  # keep outer while going
            break  # inner for-loop broke early (time limit)

        mlflow.log_metric("best_val_iou", best_val_iou)
        print(f"\nBest val_iou: {best_val_iou:.4f}")
        print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",   default="baseline",
                        help="MLflow run name")
    parser.add_argument("--time-limit", type=int, default=TIME_LIMIT,
                        help="Training time budget in seconds")
    args = parser.parse_args()
    train(run_name=args.run_name, time_limit=args.time_limit)
