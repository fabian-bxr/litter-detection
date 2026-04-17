"""
train.py — CNN semantic segmentation training for litter detection.

This file IS modified by the agent. Everything is fair game:
  - Model architecture (encoder depth, decoder, attention, backbone, etc.)
  - Loss function (BCE, Dice, Focal, combo)
  - Optimizer and LR schedule
  - Data augmentation strategy
  - Batch size, image crop size
  - Any other technique the agent wants to try

Constraint: training stops after a fixed number of EPOCHS so every experiment
sees the same amount of data regardless of hardware. The primary metric logged
to MLflow is val_iou (higher is better).

Usage:
    uv run python train.py [--run-name NAME] [--epochs N] [--seed N]
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as tv_models
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ── Hyperparameters (edit freely) ─────────────────────────────────────────────

EPOCHS           = 20         # epochs of training per run (hardware-independent)
SEED             = 42         # RNG seed for reproducibility across runs
BATCH_SIZE       = 8
CROP_SIZE        = 384        # random-crop spatial resolution during training
ACCUM_STEPS      = 1          # gradient accumulation (1 = disabled)
LR               = 2e-4
WEIGHT_DECAY     = 1e-4
ENCODER_CHANNELS = [64, 128, 256, 512]   # U-Net encoder stage widths
DECODER_CHANNELS = [256, 128, 64, 32]    # U-Net decoder stage widths
DROPOUT          = 0.1
POS_WEIGHT       = 5.0        # BCEWithLogitsLoss pos_weight (handles class imbalance)
                               # override with value from data/meta.json
EMA_DECAY        = 0.999      # exponential moving average of model weights for eval
TTA_HFLIP        = True       # horizontal-flip test-time augmentation at validation

# ── Data ──────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR  = DATA_DIR / "masks"


def load_meta() -> dict:
    p = DATA_DIR / "meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


COPY_PASTE_PROB = 0.5   # Tier 1.2: probability of applying copy-paste per sample


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
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.3, hue=0.05, p=0.7),
                A.GaussNoise(p=0.2),
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
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

    def _load_raw(self, stem: str):
        image = np.array(Image.open(IMAGES_DIR / f"{stem}.jpg").convert("RGB"))
        mask  = (np.array(Image.open(MASKS_DIR / f"{stem}.png")) > 127).astype(np.uint8)
        return image, mask

    def _copy_paste(self, image: np.ndarray, mask: np.ndarray):
        """
        Sample a second (image, mask), find connected components in its mask,
        and paste each component onto the current image with small affine
        jitter (rotation + scale + translation). OR the pasted regions into
        the current mask. Operates in uint8 pixel space, before A.Normalize.
        """
        idx2 = random.randrange(len(self.stems))
        src_img, src_mask = self._load_raw(self.stems[idx2])

        # Resize source to match destination so component coordinates align
        h, w = image.shape[:2]
        src_img  = cv2.resize(src_img,  (w, h), interpolation=cv2.INTER_LINEAR)
        src_mask = cv2.resize(src_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        n_comp, comp_labels = cv2.connectedComponents(src_mask)
        if n_comp <= 1:
            return image, mask  # source has no foreground

        out_img  = image.copy()
        out_mask = mask.copy()
        for cid in range(1, n_comp):
            comp = (comp_labels == cid).astype(np.uint8)
            if comp.sum() < 32:        # skip tiny noise blobs
                continue
            # Per-component affine jitter
            angle = random.uniform(-25, 25)
            scale = random.uniform(0.7, 1.3)
            tx    = random.uniform(-0.15, 0.15) * w
            ty    = random.uniform(-0.15, 0.15) * h
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            warped_comp = cv2.warpAffine(
                comp, M, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            if warped_comp.sum() == 0:
                continue
            warped_img = cv2.warpAffine(
                src_img, M, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            region = warped_comp.astype(bool)
            out_img[region]  = warped_img[region]
            out_mask[region] = 1
        return out_img, out_mask

    def __getitem__(self, idx):
        stem = self.stems[idx]
        image, mask = self._load_raw(stem)

        if self.augment and random.random() < COPY_PASTE_PROB:
            image, mask = self._copy_paste(image, mask)

        mask = mask.astype(np.float32)
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-scale context.
    Applies dilated convolutions at multiple rates and fuses outputs.
    """
    def __init__(self, in_ch: int, out_ch: int, rates=(6, 12, 18)):
        super().__init__()
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Dilated 3x3 convolutions
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ) for r in rates
        ])
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Fusion projection: (1 + len(rates) + 1) * out_ch → out_ch
        n_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        branches = [self.conv1(x)]
        for dil in self.dilated:
            branches.append(dil(x))
        gap_out = self.gap(x)
        gap_out = F.interpolate(gap_out, size=(h, w), mode='bilinear', align_corners=False)
        branches.append(gap_out)
        return self.project(torch.cat(branches, dim=1))


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


class ResNet34UNet(nn.Module):
    """
    U-Net with a pretrained ResNet34 encoder.

    Skip connections come from ResNet34 feature stages:
      stem  (64 ch,  H/2)   — after maxpool (stride-2 conv + BN + ReLU)
      layer1 (64 ch,  H/4)
      layer2 (128 ch, H/8)
      layer3 (256 ch, H/16)
      layer4 (512 ch, H/32)  — used as bottleneck

    The decoder mirrors a 4-stage U-Net decoder.
    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """
    # Skip channel sizes from stem through layer3
    ENC_CHANNELS = [64, 64, 128, 256]   # stem, layer1, layer2, layer3
    BOTTLENECK_CH = 512                  # layer4

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        # ── Pretrained ResNet34 backbone ──────────────────────────────────
        backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)

        # Stem: conv1 + bn1 + relu (output: 64 ch, stride 2)
        self.stem_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stem_pool = backbone.maxpool   # stride 2 → H/4 total after stem+pool
        self.layer1 = backbone.layer1       # 64 ch,  H/4  (maxpool already applied)
        self.layer2 = backbone.layer2       # 128 ch, H/8
        self.layer3 = backbone.layer3       # 256 ch, H/16
        self.layer4 = backbone.layer4       # 512 ch, H/32  (bottleneck)

        # Freeze BN parameters in the backbone to preserve ImageNet stats
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # ── Decoder (4 stages) ────────────────────────────────────────────
        # Stage 1: upsample from 512 → 256, concat with layer3 skip (256) → 256
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256 + 256, 256, dropout)

        # Stage 2: upsample from 256 → 128, concat with layer2 skip (128) → 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 128, 128, dropout)

        # Stage 3: upsample from 128 → 64, concat with layer1 skip (64) → 64
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 64, 64, dropout)

        # Stage 4: upsample from 64 → 32, concat with stem skip (64) → 32
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(32 + 64, 32, dropout)

        # Final upsample ×2 to recover full input resolution
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(16, 16, dropout)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def _align(self, x, ref):
        """Bilinear resize x to match ref spatial dimensions if needed."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        s0 = self.stem_conv(x)         # 64 ch, H/2
        s1 = self.layer1(self.stem_pool(s0))  # 64 ch, H/4
        s2 = self.layer2(s1)           # 128 ch, H/8
        s3 = self.layer3(s2)           # 256 ch, H/16
        s4 = self.layer4(s3)           # 512 ch, H/32  (bottleneck)

        # Decoder
        d = self.up1(s4)
        d = self._align(d, s3)
        d = self.dec1(torch.cat([d, s3], dim=1))  # 256 ch, H/16

        d = self.up2(d)
        d = self._align(d, s2)
        d = self.dec2(torch.cat([d, s2], dim=1))  # 128 ch, H/8

        d = self.up3(d)
        d = self._align(d, s1)
        d = self.dec3(torch.cat([d, s1], dim=1))  # 64 ch, H/4

        d = self.up4(d)
        d = self._align(d, s0)
        d = self.dec4(torch.cat([d, s0], dim=1))  # 32 ch, H/2

        d = self.final_up(d)           # 16 ch, H/1
        d = self.final_conv(d)

        return self.head(d)            # 1 ch, H/1


class ResNet50UNet(nn.Module):
    """
    U-Net with a pretrained ResNet50 encoder.

    ResNet50 uses bottleneck blocks so channel counts differ from ResNet34:
      stem   (64 ch,  H/2)
      layer1 (256 ch, H/4)
      layer2 (512 ch, H/8)
      layer3 (1024 ch, H/16)
      layer4 (2048 ch, H/32)  — used as bottleneck

    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        # ── Pretrained ResNet50 backbone ──────────────────────────────────
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)

        # Stem: conv1 + bn1 + relu (output: 64 ch, stride 2)
        self.stem_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stem_pool = backbone.maxpool   # stride 2 → H/4 total after stem+pool
        self.layer1 = backbone.layer1       # 256 ch, H/4
        self.layer2 = backbone.layer2       # 512 ch, H/8
        self.layer3 = backbone.layer3       # 1024 ch, H/16
        self.layer4 = backbone.layer4       # 2048 ch, H/32  (bottleneck)

        # Freeze BN parameters in the backbone to preserve ImageNet stats
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # ── Decoder (4 stages) ────────────────────────────────────────────
        # Stage 1: upsample from 2048 → 512, concat with layer3 skip (1024) → 512
        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(512 + 1024, 512, dropout)

        # Stage 2: upsample from 512 → 256, concat with layer2 skip (512) → 256
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256 + 512, 256, dropout)

        # Stage 3: upsample from 256 → 128, concat with layer1 skip (256) → 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 256, 128, dropout)

        # Stage 4: upsample from 128 → 64, concat with stem skip (64) → 64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(64 + 64, 64, dropout)

        # Final upsample ×2 to recover full input resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(32, 32, dropout)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def _align(self, x, ref):
        """Bilinear resize x to match ref spatial dimensions if needed."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        s0 = self.stem_conv(x)                   # 64 ch, H/2
        s1 = self.layer1(self.stem_pool(s0))      # 256 ch, H/4
        s2 = self.layer2(s1)                      # 512 ch, H/8
        s3 = self.layer3(s2)                      # 1024 ch, H/16
        s4 = self.layer4(s3)                      # 2048 ch, H/32  (bottleneck)

        # Decoder
        d = self.up1(s4)
        d = self._align(d, s3)
        d = self.dec1(torch.cat([d, s3], dim=1))  # 512 ch, H/16

        d = self.up2(d)
        d = self._align(d, s2)
        d = self.dec2(torch.cat([d, s2], dim=1))  # 256 ch, H/8

        d = self.up3(d)
        d = self._align(d, s1)
        d = self.dec3(torch.cat([d, s1], dim=1))  # 128 ch, H/4

        d = self.up4(d)
        d = self._align(d, s0)
        d = self.dec4(torch.cat([d, s0], dim=1))  # 64 ch, H/2

        d = self.final_up(d)                       # 32 ch, H/1
        d = self.final_conv(d)

        return self.head(d)                        # 1 ch, H/1


class EfficientNetB3UNet(nn.Module):
    """
    U-Net with a pretrained EfficientNet-B3 encoder.

    Skip connections from EfficientNet-B3 feature stages:
      features[1]: 24 ch,  H/2  (stem after initial conv)
      features[2]: 32 ch,  H/4
      features[3]: 48 ch,  H/8
      features[5]: 136 ch, H/16
      features[7]: 384 ch, H/32  — used as bottleneck

    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        # ── Pretrained EfficientNet-B3 backbone ───────────────────────────
        backbone = tv_models.efficientnet_b3(
            weights=tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        features = backbone.features

        # Extract feature stages as separate modules
        self.stage0 = features[0]    # 40 ch,  H/2 (initial conv+bn+act)
        self.stage1 = features[1]    # 24 ch,  H/2 (MBConv1 blocks)
        self.stage2 = features[2]    # 32 ch,  H/4 (MBConv6 stride-2)
        self.stage3 = features[3]    # 48 ch,  H/8 (MBConv6 stride-2)
        self.stage4 = features[4]    # 96 ch,  H/16 (MBConv6 stride-2)
        self.stage5 = features[5]    # 136 ch, H/16 (MBConv6)
        self.stage6 = features[6]    # 232 ch, H/32 (MBConv6 stride-2)
        self.stage7 = features[7]    # 384 ch, H/32 (MBConv6)

        # Freeze BN parameters in the backbone to preserve ImageNet stats
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # ── Decoder (4 stages) ────────────────────────────────────────────
        # Stage 1: upsample from 384 → 136, concat with stage5 skip (136) → 256
        self.up1 = nn.ConvTranspose2d(384, 136, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(136 + 136, 256, dropout)

        # Stage 2: upsample from 256 → 128, concat with stage3 skip (48) → 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 48, 128, dropout)

        # Stage 3: upsample from 128 → 64, concat with stage2 skip (32) → 64
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 32, 64, dropout)

        # Stage 4: upsample from 64 → 32, concat with stage1 skip (24) → 32
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(32 + 24, 32, dropout)

        # Final upsample ×2 to recover full input resolution
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(16, 16, dropout)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def _align(self, x, ref):
        """Bilinear resize x to match ref spatial dimensions if needed."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        # Encoder
        s0 = self.stage0(x)          # 40 ch, H/2
        s1 = self.stage1(s0)         # 24 ch, H/2
        s2 = self.stage2(s1)         # 32 ch, H/4
        s3 = self.stage3(s2)         # 48 ch, H/8
        s4 = self.stage4(s3)         # 96 ch, H/16
        s5 = self.stage5(s4)         # 136 ch, H/16
        s6 = self.stage6(s5)         # 232 ch, H/32
        s7 = self.stage7(s6)         # 384 ch, H/32  (bottleneck)

        # Decoder
        d = self.up1(s7)
        d = self._align(d, s5)
        d = self.dec1(torch.cat([d, s5], dim=1))  # 256 ch, H/16

        d = self.up2(d)
        d = self._align(d, s3)
        d = self.dec2(torch.cat([d, s3], dim=1))  # 128 ch, H/8

        d = self.up3(d)
        d = self._align(d, s2)
        d = self.dec3(torch.cat([d, s2], dim=1))  # 64 ch, H/4

        d = self.up4(d)
        d = self._align(d, s1)
        d = self.dec4(torch.cat([d, s1], dim=1))  # 32 ch, H/2

        d = self.final_up(d)                       # 16 ch, H/1
        d = self.final_conv(d)

        return self.head(d)                        # 1 ch, H/1


class EfficientNetB4UNet(nn.Module):
    """
    U-Net with a pretrained EfficientNet-B4 encoder.

    Skip connections from EfficientNet-B4 feature stages:
      features[0]: 48 ch,  H/2   (initial conv+bn+act)
      features[1]: 24 ch,  H/2   (MBConv1 blocks)
      features[2]: 32 ch,  H/4   (MBConv6 stride-2)
      features[3]: 56 ch,  H/8   (MBConv6 stride-2)
      features[5]: 160 ch, H/16  (MBConv6)
      features[7]: 448 ch, H/32  — used as bottleneck

    BN layers in the backbone are frozen to preserve ImageNet statistics.
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        backbone = tv_models.efficientnet_b4(
            weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        features = backbone.features

        self.stage0 = features[0]    # 48 ch,  H/2
        self.stage1 = features[1]    # 24 ch,  H/2
        self.stage2 = features[2]    # 32 ch,  H/4
        self.stage3 = features[3]    # 56 ch,  H/8
        self.stage4 = features[4]    # 112 ch, H/16
        self.stage5 = features[5]    # 160 ch, H/16
        self.stage6 = features[6]    # 272 ch, H/32
        self.stage7 = features[7]    # 448 ch, H/32  (bottleneck)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # Stage 1: 448 → 160, concat stage5 (160) → 256
        self.up1  = nn.ConvTranspose2d(448, 160, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(160 + 160, 256, dropout)

        # Stage 2: 256 → 128, concat stage3 (56) → 128
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 56, 128, dropout)

        # Stage 3: 128 → 64, concat stage2 (32) → 64
        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 32, 64, dropout)

        # Stage 4: 64 → 32, concat stage1 (24) → 32
        self.up4  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(32 + 24, 32, dropout)

        # Final ×2 to full resolution
        self.final_up   = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(16, 16, dropout)
        self.head       = nn.Conv2d(16, 1, kernel_size=1)

    def _align(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        s0 = self.stage0(x)          # 48 ch,  H/2
        s1 = self.stage1(s0)         # 24 ch,  H/2
        s2 = self.stage2(s1)         # 32 ch,  H/4
        s3 = self.stage3(s2)         # 56 ch,  H/8
        s4 = self.stage4(s3)         # 112 ch, H/16
        s5 = self.stage5(s4)         # 160 ch, H/16
        s6 = self.stage6(s5)         # 272 ch, H/32
        s7 = self.stage7(s6)         # 448 ch, H/32

        d = self.up1(s7);  d = self._align(d, s5)
        d = self.dec1(torch.cat([d, s5], dim=1))   # 256 ch, H/16

        d = self.up2(d);   d = self._align(d, s3)
        d = self.dec2(torch.cat([d, s3], dim=1))   # 128 ch, H/8

        d = self.up3(d);   d = self._align(d, s2)
        d = self.dec3(torch.cat([d, s2], dim=1))   # 64 ch,  H/4

        d = self.up4(d);   d = self._align(d, s1)
        d = self.dec4(torch.cat([d, s1], dim=1))   # 32 ch,  H/2

        d = self.final_up(d)
        d = self.final_conv(d)
        return self.head(d)                         # 1 ch,   H/1


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """BCE (with label smoothing) + Lovász-Softmax (binary).

    Lovász directly optimises the IoU surrogate, which is the eval metric.
    Equal weighting (0.5 / 0.5) matches the recipe from the program plan.
    """
    def __init__(self, pos_weight: float = POS_WEIGHT, label_smoothing: float = 0.01):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.lovasz = smp.losses.LovaszLoss(mode="binary", from_logits=True)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets_smooth = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        return 0.5 * self.bce(logits, targets_smooth) + 0.5 * self.lovasz(logits, targets)


# ── EMA ───────────────────────────────────────────────────────────────────────

class ModelEma:
    """Exponential moving average of model parameters (Polyak averaging).

    Buffers (e.g. BN running stats) are copied directly each step.
    """
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)


@torch.no_grad()
def forward_with_tta(model: nn.Module, images: torch.Tensor,
                     hflip: bool = True) -> torch.Tensor:
    logits = model(images)
    if hflip:
        flipped = torch.flip(images, dims=[3])
        logits_f = torch.flip(model(flipped), dims=[3])
        logits = (logits + logits_f) / 2
    return logits


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(run_name: str, epochs: int, seed: int = SEED):
    set_seed(seed)
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
    # Tier 1 step 1: smp DeepLabV3+ with MiT-B0 encoder.
    # MiT-B0 (~3.7M params) is more sample-efficient than ImageNet ResNets on
    # small datasets; DeepLabV3+ adds ASPP for multi-scale context.
    model_name = "smp.DeepLabV3Plus(mit_b1)"
    model = smp.DeepLabV3Plus(
        encoder_name="mit_b1",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── EMA copy (Tier 1.3) ───────────────────────────────────────────────
    ema = ModelEma(model, decay=EMA_DECAY)

    # ── Optimizer + Schedule ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    # OneCycleLR step count must match actual optimizer.step() calls when
    # gradient accumulation is in use.
    optim_steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=optim_steps_per_epoch,
        epochs=epochs,
        pct_start=0.05,
    )
    criterion = CombinedLoss(pos_weight=pos_weight).to(device)

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    experiment_name = "litter-segmentation-v2"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(REPO_ROOT / "mlartifacts").as_uri(),
        )
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(log_models=True)
    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.set_tags({
            "model_class": model_name,
        })
        mlflow.log_params({
            "batch_size":        BATCH_SIZE,
            "accum_steps":       ACCUM_STEPS,
            "effective_batch":   BATCH_SIZE * ACCUM_STEPS,
            "crop_size":         CROP_SIZE,
            "lr":                LR,
            "weight_decay":      WEIGHT_DECAY,
            "encoder_channels":  "mit_b1-pretrained",
            "decoder_channels":  "DeepLabV3Plus",
            "dropout":           DROPOUT,
            "copy_paste_prob":   COPY_PASTE_PROB,
            "ema_decay":         EMA_DECAY,
            "tta_hflip":         TTA_HFLIP,
            "pos_weight":        pos_weight,
            "optimizer":         "AdamW",
            "scheduler":         "OneCycleLR",
            "loss":              "0.5*BCE+0.5*Lovasz",
            "total_params":      total_params,
            "device":            str(device),
            "epochs":            epochs,
            "seed":              seed,
        })

        t0 = time.time()
        step = 0
        best_val_iou = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_iou  = 0.0
            epoch_start = time.time()
            samples_seen = 0

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)

                logits = model(images)
                loss   = criterion(logits, masks)
                (loss / ACCUM_STEPS).backward()

                is_step = (batch_idx + 1) % ACCUM_STEPS == 0 \
                          or (batch_idx + 1) == len(train_loader)
                if is_step:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    ema.update(model)
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item()
                train_iou  += compute_iou(logits, masks)
                samples_seen += images.size(0)
                step += 1

            epoch_time = time.time() - epoch_start
            samples_per_s = samples_seen / max(epoch_time, 1e-6)

            # ── Validation (EMA weights + h-flip TTA) ─────────────────
            ema.module.eval()
            val_loss = 0.0
            val_iou  = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks  = masks.to(device,  non_blocking=True)
                    logits = forward_with_tta(ema.module, images, hflip=TTA_HFLIP)
                    val_loss += criterion(logits, masks).item()
                    val_iou  += compute_iou(logits, masks)

            n_train = len(train_loader)
            n_val   = len(val_loader)
            elapsed = time.time() - t0

            metrics = {
                "train_loss":        train_loss / max(n_train, 1),
                "train_iou":         train_iou  / max(n_train, 1),
                "val_loss":          val_loss   / max(n_val,   1),
                "val_iou":           val_iou    / max(n_val,   1),
                "epoch":             epoch,
                "elapsed_s":         elapsed,
                "lr":                scheduler.get_last_lr()[0],
                "samples_per_s":     samples_per_s,
            }
            mlflow.log_metrics(metrics, step=epoch)

            if val_iou / max(n_val, 1) > best_val_iou:
                best_val_iou = val_iou / max(n_val, 1)
                # Save EMA weights — those are what we evaluated and what we export.
                torch.save(ema.module.state_dict(), MODELS_DIR / "best_model.pth")

            print(
                f"epoch {epoch:3d}/{epochs}  "
                f"train_loss={metrics['train_loss']:.4f}  "
                f"train_iou={metrics['train_iou']:.4f}  "
                f"val_loss={metrics['val_loss']:.4f}  "
                f"val_iou={metrics['val_iou']:.4f}  "
                f"[{elapsed:.0f}s  {samples_per_s:.1f} samp/s]"
            )

        mlflow.log_metric("best_val_iou", best_val_iou)

        best_ckpt = MODELS_DIR / "best_model.pth"
        if best_ckpt.exists():
            # Reload best EMA weights for logging + ONNX export.
            ema.module.load_state_dict(torch.load(best_ckpt, map_location=device))
            mlflow.pytorch.log_model(
                ema.module,
                name="model",
                registered_model_name="litter-segmentation",
            )

            onnx_path = MODELS_DIR / "best_model.onnx"
            ema.module.eval()
            dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE, device=device)
            torch.onnx.export(
                ema.module,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                external_data=False,
            )
            mlflow.log_artifact(str(onnx_path))

        print(f"\nBest val_iou: {best_val_iou:.4f}")
        print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline",
                        help="MLflow run name")
    parser.add_argument("--epochs",   type=int, default=EPOCHS,
                        help="Number of training epochs (hardware-independent stop criterion)")
    parser.add_argument("--seed",     type=int, default=SEED,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    train(run_name=args.run_name, epochs=args.epochs, seed=args.seed)
