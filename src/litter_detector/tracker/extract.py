from __future__ import annotations

import cv2
import numpy as np

from litter_detector.tracker.types import BBox, Detection


def clean_mask(mask: np.ndarray, erode_kernel: int = 0) -> np.ndarray:
    """Optional morphological cleanup before connected-component extraction.

    Erosion shrinks every foreground region by `erode_kernel/2` pixels on each
    side, which separates two pieces of litter that touch by a thin bridge into
    distinct blobs (and therefore distinct tracks). Set `erode_kernel <= 0` to
    skip — the function is a no-op then.
    """
    if erode_kernel <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel, erode_kernel))
    return cv2.erode(mask, kernel)


_APPEARANCE_HSV_BINS: tuple[int, int, int] = (8, 8, 8)


def _hsv_histogram(frame_bgr: np.ndarray, blob_pixels: np.ndarray) -> np.ndarray:
    """L2-normalised HSV color histogram over the pixels in `blob_pixels`.

    `blob_pixels` is a boolean H×W mask. The histogram has shape
    (H_bins * S_bins * V_bins,) — a single flat vector ready for cosine sim.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[blob_pixels]
    if pix.size == 0:
        n = _APPEARANCE_HSV_BINS[0] * _APPEARANCE_HSV_BINS[1] * _APPEARANCE_HSV_BINS[2]
        return np.zeros(n, dtype=np.float32)
    hist, _ = np.histogramdd(
        pix.astype(np.float32),
        bins=_APPEARANCE_HSV_BINS,
        range=((0, 180), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def mask_to_detections(
    mask: np.ndarray,
    min_area_px: int = 50,
    probs: np.ndarray | None = None,
    min_confidence: float = 0.0,
    frame_bgr: np.ndarray | None = None,
) -> list[Detection]:
    """Run connected components on a binary mask and return one detection per blob.

    The detector's segmentation mask collapses every "litter" pixel into a single
    foreground class — connected components is the cheapest way to recover
    per-object boxes without a dedicated detection head.

    Args:
        mask: H×W uint8, 0 = background, anything > 0 = foreground.
        min_area_px: drop blobs smaller than this many pixels (noise filter).
        probs: optional H×W float prob map (same shape as `mask`). When given,
            each detection gets `confidence = mean(probs)` over its blob pixels.
        min_confidence: drop blobs whose mean prob is below this value. Ignored
            when `probs` is None.

    Returns:
        Detections sorted by descending area (largest first).
    """
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape {mask.shape}")
    if probs is not None and probs.shape != mask.shape:
        raise ValueError(
            f"probs shape {probs.shape} must match mask shape {mask.shape}"
        )

    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    detections: list[Detection] = []
    # label 0 is the background — skip it
    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        blob_pixels = labels == label
        confidence = 1.0
        if probs is not None:
            confidence = float(probs[blob_pixels].mean())
            if confidence < min_confidence:
                continue

        appearance: np.ndarray | None = None
        if frame_bgr is not None:
            appearance = _hsv_histogram(frame_bgr, blob_pixels)

        detections.append(
            Detection(
                bbox=BBox(x=x, y=y, w=w, h=h),
                area_px=area,
                confidence=confidence,
                appearance=appearance,
            )
        )

    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections
