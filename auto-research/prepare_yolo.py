"""
prepare_yolo.py — Download TACO and convert to YOLO segmentation format.

Two false-positive-reducing steps are applied during conversion:
  #3  Degenerate/tiny polygons (< MIN_POLY_AREA_FRAC of the image) are dropped,
      so the model is not trained on noisy speck-annotations.
  #4  Background tiles (crops with NO annotation overlap) are harvested from the
      TACO images themselves and saved with empty labels. These act as negative
      samples that teach the model "empty ground is not litter" — without needing
      any new images. Tiles get a "_bg" suffix.

Output layout:
    data/yolo/
        images/train/   *.jpg          (+ *_bg.jpg background tiles)
        images/val/     *.jpg
        labels/train/   *.txt  (YOLO seg: class x1 y1 x2 y2 ... normalized;
                                empty for *_bg background tiles)
        labels/val/     *.txt
"""

import io
import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE   = 640
VAL_FRACTION = 0.15
RANDOM_SEED  = 42
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR     = REPO_ROOT / "data" / "yolo"
HF_REPO      = "Zesky665/TACO"
ZIP_INNER    = "COCO_format.zip"
ANNOTATIONS  = "data/annotations.json"

# #3 Drop polygons smaller than this fraction of the image area (noise filter).
MIN_POLY_AREA_FRAC = 0.0002
# #4 Harvest a background tile from roughly this fraction of images (≈ negatives
# as a share of the dataset). Tile side = NEG_TILE_SCALE * min(width, height).
NEG_TILE_RATIO = 0.10
NEG_TILE_SCALE = 0.35
NEG_MAX_TRIES  = 25


def find_zip(snapshot_dir: str) -> str:
    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            if f == ZIP_INNER:
                return os.path.join(root, f)
    raise FileNotFoundError(f"{ZIP_INNER} not found under {snapshot_dir}")


def _polygon_area(xs: list, ys: list) -> float:
    """Shoelace area of a polygon given as parallel x/y lists (px²)."""
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5


def polygon_to_yolo(segmentation: list, img_w: int, img_h: int) -> list[str]:
    """Convert COCO flat-polygon list to normalized YOLO seg point strings.

    Polygons smaller than MIN_POLY_AREA_FRAC of the image are dropped (#3).
    """
    lines = []
    img_area = float(img_w * img_h)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        xs = poly[0::2]
        ys = poly[1::2]
        if img_area > 0 and _polygon_area(xs, ys) / img_area < MIN_POLY_AREA_FRAC:
            continue
        points = " ".join(
            f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in zip(xs, ys)
        )
        lines.append(f"0 {points}")
    return lines


def _ann_boxes(anns: list) -> list[tuple[float, float, float, float]]:
    """COCO bboxes [x, y, w, h] → (x1, y1, x2, y2) for all annotations."""
    boxes = []
    for ann in anns:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        boxes.append((x, y, x + w, y + h))
    return boxes


def _sample_bg_tile(
    orig_w: int, orig_h: int, boxes: list
) -> tuple[int, int, int, int] | None:
    """Find a square crop that overlaps no annotation box; None if not found.

    Returns (left, top, right, bottom) in original-image pixel coordinates.
    """
    side = int(min(orig_w, orig_h) * NEG_TILE_SCALE)
    if side < 16 or side >= orig_w or side >= orig_h:
        return None
    for _ in range(NEG_MAX_TRIES):
        left = random.randint(0, orig_w - side)
        top = random.randint(0, orig_h - side)
        right, bottom = left + side, top + side
        overlap = any(
            left < bx2 and right > bx1 and top < by2 and bottom > by1
            for (bx1, by1, bx2, by2) in boxes
        )
        if not overlap:
            return (left, top, right, bottom)
    return None


def main():
    random.seed(RANDOM_SEED)

    for split in ("train", "val"):
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {HF_REPO} snapshot …")
    snapshot_dir = snapshot_download(repo_id=HF_REPO, repo_type="dataset")

    zip_path = find_zip(snapshot_dir)
    print(f"  ZIP found: {zip_path}")

    print("Reading COCO annotations …")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ANNOTATIONS) as f:
            coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    image_ids = list(images_by_id.keys())
    print(f"  Images: {len(image_ids)}   Annotations: {len(coco['annotations'])}")

    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * VAL_FRACTION))
    splits = {"val": image_ids[:n_val], "train": image_ids[n_val:]}

    with zipfile.ZipFile(zip_path) as zf:
        name_map = {e.lower(): e for e in zf.namelist()}

        for split_name, ids in splits.items():
            skipped = 0
            n_bg = 0
            img_out = DATA_DIR / "images" / split_name
            lbl_out = DATA_DIR / "labels" / split_name

            for img_id in tqdm(ids, desc=f"Processing {split_name}"):
                meta = images_by_id[img_id]
                inner_key = f"data/{meta['file_name']}".lower()

                if inner_key not in name_map:
                    skipped += 1
                    continue

                try:
                    with zf.open(name_map[inner_key]) as img_f:
                        img = Image.open(io.BytesIO(img_f.read())).convert("RGB")
                except Exception as e:
                    tqdm.write(f"  Skipping {inner_key}: {e}")
                    skipped += 1
                    continue

                orig_w, orig_h = img.size
                img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
                    img_out / f"{img_id:06d}.jpg", quality=92
                )

                anns = anns_by_image.get(img_id, [])
                label_lines: list[str] = []
                for ann in anns:
                    seg = ann.get("segmentation", [])
                    if not seg or isinstance(seg, dict):
                        continue
                    label_lines.extend(polygon_to_yolo(seg, orig_w, orig_h))

                (lbl_out / f"{img_id:06d}.txt").write_text("\n".join(label_lines))

                # #4 Harvest a background tile (no annotation overlap) as a
                # negative sample with an empty label.
                if random.random() < NEG_TILE_RATIO:
                    tile = _sample_bg_tile(orig_w, orig_h, _ann_boxes(anns))
                    if tile is not None:
                        crop = img.crop(tile).resize(
                            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
                        )
                        crop.save(img_out / f"{img_id:06d}_bg.jpg", quality=92)
                        (lbl_out / f"{img_id:06d}_bg.txt").write_text("")
                        n_bg += 1

            if skipped:
                print(f"  Skipped {skipped} entries in {split_name}")
            print(f"  {split_name}: {n_bg} background tiles added")

    print("\nDone.")
    print(f"  Output: {DATA_DIR}")


if __name__ == "__main__":
    main()