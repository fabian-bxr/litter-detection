"""
prepare_yolo.py — Download TACO and convert to YOLO segmentation format.

Output layout:
    data/yolo/
        images/train/   *.jpg
        images/val/     *.jpg
        labels/train/   *.txt  (YOLO seg: class x1 y1 x2 y2 ... normalized)
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


def find_zip(snapshot_dir: str) -> str:
    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            if f == ZIP_INNER:
                return os.path.join(root, f)
    raise FileNotFoundError(f"{ZIP_INNER} not found under {snapshot_dir}")


def polygon_to_yolo(segmentation: list, img_w: int, img_h: int) -> list[str]:
    """Convert COCO flat-polygon list to normalized YOLO seg point strings."""
    lines = []
    for poly in segmentation:
        if len(poly) < 6:
            continue
        xs = poly[0::2]
        ys = poly[1::2]
        points = " ".join(
            f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in zip(xs, ys)
        )
        lines.append(f"0 {points}")
    return lines


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

                label_lines: list[str] = []
                for ann in anns_by_image.get(img_id, []):
                    seg = ann.get("segmentation", [])
                    if not seg or isinstance(seg, dict):
                        continue
                    label_lines.extend(polygon_to_yolo(seg, orig_w, orig_h))

                (lbl_out / f"{img_id:06d}.txt").write_text("\n".join(label_lines))

            if skipped:
                print(f"  Skipped {skipped} entries in {split_name}")

    print("\nDone.")
    print(f"  Output: {DATA_DIR}")


if __name__ == "__main__":
    main()