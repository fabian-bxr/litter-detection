"""
ingest_zerowaste_mc.py — ZeroWaste-f-Subset ins MC-DETECTION-Set (data/yolo_mc).

Material-gelabelt (Förderband-Domäne, CVPR 2022). Klassen-Mapping:
    rigid_plastic, soft_plastic -> plastik (0)
    cardboard                   -> karton  (4)
    metal                       -> metall  (2)
(kein glas/papier/bio in ZeroWaste.)

Anti-Leakage: ZeroWaste sind korrelierte VIDEO-Frames. Daher:
  - der vorgegebene Split wird respektiert (train->train, val+test->val),
  - der Train-Anteil wird STRIDED (gleichmäßig) auf --max-train gedeckelt, statt
    zufällig — so landen weniger Quasi-Duplikate im Set.

Suffix "_zw". Quelle: data/downloads/zerowaste-f-final.zip

Beispiel:
    uv run python auto-research/ingest_zerowaste_mc.py --max-train 1500 --max-val 300
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DL = REPO_ROOT / "data" / "downloads" / "zerowaste-f-final.zip"
OUT = REPO_ROOT / "data" / "yolo_mc"
IMAGE_SIZE = 640
P, M, K = 0, 2, 4  # plastik, metall, karton (kanonische MC-Indizes)
NAME_TO_ID = {
    "rigid_plastic": P, "soft_plastic": P, "cardboard": K, "metal": M,
}
CLASS_NAMES = ["plastik", "glas", "metall", "papier", "karton", "bio"]


def _yolo_line(cls, bx, by, bw, bh, w, h) -> str | None:
    if bw <= 0 or bh <= 0:
        return None
    cx, cy, nw, nh = (bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h
    vals = [min(1.0, max(0.0, v)) for v in (cx, cy, nw, nh)]
    return f"{cls} " + " ".join(f"{v:.6f}" for v in vals)


def _strided(items: list, cap: int) -> list:
    if cap <= 0 or len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def process_split(zf, src_split, dst_split, cap, counts) -> int:
    base = f"splits_final_deblurred/{src_split}"
    try:
        coco = json.loads(zf.read(f"{base}/labels.json"))
    except KeyError:
        return 0
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    ids = sorted(images.keys())          # zeitliche Reihenfolge -> strided
    ids = _strided(ids, cap)

    out_img = OUT / "images" / dst_split
    out_lbl = OUT / "labels" / dst_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for img_id in tqdm(ids, desc=f"ZeroWaste {src_split}->{dst_split}"):
        im = images[img_id]
        w, h = float(im["width"]), float(im["height"])
        lines = []
        for a in anns_by_img.get(img_id, []):
            cls = NAME_TO_ID.get(cat_name.get(a["category_id"], ""))
            if cls is None:
                continue
            ln = _yolo_line(cls, *a["bbox"], w, h)
            if ln:
                lines.append(ln)
                counts[CLASS_NAMES[cls]] += 1
        if not lines:
            continue
        member = f"{base}/data/{im['file_name']}"
        try:
            img = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
        except (KeyError, Exception):
            continue
        stem = f"zw_{src_split}_{Path(im['file_name']).stem}"
        img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
            out_img / f"{stem}.jpg", quality=92
        )
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
        n_ok += 1
    print(f"  {src_split}->{dst_split}: {n_ok} Bilder")
    return n_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--max-val", type=int, default=300)
    args = ap.parse_args()

    counts = {n: 0 for n in CLASS_NAMES}
    zf = zipfile.ZipFile(DL)
    total = process_split(zf, "train", "train", args.max_train, counts)
    total += process_split(zf, "val", "val", args.max_val // 2, counts)
    total += process_split(zf, "test", "val", args.max_val // 2, counts)

    print(f"\nFertig: {total} ZeroWaste-Bilder ins MC-Set ({OUT}).")
    print("Neue Instanzen je Klasse:")
    for n in CLASS_NAMES:
        if counts[n]:
            print(f"  {n}: {counts[n]}")


if __name__ == "__main__":
    main()