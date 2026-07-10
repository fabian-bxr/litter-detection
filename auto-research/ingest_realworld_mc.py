"""
ingest_realworld_mc.py — Echtwelt-Müll (TACO + moondream) ins Multiklassen-
DETECTION-Set (data/yolo_mc) einmischen, auf 6 Material-Klassen gemappt.

Ergänzt den studio-lastigen keremberke-Anker um Bilder aus realer Umgebung
(Boden/Outdoor) → bessere Domänen-Passung für den Go2 und glättet die Schieflage.

Kanonische Klassen (Index): 0 plastik | 1 glas | 2 metall | 3 papier | 4 karton | 5 bio

TACO-Mapping auf Kategorie-Ebene (Supergruppen mischen Materialien). "Shoe" und
mehrdeutige Kategorien werden VERWORFEN — Shoe-Drop hilft zusätzlich gegen
Schuh-FPs. moondream nur mit eindeutigen Labels.

Suffixe: _taco / _md   (Ausgabe nach data/yolo_mc/{images,labels}/{train,val})

Beispiel:
    uv run python auto-research/ingest_realworld_mc.py
"""

from __future__ import annotations

import io
import json
import os
import random
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "yolo_mc"
IMAGE_SIZE = 640
VAL_FRACTION = 0.15
RANDOM_SEED = 42

CLASS_NAMES = ["plastik", "glas", "metall", "papier", "karton", "bio"]
P, G, M, PA, K, B = 0, 1, 2, 3, 4, 5

# --- TACO: exakter Kategoriename -> Material-Index (None = verwerfen) ---
TACO_MAP: dict[str, int | None] = {
    # plastik
    "Clear plastic bottle": P, "Other plastic bottle": P, "Plastic bottle cap": P,
    "Disposable plastic cup": P, "Foam cup": P, "Other plastic cup": P,
    "Plastic lid": P, "Other plastic": P, "Crisp packet": P, "Garbage bag": P,
    "Other plastic wrapper": P, "Plastic film": P, "Polypropylene bag": P,
    "Single-use carrier bag": P, "Six pack rings": P, "Disposable food container": P,
    "Foam food container": P, "Other plastic container": P, "Spread tub": P,
    "Tupperware": P, "Plastic glooves": P, "Plastic utensils": P,
    "Plastic straw": P, "Styrofoam piece": P, "Squeezable tube": P,
    # glas
    "Glass bottle": G, "Broken glass": G, "Glass cup": G, "Glass jar": G,
    # metall
    "Aluminium foil": M, "Metal bottle cap": M, "Aerosol": M, "Drink can": M,
    "Food Can": M, "Metal lid": M, "Pop tab": M, "Scrap metal": M,
    # papier
    "Paper cup": PA, "Magazine paper": PA, "Normal paper": PA, "Tissues": PA,
    "Wrapping paper": PA, "Paper bag": PA, "Plastified paper bag": PA,
    "Paper straw": PA,
    # karton
    "Corrugated carton": K, "Drink carton": K, "Egg carton": K, "Meal carton": K,
    "Other carton": K, "Pizza box": K, "Toilet tube": K,
    # bio
    "Food waste": B,
    # verworfen
    "Battery": None, "Aluminium blister pack": None, "Carded blister pack": None,
    "Cigarette": None, "Rope & strings": None, "Shoe": None,
    "Unlabeled litter": None,
}

# --- moondream: nur eindeutige Labels ---
MOONDREAM_MAP: dict[str, int] = {"can": M, "plasticbag": P, "paper": PA}


def _out_dirs(split: str) -> tuple[Path, Path]:
    img = DATA_DIR / "images" / split
    lbl = DATA_DIR / "labels" / split
    img.mkdir(parents=True, exist_ok=True)
    lbl.mkdir(parents=True, exist_ok=True)
    return img, lbl


def _yolo_line(cls: int, bx: float, by: float, bw: float, bh: float,
               w: float, h: float) -> str | None:
    if bw <= 0 or bh <= 0:
        return None
    cx, cy = (bx + bw / 2) / w, (by + bh / 2) / h
    nw, nh = bw / w, bh / h
    vals = [min(1.0, max(0.0, v)) for v in (cx, cy, nw, nh)]
    return f"{cls} " + " ".join(f"{v:.6f}" for v in vals)


def ingest_taco(counts: dict[str, int]) -> int:
    snap = snapshot_download("Zesky665/TACO", repo_type="dataset",
                             allow_patterns=["*.zip", "*.json"])
    zp = None
    for root, _, files in os.walk(snap):
        for f in files:
            if f == "COCO_format.zip":
                zp = os.path.join(root, f)
    if not zp:
        print("  WARN: TACO COCO_format.zip nicht gefunden; übersprungen")
        return 0

    with zipfile.ZipFile(zp) as z:
        coco = json.loads(z.read("data/annotations.json"))
        name_map = {e.lower(): e for e in z.namelist()}
        cat_name = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}
        anns_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        ids = list(images.keys())
        random.Random(RANDOM_SEED).shuffle(ids)
        n_val = int(len(ids) * VAL_FRACTION)
        val_ids = set(ids[:n_val])

        n_ok = 0
        for img_id in tqdm(ids, desc="TACO"):
            im = images[img_id]
            w, h = float(im["width"]), float(im["height"])
            lines = []
            for a in anns_by_img.get(img_id, []):
                cls = TACO_MAP.get(cat_name.get(a["category_id"], ""))
                if cls is None:
                    continue
                bx, by, bw, bh = a["bbox"]
                ln = _yolo_line(cls, bx, by, bw, bh, w, h)
                if ln:
                    lines.append(ln)
                    counts[CLASS_NAMES[cls]] += 1
            if not lines:
                continue  # Bilder ohne gemappte Klasse überspringen
            inner = name_map.get(f"data/{im['file_name']}".lower())
            if not inner:
                continue
            try:
                img = Image.open(io.BytesIO(z.read(inner))).convert("RGB")
            except Exception:
                continue
            split = "val" if img_id in val_ids else "train"
            out_img, out_lbl = _out_dirs(split)
            stem = f"taco_{img_id:06d}"
            img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
                out_img / f"{stem}.jpg", quality=92
            )
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
            n_ok += 1
    print(f"  TACO: {n_ok} Bilder gemerged")
    return n_ok


def ingest_moondream(counts: dict[str, int]) -> int:
    sources = [("data/train-00000-of-00001.parquet", "train"),
               ("data/test-00000-of-00001.parquet", "val")]
    n_ok = 0
    for parquet_name, split in sources:
        try:
            path = hf_hub_download("moondream/waste_detection", parquet_name,
                                   repo_type="dataset",
                                   local_dir=str(REPO_ROOT / "data" / "moondream"))
        except Exception as e:
            print(f"  WARN moondream {parquet_name}: {e}")
            continue
        df = pd.read_parquet(path)
        out_img, out_lbl = _out_dirs(split)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"moondream/{split}"):
            lines = []
            for box, lab in zip(row["boxes"], row["labels"]):
                cls = MOONDREAM_MAP.get(str(lab).lower())
                if cls is None or len(box) != 4:
                    continue
                cx, cy, bw, bh = (float(v) for v in box)  # bereits normalisiert
                x0, y0 = cx - bw / 2, cy - bh / 2
                ln = _yolo_line(cls, x0, y0, bw, bh, 1.0, 1.0)
                if ln:
                    lines.append(ln)
                    counts[CLASS_NAMES[cls]] += 1
            if not lines:
                continue  # nur Bilder mit eindeutig gemappter Klasse
            img_field = row["image"]
            raw = img_field["bytes"] if isinstance(img_field, dict) else img_field
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            stem = f"md_{row.get('image_id', n_ok)}_{split}"
            img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
                out_img / f"{stem}.jpg", quality=92
            )
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
            n_ok += 1
    print(f"  moondream: {n_ok} Bilder gemerged")
    return n_ok


def main() -> None:
    counts = {n: 0 for n in CLASS_NAMES}
    total = ingest_taco(counts) + ingest_moondream(counts)
    print(f"\nFertig: {total} Echtwelt-Bilder ins MC-Set gemischt.")
    print("Neue Instanzen je Klasse (nur dieser Lauf):")
    for n in CLASS_NAMES:
        print(f"  {n}: {counts[n]}")


if __name__ == "__main__":
    main()