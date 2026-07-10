"""
ingest_hf_waste.py — HuggingFace-Müll-Detektionsdatensatz ins YOLO-Seg-Set mischen.

Weg 1 ohne Roboflow-Login: zieht `moondream/waste_detection` (öffentlich, kein
Token nötig) — echte Müll-Bilder in Kontext (Boden/Umgebung), passend zur
Go2-Domäne — und mischt sie als zusätzliche Positiv-Daten ins eigene Training.

Format-Handling:
  - Boxen (normalisiert cx,cy,w,h) -> 4-Punkt-Rechteck-Polygon (seg-kompatibel).
  - ALLE Klassen -> Klasse 0 ("litter"); Single-Class-Modell bleibt.
  - Bilder werden auf IMAGE_SIZE skaliert (wie prepare_yolo.py).

Mengen-Deckel, damit TACO nicht überrollt wird (--max-train / --max-val).

Ausgabe (Suffix "_md"):
    data/yolo/images/{train,val}/<image_id>_md.jpg
    data/yolo/labels/{train,val}/<image_id>_md.txt

Beispiel:
    uv run python auto-research/ingest_hf_waste.py --max-train 1500 --max-val 250
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "yolo"
CACHE_DIR = REPO_ROOT / "data" / "moondream"
IMAGE_SIZE = 640
HF_REPO = "moondream/waste_detection"
RANDOM_SEED = 42

# (parquet-Datei, Ziel-Split, max-Argname)
SOURCES = [
    ("data/train-00000-of-00001.parquet", "train", "max_train"),
    ("data/test-00000-of-00001.parquet", "val", "max_val"),
]


def _boxes_to_label(boxes) -> str:
    """Liste normalisierter (cx,cy,w,h)-Boxen -> YOLO-Seg-Zeilen (Klasse 0)."""
    lines: list[str] = []
    if boxes is None:
        return ""
    for b in boxes:
        if len(b) != 4:
            continue
        cx, cy, w, h = (float(v) for v in b)
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        # auf [0,1] klemmen (einige Boxen ragen minimal über den Rand)
        pts = [x0, y0, x1, y0, x1, y1, x0, y1]
        pts = [min(1.0, max(0.0, p)) for p in pts]
        lines.append("0 " + " ".join(f"{p:.6f}" for p in pts))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--max-val", type=int, default=250)
    args = ap.parse_args()
    caps = {"max_train": args.max_train, "max_val": args.max_val}

    total = 0
    for parquet_name, dst_split, cap_key in SOURCES:
        cap = caps[cap_key]
        try:
            path = hf_hub_download(
                HF_REPO, parquet_name, repo_type="dataset", local_dir=str(CACHE_DIR)
            )
        except Exception as e:
            print(f"  WARN: {parquet_name} nicht ladbar ({e}); übersprungen")
            continue
        df = pd.read_parquet(path)
        if cap and len(df) > cap:
            df = df.sample(n=cap, random_state=RANDOM_SEED).reset_index(drop=True)

        out_img = DATA_DIR / "images" / dst_split
        out_lbl = DATA_DIR / "labels" / dst_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        n_ok = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dst_split}"):
            img_field = row["image"]
            raw = img_field["bytes"] if isinstance(img_field, dict) else img_field
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            stem = f"{row.get('image_id', n_ok)}_md_{dst_split}"
            img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
                out_img / f"{stem}.jpg", quality=92
            )
            (out_lbl / f"{stem}.txt").write_text(_boxes_to_label(row["boxes"]))
            n_ok += 1
        print(f"  {dst_split}: {n_ok} Bilder geschrieben")
        total += n_ok

    print(f"\nFertig: {total} HF-Müll-Bilder gemerged.")
    print("-> Jetzt neu trainieren: uv run python auto-research/train_yolo.py")


if __name__ == "__main__":
    main()