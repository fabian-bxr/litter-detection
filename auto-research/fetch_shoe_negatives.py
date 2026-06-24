"""
fetch_shoe_negatives.py — Schuh-Hard-Negatives aus Open Images V7 holen.

Lädt echte Schuh-Bilder (Open-Images-Klasse "Footwear") OHNE fiftyone direkt über
die offiziellen CSV-Listen + den öffentlichen S3-Bucket und mischt sie als
*Hard Negatives* (leeres Label) ins YOLO-Trainingsset. Das bringt dem
Ein-Klassen-Litter-Modell bei: "Schuh != Müll" und reduziert Schuh-Fehlalarme.

Sauberkeits-Filter: Bilder, die laut Annotation TACO-ähnliche Confusables
enthalten (Flasche, Dose, Becher, Karton ...), werden VERWORFEN — sonst würde ein
leeres Label dem Modell beibringen, echten Müll zu unterdrücken.

Quellen (verifiziert):
  - Klassen:      https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
  - Annotationen: https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv
  - Bilder:       https://open-images-dataset.s3.amazonaws.com/{split}/{ImageID}.jpg

Ausgabe (kompatibel zu prepare_yolo.py, IMAGE_SIZE=640, Suffix "_shoeneg"):
    data/yolo/images/train/<id>_shoeneg.jpg   (+ kleiner val-Anteil)
    data/yolo/labels/train/<id>_shoeneg.txt   (leer)

Beispiel (PowerShell):
    uv run python auto-research/fetch_shoe_negatives.py --max 400
    uv run python auto-research/fetch_shoe_negatives.py --max 50 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "yolo"
CACHE_DIR = REPO_ROOT / "data" / "openimages_cache"

IMAGE_SIZE = 640          # gleiche Größe wie prepare_yolo.py
VAL_FRACTION = 0.15       # Anteil der Negatives, der in den val-Split geht
RANDOM_SEED = 42

CLASS_DESC_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
ANN_URL_TMPL = "https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv"
IMG_URL_TMPL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

# Alle schuhartigen Open-Images-Klassen (exakte Display-Namen, lowercase).
# Sandal deckt Schlappen/Flip-Flops mit ab, Boot/High heels weitere Varianten.
TARGET_NAMES = ("footwear", "sandal", "boot", "high heels")
# Display-Name-Substrings für TACO-ähnliche Confusables -> Bild verwerfen, wenn
# eine solche Klasse mit-annotiert ist (sonst falsches "leeres" Label).
EXCLUDE_KEYWORDS = (
    "bottle", "tin can", "beer", "wine", "juice", "soft drink",
    "coffee cup", "plastic bag", "drink can", "carton", "jar", "mug",
)


def _download_text(url: str, cache: Path) -> str:
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read().decode("utf-8")
    cache.write_text(data, encoding="utf-8")
    return data


def load_class_mids() -> tuple[set[str], set[str]]:
    """Schuh-Ziel-MIDs + Exclude-MIDs (Confusables) ermitteln."""
    text = _download_text(CLASS_DESC_URL, CACHE_DIR / "class-descriptions-boxable.csv")
    targets: set[str] = set()
    exclude: set[str] = set()
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 2:
            continue
        mid, name = row[0].strip(), row[1].strip().lower()
        if name in TARGET_NAMES:
            targets.add(mid)
        if any(k in name for k in EXCLUDE_KEYWORDS):
            exclude.add(mid)
    if not targets:
        raise SystemExit("Keine Schuh-Zielklassen in class-descriptions gefunden")
    return targets, exclude


def select_image_ids(
    split: str, targets: set[str], exclude: set[str], limit: int, min_shoe_frac: float
) -> list[str]:
    """ImageIDs mit prominentem Schuh, aber OHNE Confusable-Klassen.

    Behalten wird ein Bild nur, wenn (a) keine Confusable-Klasse annotiert ist und
    (b) mindestens eine Schuh-Box (irgendeine Zielklasse) >= `min_shoe_frac` der
    Bildfläche bedeckt. Letzteres siebt diffuse Menschenmengen/Ganzkörper-Szenen
    aus und bevorzugt Nahaufnahmen von Schuhen -- die Boden-FP-Confuser.
    """
    cache = CACHE_DIR / f"{split}-annotations-bbox.csv"
    text = _download_text(ANN_URL_TMPL.format(split=split), cache)
    rdr = csv.reader(io.StringIO(text))
    header = next(rdr)
    ii, li = header.index("ImageID"), header.index("LabelName")
    xi0, xi1 = header.index("XMin"), header.index("XMax")
    yi0, yi1 = header.index("YMin"), header.index("YMax")
    labels_by_img: dict[str, set[str]] = defaultdict(set)
    max_shoe_area: dict[str, float] = defaultdict(float)
    for row in rdr:
        img = row[ii]
        labels_by_img[img].add(row[li])
        if row[li] in targets:
            area = (float(row[xi1]) - float(row[xi0])) * (
                float(row[yi1]) - float(row[yi0])
            )
            if area > max_shoe_area[img]:
                max_shoe_area[img] = area
    chosen = [
        img for img, labels in labels_by_img.items()
        if (labels & targets)
        and not (labels & exclude)
        and max_shoe_area[img] >= min_shoe_frac
    ]
    random.shuffle(chosen)
    return chosen[:limit]


def _fetch_one(split: str, image_id: str) -> bytes | None:
    url = IMG_URL_TMPL.format(split=split, image_id=image_id)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except Exception:
        return None


def save_negative(raw: bytes, image_id: str, to_val: bool) -> bool:
    split = "val" if to_val else "train"
    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return False
    img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
        img_dir / f"{image_id}_shoeneg.jpg", quality=92
    )
    (lbl_dir / f"{image_id}_shoeneg.txt").write_text("")  # leeres Label
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max", type=int, default=400,
                    help="max. Anzahl Negatives (über alle Splits zusammen)")
    ap.add_argument("--splits", nargs="+", default=["validation", "test"],
                    choices=["validation", "test", "train"],
                    help="Open-Images-Splits als Quelle (mehr = mehr Volumen)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-shoe-frac", type=float, default=0.05,
                    help="min. Flächenanteil der größten Schuh-Box (0=aus); "
                         "höher = nur prominente Nahaufnahmen")
    ap.add_argument("--dry-run", action="store_true",
                    help="nur auswählen/zählen, nichts herunterladen/schreiben")
    args = ap.parse_args()

    random.seed(RANDOM_SEED)

    print("1) Klassen-Mapping laden ...")
    targets, exclude = load_class_mids()
    print(f"   Schuh-Zielklassen={len(targets)}   Exclude-Klassen={len(exclude)}")

    print(f"2) Bilder auswählen (Schuhe ohne Confusables, Splits={args.splits}) ...")
    # (split, image_id)-Paare über alle Quellen sammeln, bis --max erreicht ist.
    pairs: list[tuple[str, str]] = []
    for split in args.splits:
        remaining = args.max - len(pairs)
        if remaining <= 0:
            break
        ids = select_image_ids(
            split, targets, exclude, remaining, args.min_shoe_frac
        )
        print(f"   {split}: {len(ids)} Kandidaten")
        pairs.extend((split, i) for i in ids)
    print(f"   gesamt: {len(pairs)} Kandidaten")
    if args.dry_run:
        print("   (dry-run) — Beispiele:", pairs[:5])
        return

    print("3) Bilder herunterladen & als Hard Negatives schreiben ...")
    n_ok = n_val = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch_one, sp, i): i for (sp, i) in pairs}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            image_id = futs[fut]
            raw = fut.result()
            if raw is None:
                continue
            to_val = random.random() < VAL_FRACTION
            if save_negative(raw, image_id, to_val):
                n_ok += 1
                n_val += int(to_val)

    print(f"\nFertig: {n_ok} Schuh-Negatives geschrieben "
          f"({n_ok - n_val} train / {n_val} val).")
    print("-> Jetzt neu trainieren: uv run python auto-research/train_yolo.py")


if __name__ == "__main__":
    main()