"""
prepare_yolo_mc.py — Multiklassen-Müll-DETECTION-Datensatz (Boxen, 6 Material-
Klassen) aus keremberke/garbage-object-detection bauen.

Ansatz A ("kategorisieren statt Maske"): statt einer Klasse "litter" lernt das
Modell 6 Material-Gruppen und gibt Boxen aus. Ein Schuh passt zu keiner Klasse
-> wird strukturell seltener als Müll erkannt.

Quelle (öffentlich, kein Login): keremberke/garbage-object-detection
  - YOLO/Roboflow-Export als ZIP pro Split, Annotationen als _annotations.coco.json
  - Klassen biodegradable/cardboard/glass/metal/paper/plastic == genau unsere 6.

Kanonische Klassenreihenfolge (Plastik/Glas/Metall/Papier/Karton/Bio):
    0 plastik | 1 glas | 2 metall | 3 papier | 4 karton | 5 bio

Ausgabe (getrennt vom Masken-Set data/yolo):
    data/yolo_mc/images/{train,val}/<stem>.jpg
    data/yolo_mc/labels/{train,val}/<stem>.txt   (YOLO-Detection: cls cx cy w h)

Beispiel:
    uv run python auto-research/prepare_yolo_mc.py
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "yolo_mc"
CACHE_DIR = REPO_ROOT / "data" / "hf_garbage"
IMAGE_SIZE = 640
HF_REPO = "keremberke/garbage-object-detection"

# Kanonische 6-Klassen-Reihenfolge (Index = YOLO-Klassen-ID).
CLASS_NAMES = ["plastik", "glas", "metall", "papier", "karton", "bio"]
# Roh-Klassenname (Quelle, lowercase) -> kanonischer Index.
NAME_TO_ID = {
    "plastic": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "cardboard": 4,
    "biodegradable": 5,
}

# (ZIP-Name in HF-Repo, Ziel-Split)
SOURCES = [
    ("data/train.zip", "train"),
    ("data/valid.zip", "val"),
    ("data/test.zip", "val"),
]


def _coco_to_yolo_lines(anns: list, cat_id_to_name: dict, w: float, h: float) -> list[str]:
    """COCO-Annotationen eines Bildes -> YOLO-Detection-Zeilen (cls cx cy w h)."""
    lines: list[str] = []
    for a in anns:
        name = cat_id_to_name.get(a["category_id"], "").lower()
        cls = NAME_TO_ID.get(name)
        if cls is None:
            continue  # unbekannte Klasse -> überspringen
        bx, by, bw, bh = a["bbox"]  # COCO: x,y (top-left), w, h in px
        if bw <= 0 or bh <= 0:
            continue
        cx = (bx + bw / 2) / w
        cy = (by + bh / 2) / h
        nw, nh = bw / w, bh / h
        # auf [0,1] klemmen
        vals = [min(1.0, max(0.0, v)) for v in (cx, cy, nw, nh)]
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in vals))
    return lines


def main() -> None:
    for split in ("train", "val"):
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    grand_total = 0
    cls_counts = {name: 0 for name in CLASS_NAMES}
    for zip_name, dst_split in SOURCES:
        path = hf_hub_download(
            HF_REPO, zip_name, repo_type="dataset", local_dir=str(CACHE_DIR)
        )
        z = zipfile.ZipFile(path)
        coco = json.loads(z.read("_annotations.coco.json"))
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}
        anns_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        out_img = DATA_DIR / "images" / dst_split
        out_lbl = DATA_DIR / "labels" / dst_split
        n_ok = 0
        for img_id, im in tqdm(images.items(), desc=f"{zip_name}->{dst_split}"):
            fn = im["file_name"]
            w, h = float(im["width"]), float(im["height"])
            lines = _coco_to_yolo_lines(
                anns_by_img.get(img_id, []), cat_id_to_name, w, h
            )
            for ln in lines:
                cls_counts[CLASS_NAMES[int(ln.split()[0])]] += 1
            try:
                img = Image.open(io.BytesIO(z.read(fn))).convert("RGB")
            except Exception:
                continue
            stem = Path(fn).stem
            img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
                out_img / f"{stem}.jpg", quality=92
            )
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
            n_ok += 1
        print(f"  {zip_name}: {n_ok} Bilder -> {dst_split}")
        grand_total += n_ok

    print(f"\nFertig: {grand_total} Bilder nach {DATA_DIR}")
    print("Instanzen je Klasse:")
    for name in CLASS_NAMES:
        print(f"  {name}: {cls_counts[name]}")


if __name__ == "__main__":
    main()