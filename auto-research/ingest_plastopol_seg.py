"""
ingest_plastopol_seg.py — PlastOPol (Ein-Klasse "Trash") ins SEG-Set (data/yolo)
mischen. Reale Outdoor-/Strand-/Fluss-Litter-Szenen, CC-BY 4.0.

Box-Annotationen (COCO) -> 4-Punkt-Rechteck-Polygon, Klasse 0 ("litter"),
seg-kompatibel. Eigener 85/15-Split. Suffix "_plast".

Quelle: data/downloads/PlastOpol_oneclass.zip  (enthält images.zip + annotations)

Beispiel:
    uv run python auto-research/ingest_plastopol_seg.py
"""

from __future__ import annotations

import io
import json
import random
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DL = REPO_ROOT / "data" / "downloads" / "PlastOpol_oneclass.zip"
TMP = REPO_ROOT / "data" / "downloads" / "plastopol_tmp"
OUT = REPO_ROOT / "data" / "yolo"
IMAGE_SIZE = 640
VAL_FRACTION = 0.15
RANDOM_SEED = 42


def _rect_poly(bx, by, bw, bh, w, h) -> str | None:
    if bw <= 0 or bh <= 0:
        return None
    x0, y0, x1, y1 = bx / w, by / h, (bx + bw) / w, (by + bh) / h
    pts = [x0, y0, x1, y0, x1, y1, x0, y1]
    pts = [min(1.0, max(0.0, p)) for p in pts]
    return "0 " + " ".join(f"{p:.6f}" for p in pts)


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    # 1) innere ZIPs auf Platte legen (images.zip ist groß -> nicht in RAM)
    with zipfile.ZipFile(DL) as outer:
        for inner in ("images.zip", "plastopol-annotations_folds.zip"):
            if not (TMP / inner).exists():
                (TMP / inner).write_bytes(outer.read(inner))

    ann = json.loads(
        zipfile.ZipFile(TMP / "plastopol-annotations_folds.zip")
        .read("annotations/full_plastopol.json")
    )
    images = {im["id"]: im for im in ann["images"]}
    anns_by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    ids = list(images.keys())
    random.Random(RANDOM_SEED).shuffle(ids)
    n_val = int(len(ids) * VAL_FRACTION)
    val_ids = set(ids[:n_val])

    imgzip = zipfile.ZipFile(TMP / "images.zip")
    members = {Path(n).name: n for n in imgzip.namelist()}

    n_ok = 0
    for img_id in tqdm(ids, desc="PlastOPol"):
        im = images[img_id]
        w, h = float(im["width"]), float(im["height"])
        lines = []
        for a in anns_by_img.get(img_id, []):
            bx, by, bw, bh = a["bbox"]
            ln = _rect_poly(bx, by, bw, bh, w, h)
            if ln:
                lines.append(ln)
        if not lines:
            continue
        fname = Path(im["file_name"]).name
        member = members.get(fname)
        if member is None:
            continue
        try:
            img = Image.open(io.BytesIO(imgzip.read(member))).convert("RGB")
        except Exception:
            continue
        split = "val" if img_id in val_ids else "train"
        out_img = OUT / "images" / split
        out_lbl = OUT / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        stem = f"plast_{Path(fname).stem}"
        img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
            out_img / f"{stem}.jpg", quality=92
        )
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
        n_ok += 1

    print(f"\nFertig: {n_ok} PlastOPol-Bilder ins Seg-Set ({OUT}).")
    print("Tipp: data/downloads/plastopol_tmp kann danach gelöscht werden.")


if __name__ == "__main__":
    main()