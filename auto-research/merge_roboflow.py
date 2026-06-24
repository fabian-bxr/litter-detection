"""
merge_roboflow.py — Roboflow-Müll-Datensatz in das YOLO-Seg-Trainingsset mischen.

Weg 1 ("ergänzend"): zusätzliche, diverse Müll-Bilder ins eigene Training holen,
um die "litter"-Klasse zu schärfen und unspezifische False Positives (z. B.
Schuhe) weiter zurückzudrängen. Das eigene lokale Seg-Modell bleibt erhalten.

Kann eine bereits heruntergeladene YOLO-Dataset-Mappe verarbeiten (--src) ODER
direkt per Roboflow-API laden (--rf "workspace/project:version", braucht
ROBOFLOW_API_KEY in der Umgebung).

Label-Handling (macht jeden Roboflow-Müll-Datensatz seg-kompatibel):
  - Polygon-Labels (Instanz-Segmentierung) -> unverändert übernommen.
  - Box-Labels (cls cx cy w h)             -> in 4-Punkt-Rechteck-Polygon gewandelt.
  - ALLE Klassen werden auf Klasse 0 ("litter") remappt (Single-Class-Modell).
  - Leere Label-Dateien bleiben leer (Negatives).

Ausgabe (kompatibel zu prepare_yolo.py, Suffix "_rf"):
    data/yolo/images/{train,val}/<stem>_rf.jpg
    data/yolo/labels/{train,val}/<stem>_rf.txt

Beispiele (PowerShell):
    # A) bereits heruntergeladenes Dataset (vom Roboflow-Web-Export entpackt)
    uv run python auto-research/merge_roboflow.py --src C:\pfad\zum\dataset

    # B) direkt per API (roboflow-Paket nötig)
    $env:ROBOFLOW_API_KEY = "DEIN_KEY"
    uv run --with roboflow python auto-research/merge_roboflow.py `
        --rf "projectverba/yolo-waste-detection:1"
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "yolo"
IMAGE_SIZE = 640
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow-Splitnamen -> unsere Splits.
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "val"}


def _convert_label(text: str) -> str:
    """Label-Zeilen auf Klasse 0 remappen; Boxen -> Rechteck-Polygone."""
    out_lines: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        coords = [float(x) for x in parts[1:]]
        if len(coords) == 4:
            # Box (cx, cy, w, h) -> 4-Punkt-Polygon (im Uhrzeigersinn).
            cx, cy, w, h = coords
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            coords = [x0, y0, x1, y0, x1, y1, x0, y1]
        elif len(coords) < 6 or len(coords) % 2 != 0:
            continue  # degeneriertes Label -> überspringen
        pts = " ".join(f"{c:.6f}" for c in coords)
        out_lines.append(f"0 {pts}")
    return "\n".join(out_lines)


def _iter_split_dirs(root: Path):
    """(src_split, images_dir, labels_dir) für jeden vorhandenen Split liefern."""
    for src_split in SPLIT_MAP:
        img_dir = root / src_split / "images"
        lbl_dir = root / src_split / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            yield src_split, img_dir, lbl_dir


def _download_roboflow(rf_spec: str) -> Path:
    """rf_spec = 'workspace/project:version' -> lokaler Dataset-Pfad."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY ist nicht gesetzt (Umgebungsvariable).")
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "roboflow-Paket fehlt. Starte mit:  uv run --with roboflow python ..."
        )
    ws_proj, _, ver = rf_spec.partition(":")
    ws, _, proj = ws_proj.partition("/")
    if not (ws and proj and ver):
        raise SystemExit('--rf erwartet Format "workspace/project:version"')
    dest = REPO_ROOT / "data" / "roboflow_cache" / f"{proj}-{ver}"
    rf = Roboflow(api_key=key)
    print(f"  Roboflow-Download: {ws}/{proj} v{ver} -> {dest}")
    rf.workspace(ws).project(proj).version(int(ver)).download(
        "yolov8", location=str(dest)
    )
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", type=Path,
                   help="Pfad zu einem bereits heruntergeladenen YOLO-Dataset")
    g.add_argument("--rf", type=str,
                   help='Roboflow "workspace/project:version" (braucht API-Key)')
    ap.add_argument("--max", type=int, default=0,
                    help="max. Bilder gesamt (0 = alle)")
    args = ap.parse_args()

    root = _download_roboflow(args.rf) if args.rf else args.src
    if not root.is_dir():
        raise SystemExit(f"Kein Dataset-Ordner: {root}")

    n_ok = n_box = n_poly = n_empty = 0
    by_split: dict[str, int] = {"train": 0, "val": 0}
    for src_split, img_dir, lbl_dir in _iter_split_dirs(root):
        dst_split = SPLIT_MAP[src_split]
        out_img = DATA_DIR / "images" / dst_split
        out_lbl = DATA_DIR / "labels" / dst_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        for img_path in tqdm(images, desc=f"{src_split}->{dst_split}"):
            if args.max and n_ok >= args.max:
                break
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            raw = lbl_path.read_text() if lbl_path.exists() else ""
            converted = _convert_label(raw)
            # Statistik: Box vs. Polygon im Quell-Label erkennen.
            first = next((ln.split() for ln in raw.splitlines() if ln.split()), None)
            if first is None:
                n_empty += 1
            elif len(first) - 1 == 4:
                n_box += 1
            else:
                n_poly += 1

            stem = f"{img_path.stem}_rf"
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(
                out_img / f"{stem}.jpg", quality=92
            )
            (out_lbl / f"{stem}.txt").write_text(converted)
            n_ok += 1
            by_split[dst_split] += 1

    print(f"\nFertig: {n_ok} Bilder gemerged "
          f"({by_split['train']} train / {by_split['val']} val).")
    print(f"  Quell-Labels: {n_poly} Polygon, {n_box} Box->Rechteck, {n_empty} leer.")
    if n_ok == 0:
        print("  WARN: nichts gefunden - hat das Dataset train/images + train/labels?")
    else:
        print("-> Jetzt neu trainieren: uv run python auto-research/train_yolo.py")


if __name__ == "__main__":
    main()