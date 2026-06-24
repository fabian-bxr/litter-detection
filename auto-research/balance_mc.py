"""
balance_mc.py — Klassen-Schieflage im MC-Set glätten (bio deckeln).

Der keremberke-Anker bringt extrem viele bio-Instanzen mit (bio >> übrige
Klassen). Das verzerrt das Training. Dieses Script entfernt zufällig
*bio-dominante* keremberke-Trainingsbilder (bio = Mehrheit der Boxen im Bild),
bis die bio-Instanzzahl unter das Ziel fällt. Echtwelt-Bilder (taco_/md_) und
alle Nicht-bio-dominanten Bilder bleiben unberührt.

Beispiel:
    uv run python auto-research/balance_mc.py --target-bio 7000
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LBL_DIR = REPO_ROOT / "data" / "yolo_mc" / "labels" / "train"
IMG_DIR = REPO_ROOT / "data" / "yolo_mc" / "images" / "train"
RANDOM_SEED = 42
CLASS_NAMES = ["plastik", "glas", "metall", "papier", "karton", "bio"]
# Echtwelt-Quellen NIE ausdünnen (Diversität erhalten); nur Studio/Förderband.
KEEP_PREFIXES = ("taco_", "md_")


def _counts(label_path: Path) -> collections.Counter:
    c = collections.Counter()
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if p:
            c[int(p[0])] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class-name", default="bio", choices=CLASS_NAMES,
                    help="zu deckelnde Klasse")
    ap.add_argument("--target", type=int, default=7000,
                    help="max. Instanzen dieser Klasse")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cls = CLASS_NAMES.index(args.class_name)

    per_img: dict[Path, collections.Counter] = {}
    total = 0
    for lp in LBL_DIR.glob("*.txt"):
        c = _counts(lp)
        per_img[lp] = c
        total += c[cls]
    print(f"{args.class_name}-Instanzen aktuell: {total}  (Ziel <= {args.target})")

    # Kandidaten: Nicht-Echtwelt-Bilder, in denen die Zielklasse die Mehrheit ist.
    candidates = []
    for lp, c in per_img.items():
        if lp.stem.startswith(KEEP_PREFIXES):
            continue
        n = sum(c.values())
        if n > 0 and c[cls] * 2 > n:  # Zielklasse > 50 %
            candidates.append(lp)
    random.Random(RANDOM_SEED).shuffle(candidates)

    removed = removed_inst = 0
    for lp in candidates:
        if total <= args.target:
            break
        total -= per_img[lp][cls]
        removed_inst += per_img[lp][cls]
        removed += 1
        if not args.dry_run:
            (IMG_DIR / f"{lp.stem}.jpg").unlink(missing_ok=True)
            lp.unlink(missing_ok=True)

    print(f"Entfernt: {removed} {args.class_name}-dominante Bilder "
          f"({removed_inst} {args.class_name}-Boxen).")
    print(f"{args.class_name}-Instanzen jetzt: {total}"
          + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()