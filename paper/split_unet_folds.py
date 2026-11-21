"""Create the stratified 5-fold split used to train the U-Net ensemble.

Reproduces exactly the split of the Kaggle training notebook:
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42), stratified by image class,
    samples ordered normal -> lichen -> other, each sorted by file name (Linux/Kaggle ordering).

    python paper/split_unet_folds.py --data-root /path/to/dataset --out paper/data/folds.csv

DATA_ROOT layout:  {normal,lichen,other}/{images,masks}/
Output CSV columns: image (relative path), class (0 normal, 1 lichen, 2 other), val_fold (0-4:
the fold in which the image is used for validation; it is in the training set of the other four).

Note: the split is by IMAGE, not by patient (patient identifiers are not available in the dataset).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = [("normal", 0), ("lichen", 1), ("other", 2)]


def collect_samples(root: Path):
    samples = []
    for folder, cls in CLASSES:
        img_dir = root / folder / "images"
        for p in sorted((q for q in img_dir.iterdir() if q.suffix.lower() in EXTS), key=lambda q: q.name):
            samples.append((f"{folder}/images/{p.name}", cls))
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("paper/data/folds.csv"))
    args = ap.parse_args()

    samples = collect_samples(args.data_root)
    labels = np.array([c for _, c in samples])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    val_fold = np.zeros(len(samples), dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(samples, labels)):
        val_fold[val_idx] = k

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "val_fold"])
        for (name, cls), k in zip(samples, val_fold):
            w.writerow([name, cls, int(k)])

    print(f"{len(samples)} images: normal={(labels == 0).sum()} lichen={(labels == 1).sum()} "
          f"other={(labels == 2).sum()}")
    for k in range(N_FOLDS):
        print(f"  fold {k}: train={int((val_fold != k).sum())} val={int((val_fold == k).sum())}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
