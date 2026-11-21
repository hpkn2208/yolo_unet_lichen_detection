"""
Generate YOLOv8 segmentation labels from pixel masks.

Sources (merged, all remapped to class 0 = lesion):
  1. Existing YOLO seg zips  (Pemphigoid, Pemphigus, lichen_retake, LE ...)
  2. Pixel masks in lichen_masks / other_masks  (auto-generated via contour)
  3. normal_images  → hard-negatives (empty label files)

Output:
    yolo_seg_dataset/
        images/train/   images/val/
        labels/train/   labels/val/
        data.yaml

All lesion classes (Lichen=0, Other=1 from zips) are remapped to 0 = lesion.
Normal images get empty label files (hard-negatives).

Data split: lesion images and background images are shuffled separately (seed 42) and 20% of each
goes to validation (stratified 80/20). The 20%-background filter of the TRAIN split is applied
afterwards by train_yolo.py.

Usage (data are not part of this repository):
    python paper/prepare_yolo_data.py --data-root /path/to/unet_data --out yolo_seg_dataset \
        [--yolo-zips-dir /path/with/Pemphigoid_yolo.zip ...]
--data-root must contain lichen_images/, lichen_masks/, other_images/, other_masks/, normal_images/.
"""

import cv2
import numpy as np
import os
import glob
import random
import shutil
import zipfile
import yaml
from pathlib import Path

# ── Config (paths are set from the command line in main()) ───────────────────
VAL_SPLIT  = 0.2
SEED       = 42

# Zip files that already have YOLO seg labels
YOLO_ZIPS = [
    "Pemphigoid_yolo.zip",
    "Pemphigus_yolo.zip",
    "lichen_retake_yolo.zip",
    "LE_yolo.zip",
]

TASK_DIR = None          # folder holding YOLO_ZIPS (optional)
OUT_DIR = None
MASK_SOURCES = []
NORMAL_IMG_DIR = None

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def mask_to_yolo_seg(mask_gray, min_area=50):
    """
    Convert a binary mask to YOLO seg polygon lines (class 0).
    Returns list of strings, one per contour region.
    Empty list if no foreground found.
    """
    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = mask_gray.shape
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Simplify contour slightly to reduce point count
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        coords = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in pts)
        lines.append(f"0 {coords}")
    return lines


def remap_label_line(line):
    """Remap any class id to 0 (lesion), keep polygon coords unchanged."""
    line = line.strip()
    if not line:
        return ""
    parts = line.split()
    # parts[0] = class id, rest = polygon coords
    return "0 " + " ".join(parts[1:])


# ── Step 1: extract existing YOLO zips ───────────────────────────────────────

def extract_zips():
    """Extract zips → list of (img_path, label_lines, source_name)."""
    samples = []
    for zip_name in YOLO_ZIPS:
        zip_path = TASK_DIR / zip_name
        if not zip_path.exists():
            print(f"  [skip] not found: {zip_path}")
            continue

        with zipfile.ZipFile(zip_path) as z:
            entries = z.namelist()
            img_entries   = [e for e in entries if Path(e).suffix.lower() in EXTS
                             and "/images/" in e]
            label_entries = {Path(e).stem: e for e in entries if e.endswith(".txt")
                             and "/labels/" in e}

            extracted = 0
            for img_entry in img_entries:
                stem = Path(img_entry).stem
                lbl_entry = label_entries.get(stem)

                img_bytes = z.read(img_entry)
                lbl_lines = []
                if lbl_entry:
                    raw = z.read(lbl_entry).decode("utf-8")
                    lbl_lines = [remap_label_line(l) for l in raw.splitlines() if l.strip()]

                samples.append({
                    "stem":   stem,
                    "suffix": Path(img_entry).suffix,
                    "img_bytes": img_bytes,
                    "label_lines": lbl_lines,
                    "source": zip_name,
                })
                extracted += 1

        print(f"  {zip_name}: {extracted} images")
    return samples


# ── Step 2: generate from pixel masks ────────────────────────────────────────

def generate_from_masks():
    """Read lichen/other masks → YOLO seg polygons."""
    samples = []
    for src in MASK_SOURCES:
        img_dir  = src["images"]
        mask_dir = src["masks"]
        if not img_dir.exists():
            print(f"  [skip] {img_dir} not found")
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in EXTS:
                continue
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                print(f"  [warn] no mask for {img_path.name}, skipping")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            lines = mask_to_yolo_seg(mask)
            samples.append({
                "stem":        img_path.stem,
                "suffix":      img_path.suffix,
                "img_path":    img_path,
                "label_lines": lines,
                "source":      img_dir.name,
            })

    print(f"  mask-derived: {len(samples)} images")
    return samples


# ── Step 3: normal images as hard-negatives ───────────────────────────────────

def collect_normals():
    samples = []
    if not NORMAL_IMG_DIR.exists():
        return samples
    for img_path in sorted(NORMAL_IMG_DIR.iterdir()):
        if img_path.suffix.lower() not in EXTS:
            continue
        samples.append({
            "stem":        img_path.stem,
            "suffix":      img_path.suffix,
            "img_path":    img_path,
            "label_lines": [],   # empty = no lesion
            "source":      "normal",
        })
    print(f"  normal (hard-neg): {len(samples)} images")
    return samples


# ── Step 4: deduplicate by stem ───────────────────────────────────────────────

def deduplicate(all_samples):
    """
    If the same image stem appears in both a zip AND mask-derived,
    prefer the zip version (manually annotated).
    """
    seen = {}
    for s in all_samples:
        stem = s["stem"]
        if stem not in seen:
            seen[stem] = s
        else:
            # prefer zip (has_bytes) over mask-derived
            if "img_bytes" in s and "img_bytes" not in seen[stem]:
                seen[stem] = s
    deduped = list(seen.values())
    print(f"  after dedup: {len(deduped)} unique images")
    return deduped


# ── Step 5: split & write ─────────────────────────────────────────────────────

def write_dataset(all_samples):
    random.seed(SEED)

    # Separate lesion vs normal for stratified split
    lesion  = [s for s in all_samples if s["label_lines"]]
    normal  = [s for s in all_samples if not s["label_lines"]]
    random.shuffle(lesion); random.shuffle(normal)

    def split(lst):
        n = max(1, int(len(lst) * VAL_SPLIT))
        return lst[n:], lst[:n]

    train_l, val_l = split(lesion)
    train_n, val_n = split(normal)
    train = train_l + train_n
    val   = val_l   + val_n
    random.shuffle(train); random.shuffle(val)

    print(f"\n  train={len(train)} (lesion={len(train_l)} normal={len(train_n)})")
    print(f"  val  ={len(val)}   (lesion={len(val_l)}   normal={len(val_n)})")

    for split_name, split_samples in [("train", train), ("val", val)]:
        img_dir = OUT_DIR / "images" / split_name
        lbl_dir = OUT_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for s in split_samples:
            fname = s["stem"] + s["suffix"]
            lbl_fname = s["stem"] + ".txt"

            # Write image
            dst_img = img_dir / fname
            if "img_bytes" in s:
                dst_img.write_bytes(s["img_bytes"])
            else:
                shutil.copy2(s["img_path"], dst_img)

            # Write label
            (lbl_dir / lbl_fname).write_text("\n".join(s["label_lines"]))

    # data.yaml
    data_yaml = {
        "path":  str(OUT_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["lesion"],
    }
    with open(OUT_DIR / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nDone -> {OUT_DIR}")
    print(f"Next: python paper/train_yolo.py --dataset {OUT_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global TASK_DIR, OUT_DIR, MASK_SOURCES, NORMAL_IMG_DIR
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("yolo_seg_dataset"))
    ap.add_argument("--yolo-zips-dir", type=Path, default=None)
    a = ap.parse_args()
    base = a.data_root
    TASK_DIR = a.yolo_zips_dir or base / "_no_zips"
    OUT_DIR = a.out
    MASK_SOURCES = [
        {"images": base / "lichen_images", "masks": base / "lichen_masks"},
        {"images": base / "other_images",  "masks": base / "other_masks"},
    ]
    NORMAL_IMG_DIR = base / "normal_images"

    print("=== Collecting from zips ===")
    zip_samples  = extract_zips()

    print("\n=== Generating from pixel masks ===")
    mask_samples = generate_from_masks()

    print("\n=== Normal hard-negatives ===")
    norm_samples = collect_normals()

    print("\n=== Deduplicating ===")
    all_samples = deduplicate(zip_samples + mask_samples + norm_samples)

    print("\n=== Writing dataset ===")
    write_dataset(all_samples)


if __name__ == "__main__":
    main()
