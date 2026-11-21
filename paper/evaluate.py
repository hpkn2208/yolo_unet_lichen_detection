"""Evaluate the final pipeline on an independent test set.

Expected test-set layout (images and masks are NOT part of this repository: clinical data):

    TEST_DIR/
        normal/images/*.jpg|png   normal/masks/*.png   (masks may be absent or all-zero)
        lichen/images/...         lichen/masks/...     (any non-zero pixel = lesion)
        other/images/...          other/masks/...

Run from the repository root:

    python paper/evaluate.py --test-dir /path/to/TEST_DIR --out paper/results

Outputs (in --out): metrics.json, metrics_summary.md, per_image_results.csv,
confusion_matrix_image_level.{csv,png}, confusion_matrix_pixel_level.csv

Metrics
  Pixel level, per class c in {OLP, other}
    * Dice / IoU: per image, averaged over the images whose ground truth contains c
      (undefined on lesion-free images); mean, SD and pooled (all pixels) values are reported.
    * Accuracy / Sensitivity / Specificity / Precision / F1: one-vs-rest on all pixels of all images pooled.
    * Specificity on images that do NOT contain class c (pixels kept negative).
  Image level (label from the area rule, see configs/final_config.yaml)
    * 3x3 confusion matrix, 3-class accuracy, macro-F1.
    * One-vs-rest Sensitivity / Specificity / Precision / F1 / Accuracy for OLP and for other.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olp_pipeline import DEFAULT_CONFIG, REPO_ROOT, OLPPipeline, load_config  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif", ".webp"}
FOLDERS = {"normal": 0, "lichen": 1, "other": 2}
NAMES = ["normal", "olp", "other"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    sens, spec = safe_div(tp, tp + fn), safe_div(tn, tn + fp)
    prec = safe_div(tp, tp + fp)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": safe_div(2 * prec * sens, prec + sens) if prec == prec and sens == sens else float("nan"),
    }


def one_vs_rest(cm: np.ndarray, c: int) -> dict:
    tp = cm[c, c]
    fn = cm[c].sum() - tp
    fp = cm[:, c].sum() - tp
    tn = cm.sum() - tp - fn - fp
    return binary_metrics(tp, fp, fn, tn)


def collect(test_dir: Path):
    items = []
    for folder, cls in FOLDERS.items():
        img_dir = test_dir / folder / "images"
        if not img_dir.is_dir():
            sys.exit(f"Missing folder: {img_dir}")
        for p in sorted(img_dir.iterdir(), key=lambda q: q.name):
            if p.suffix.lower() in EXTS:
                items.append((p, test_dir / folder / "masks" / f"{p.stem}.png", cls))
    return items


def load_gt(mask_path: Path, cls: int, shape: tuple) -> tuple[np.ndarray, bool]:
    """GT class map: (mask > 0) * class id. Returns (map, was_resized)."""
    H, W = shape
    if cls == 0 or not mask_path.exists():
        return np.zeros((H, W), dtype=np.uint8), False
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        sys.exit(f"Unreadable mask: {mask_path}")
    if m.ndim == 3:                      # OpenCV >= 5 returns (H, W, 1) for grayscale reads
        m = m[..., 0]
    resized = m.shape != (H, W)
    if resized:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return ((m > 0).astype(np.uint8) * cls), resized


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "results")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (smoke test)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    pipe = OLPPipeline(cfg)
    items = collect(args.test_dir)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} test images | device={pipe.device}")

    thr = cfg["image_label"]["area_threshold_pct"]
    img_cm = np.zeros((3, 3), dtype=np.int64)
    pix_cm = np.zeros((3, 3), dtype=np.int64)       # rows GT, cols prediction, pooled over all pixels
    rows, per_img_dice, per_img_iou = [], {1: [], 2: []}, {1: [], 2: []}
    neg_tn = {1: 0, 2: 0}
    neg_fp = {1: 0, 2: 0}
    n_resized = 0

    for i, (img_path, mask_path, cls) in enumerate(items, 1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            sys.exit(f"Unreadable image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt, resized = load_gt(mask_path, cls, rgb.shape[:2])
        n_resized += int(resized)
        out = pipe.predict(rgb)
        pred = out["mask"]

        pred_label = {"normal": 0, "olp": 1, "other": 2}[out["label"]]
        img_cm[cls, pred_label] += 1
        pix_cm += np.bincount((gt.astype(np.int64) * 3 + pred).ravel(), minlength=9).reshape(3, 3)

        row = {"image": f"{img_path.parent.parent.name}/{img_path.name}",
               "gt": NAMES[cls], "pred": out["label"], "olp_pct": round(out["olp_pct"], 4),
               "other_pct": round(out["other_pct"], 4), "yolo_boxes": out["n_boxes"]}
        for c in (1, 2):
            g, p = gt == c, pred == c
            inter, union_sum = (g & p).sum(), g.sum() + p.sum()
            if g.any():
                d = 2 * inter / union_sum if union_sum else float("nan")
                u = inter / (g | p).sum()
                per_img_dice[c].append(float(d))
                per_img_iou[c].append(float(u))
                row[f"dice_{NAMES[c]}"], row[f"iou_{NAMES[c]}"] = round(float(d), 4), round(float(u), 4)
            else:
                neg_fp[c] += int(p.sum())
                neg_tn[c] += int((~g & ~p).sum())
        rows.append(row)
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    # -- aggregate ------------------------------------------------------------------------
    metrics = {"n_images": len(items), "n_masks_resized_to_image_size": n_resized,
               "image_level": {}, "pixel_level": {}}
    img_acc = safe_div(np.trace(img_cm), img_cm.sum())
    per_class_img = {NAMES[c]: one_vs_rest(img_cm, c) for c in range(3)}
    metrics["image_level"] = {
        "confusion_matrix_rows_gt_cols_pred": img_cm.tolist(), "labels": NAMES,
        "accuracy_3class": img_acc,
        "macro_f1": float(np.nanmean([per_class_img[n]["f1"] for n in NAMES])),
        "one_vs_rest": per_class_img,
    }
    for c in (1, 2):
        d, u = np.array(per_img_dice[c]), np.array(per_img_iou[c])
        ovr = one_vs_rest(pix_cm, c)
        metrics["pixel_level"][NAMES[c]] = {
            "n_images_with_lesion": len(d),
            "dice_mean": float(np.nanmean(d)), "dice_sd": float(np.nanstd(d, ddof=1)),
            "iou_mean": float(np.mean(u)), "iou_sd": float(np.std(u, ddof=1)),
            "specificity_on_lesion_free_images": safe_div(neg_tn[c], neg_tn[c] + neg_fp[c]),
            "pooled_one_vs_rest": ovr,
            "pooled_dice": safe_div(2 * ovr["tp"], 2 * ovr["tp"] + ovr["fp"] + ovr["fn"]),
            "pooled_iou": safe_div(ovr["tp"], ovr["tp"] + ovr["fp"] + ovr["fn"]),
        }
    metrics["pixel_level"]["confusion_matrix_rows_gt_cols_pred"] = pix_cm.tolist()

    w = [REPO_ROOT / cfg["yolo"]["weights"]] + [
        REPO_ROOT / cfg["unet"]["weights_dir"] / cfg["unet"]["weights_pattern"].format(k=k)
        for k in range(cfg["unet"]["n_folds"])]
    metrics["environment"] = {
        "config": json.loads(json.dumps(cfg)),
        "weights_sha256": {p.name: sha256(p) for p in w},
        "python": platform.python_version(), "torch": torch.__version__, "device": str(pipe.device),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with open(args.out / "per_image_results.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["image", "gt", "pred", "olp_pct", "other_pct", "yolo_boxes",
                "dice_olp", "iou_olp", "dice_other", "iou_other"]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    for name, cm in (("image_level", img_cm), ("pixel_level", pix_cm)):
        with open(args.out / f"confusion_matrix_{name}.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["gt \\ pred"] + NAMES)
            for c in range(3):
                wr.writerow([NAMES[c]] + cm[c].tolist())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(img_cm, cmap="Blues")
        ax.set_xticks(range(3), NAMES)
        ax.set_yticks(range(3), NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, int(img_cm[r, c]), ha="center", va="center",
                        color="white" if img_cm[r, c] > img_cm.max() / 2 else "black")
        ax.set_title(f"Image-level confusion matrix (n={img_cm.sum()})")
        fig.tight_layout()
        fig.savefig(args.out / "confusion_matrix_image_level.png", dpi=200)
    except Exception as e:  # plotting is optional
        print("matplotlib plot skipped:", e)

    write_summary(metrics, args.out / "metrics_summary.md")
    print((args.out / "metrics_summary.md").read_text(encoding="utf-8"))


def fmt(x: float, pct: bool = False) -> str:
    if x != x:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def write_summary(m: dict, path: Path):
    il, pl = m["image_level"], m["pixel_level"]
    L = [f"# Evaluation summary ({m['n_images']} images)", "",
         "## Image-level (label from area rule)", "",
         "Confusion matrix (rows = ground truth, columns = prediction):", "",
         "| GT \\ Pred | normal | olp | other |", "|---|---|---|---|"]
    for i, n in enumerate(il["labels"]):
        L.append(f"| {n} | " + " | ".join(str(v) for v in il["confusion_matrix_rows_gt_cols_pred"][i]) + " |")
    L += ["", f"3-class accuracy: **{fmt(il['accuracy_3class'], True)}**, macro-F1: **{fmt(il['macro_f1'])}**", "",
          "| Class (one-vs-rest) | Accuracy | Sensitivity | Specificity | Precision | F1 |", "|---|---|---|---|---|---|"]
    for n in il["labels"]:
        r = il["one_vs_rest"][n]
        L.append(f"| {n} | {fmt(r['accuracy'], True)} | {fmt(r['sensitivity'], True)} | "
                 f"{fmt(r['specificity'], True)} | {fmt(r['precision'], True)} | {fmt(r['f1'])} |")
    L += ["", "## Pixel-level segmentation", "",
          "| Class | n lesion images | Dice (mean ± SD) | IoU (mean ± SD) | Specificity on lesion-free images | "
          "Pooled Sens | Pooled Spec | Pooled Prec | Pooled F1 | Pooled Acc |", "|" + "---|" * 10]
    for n in ("olp", "other"):
        p = pl[n]
        o = p["pooled_one_vs_rest"]
        L.append(f"| {n} | {p['n_images_with_lesion']} | {fmt(p['dice_mean'])} ± {fmt(p['dice_sd'])} | "
                 f"{fmt(p['iou_mean'])} ± {fmt(p['iou_sd'])} | {fmt(p['specificity_on_lesion_free_images'], True)} | "
                 f"{fmt(o['sensitivity'], True)} | {fmt(o['specificity'], True)} | {fmt(o['precision'], True)} | "
                 f"{fmt(o['f1'])} | {fmt(o['accuracy'], True)} |")
    L += ["", "Dice/IoU are averaged per image over the images whose ground truth contains the class. "
          "Pooled columns use all pixels of all images.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
