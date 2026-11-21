"""Train the YOLOv8s-seg lesion detector (single class "lesion") — script version of the Kaggle
notebook that produced models/yolo_best.pt (run name `lesion_seg_v2`).

    python paper/train_yolo.py --dataset /path/to/yolo_seg_dataset --work-dir runs/yolo

DATASET layout (made by prepare_yolo_data.py): images/{train,val}, labels/{train,val}, data.yaml.
Step 1 keeps all lesion images of the training split and only 20% (by count) of the background
(normal / empty-label) images -> 588 train images + 185 val images (773) in the paper version.
Step 2 trains: yolov8s-seg.pt start, 120 epochs, batch 16, imgsz 640, patience 25, AdamW lr0 5e-4.
Seed: the Ultralytics default (0), as in args.yaml of the original run.
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def filter_background(dataset: Path, work_dir: Path, bg_ratio: float = 0.2) -> Path:
    """Copy the dataset to work_dir and drop surplus background images from the TRAIN split."""
    out = work_dir / "dataset_filtered"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dataset, out)
    img_dir, lbl_dir = out / "images" / "train", out / "labels" / "train"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    lesion = [p for p in imgs if (lbl_dir / f"{p.stem}.txt").exists()
              and (lbl_dir / f"{p.stem}.txt").stat().st_size > 0]
    lesion_set = set(lesion)
    background = [p for p in imgs if p not in lesion_set]
    keep = int(len(lesion) * bg_ratio)
    for p in background[keep:]:
        p.unlink(missing_ok=True)
        (lbl_dir / f"{p.stem}.txt").unlink(missing_ok=True)
    print(f"train: lesion={len(lesion)} background {len(background)} -> kept {keep}")

    cfg = yaml.safe_load((dataset / "data.yaml").read_text())
    cfg["path"] = str(out.resolve())
    cfg["train"], cfg["val"] = "images/train", "images/val"
    data_yaml = work_dir / "data_filtered.yaml"
    data_yaml.write_text(yaml.dump(cfg, default_flow_style=False))
    return data_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("runs/yolo"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = filter_background(args.dataset, args.work_dir)
    model = YOLO("yolov8s-seg.pt")            # small > nano: better recall on a small dataset
    results = model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=640, batch=16, patience=25,
        optimizer="AdamW", lr0=5e-4, lrf=0.01, weight_decay=5e-4, warmup_epochs=5,
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.4, degrees=20, translate=0.15, scale=0.4,
        flipud=0.3, fliplr=0.5, mosaic=1.0, copy_paste=0.5,
        project=str(args.work_dir.resolve()), name="lesion_seg_v2", exist_ok=True,
        device=args.device, workers=2,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print("Best weights:", best)

    # Validation metrics on the validation split (box + mask)
    m = YOLO(str(best)).val(data=str(data_yaml), imgsz=640, device=args.device, verbose=False)
    print(f"box  P={m.box.mp:.4f} R={m.box.mr:.4f} mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")
    print(f"mask P={m.seg.mp:.4f} R={m.seg.mr:.4f} mAP50={m.seg.map50:.4f} mAP50-95={m.seg.map:.4f}")


if __name__ == "__main__":
    main()
