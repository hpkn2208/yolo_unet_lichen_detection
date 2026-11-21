"""Validation metrics (box + mask) of the YOLO detector on the validation split.

    python paper/eval_yolo_val.py --dataset /path/to/dataset_filtered --out paper/results/yolo_val_metrics.json

DATASET: images/{train,val}, labels/{train,val} (the filtered dataset written by train_yolo.py).
Note: the validation split is also used by Ultralytics to select best.pt, so these are validation
(not independent test) metrics.
"""
import argparse
import json
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=REPO_ROOT / "models" / "yolo_best.pt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "results" / "yolo_val_metrics.json")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = Path(tmp) / "data.yaml"
        data_yaml.write_text(yaml.dump({"path": str(args.dataset.resolve()), "train": "images/train",
                                        "val": "images/val", "nc": 1, "names": ["lesion"]}))
        m = YOLO(str(args.weights)).val(data=str(data_yaml), imgsz=640, device=args.device, workers=0,
                                        verbose=False, plots=False, project=tmp, name="val")
    out = {
        "split": "validation (images/val; also used to select best.pt)",
        "weights": args.weights.name, "imgsz": 640,
        "box": {"precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map},
        "mask": {"precision": m.seg.mp, "recall": m.seg.mr, "mAP50": m.seg.map50, "mAP50-95": m.seg.map},
    }
    out = json.loads(json.dumps(out, default=float))
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
