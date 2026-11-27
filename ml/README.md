# Training, evaluation and reproduction guide

Code, configuration and weights of the released model (git tag `v1.0`).
Later development of the web apps happens on `main` and does not change this tag.

> Clinical images, masks, patient information and feedback data are **not** in this repository
> (see [data/README.md](data/README.md)).

## 1. Final configuration

All parameters live in one file: [`configs/final_config.yaml`](configs/final_config.yaml).

| Component | Setting |
|---|---|
| Detector | YOLOv8s-seg, 1 class (lesion), 640 × 640, 120 epochs, batch 16, AdamW lr0 5e-4, patience 25 |
| Detector data | 588 train + 185 validation images (773) |
| Inference gate | YOLO confidence **0.15**, ROI padding **40 px** |
| Segmenter | U-Net, EfficientNet-B0 encoder (ImageNet), 3 classes (normal / OLP / other), **384 × 384** input |
| Segmenter data | 939 images (327 normal, 302 OLP, 310 other), **5-fold** stratified CV (seed 42), ensemble of the 5 fold models, TTA = identity + h-flip + v-flip |
| Loss / optimisation | CE (label smoothing 0.1, class weights) + Tversky (α 0.3, β 0.7); AdamW lr 5e-4 (1e-4 after unfreezing), wd 2e-3, cosine, batch 4, ≤ 60 epochs, patience 15; best checkpoint by pixel-level OLP F1 |
| OLP threshold | softmax p(OLP) ≥ **0.65** (ROI mode); **0.80** when YOLO finds nothing and U-Net runs on the whole image |
| Noise filter | connected components < **200 px** (OLP) and < **80 px** (other) removed, measured on the 384 × 384 crop |
| Image label | both areas < **0.5 %** of the image → normal; otherwise the class with the larger area |
| Independent test set | 89 images: 30 normal, 29 OLP, 30 other; none of them is in the training data (checked by image content, see [`data/README.md`](data/README.md)) |

Weights: `models/yolo_best.pt` (= `best.pt` of the Kaggle run `lesion_seg_v2`, identical tensors) and
`models/unet_folds/UNet_fold{0-4}_best.pth` (output of `training/5fold-unet.ipynb`: scoring the 5 checkpoints on their
validation folds reproduces the notebook's per-fold F1 0.687 / 0.689 / 0.627 / 0.596 / 0.707, mean 0.661 ± 0.047);
SHA-256 in [`weights_sha256.txt`](weights_sha256.txt). The other files in `models/` are not part of the evaluated model:
`UNet++_+_scSE_*` is an architecture comparison, and `unet_folds_v2/` is an experimental run on a larger image set that
includes the 30 Normal test images.

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # or the CPU build
pip install -r ml/requirements-ml.txt
sha256sum -c ml/weights_sha256.txt                    # run from the repository root
```

## 3. Re-run the evaluation (89 test images)

```bash
python ml/evaluate.py --test-dir /path/to/test_set --out ml/results
```

`test_set/{normal,lichen,other}/{images,masks}/` — mask = any non-zero pixel is lesion.
Outputs: `metrics.json`, `metrics_summary.md`, `per_image_results.csv`, `confusion_matrix_*.csv/png`
(Dice, IoU, accuracy, sensitivity, specificity, precision, F1, image-level and pixel-level confusion matrices).
The results shipped in [`results/`](results/) were produced with the weights of this tag; the evaluation is
deterministic (no sampling), so a re-run on the same data and library versions should reproduce them.

Use the model on a single image from Python:

```python
import cv2, sys; sys.path.insert(0, "ml")
from olp_pipeline import OLPPipeline
out = OLPPipeline().predict(cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB))
print(out["label"], out["olp_pct"], out["other_pct"])      # out["mask"]: 0 normal, 1 OLP, 2 other
```

## 4. Retrain from scratch (needs the data)

| Step | Command | Output |
|---|---|---|
| YOLO dataset (80/20 split, seed 42) | `python ml/prepare_yolo_data.py --data-root DATA --out yolo_seg_dataset` | `images/`, `labels/`, `data.yaml` |
| YOLO training (+ 20 % background filter) | `python ml/train_yolo.py --dataset yolo_seg_dataset --work-dir runs/yolo` | `runs/yolo/lesion_seg_v2/weights/best.pt` |
| U-Net 5-fold split | `python ml/split_unet_folds.py --data-root UNET_DATA --out ml/data/folds.csv` | `folds.csv` (shipped) |
| U-Net 5-fold training | `python ml/train_unet_5fold.py --data-root UNET_DATA --out-dir runs/unet` | `runs/unet/UNet_fold{0-4}_best.pth`, `cv_results.csv` |

The models were trained on Kaggle GPUs with the notebooks in [`../training/`](../training/README.md); retraining is not bit-for-bit reproducible across hardware/library versions.
`prepare_yolo_data.py` also merges existing YOLO-format annotation archives (Pemphigoid, Pemphigus, lichen_retake, LE) that
are not distributed here. The 20 % background filter of `train_yolo.py` was checked to reproduce exactly the
training-image list of the original run.

## 5. Known limitations

* Splits are by image, not by patient (patient IDs are not available); fold metrics are measured on the validation
  fold that also selects the checkpoint (slightly optimistic). YOLO validation metrics are likewise validation, not test, metrics.
* The 89-image test set is small (29–30 images per class); no confidence intervals are provided.
* The development set contains a few look-alike images (13 pairs, see [`data/README.md`](data/README.md)); 12 of the pairs fall into
  different folds, so the cross-validation scores are slightly optimistic.

## 6. License

AGPL-3.0 (see [`../LICENSE`](../LICENSE)), the license of Ultralytics YOLO on which the detector and the weights depend.
