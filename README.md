# Oral Lichen Detection System — Stage 3

A two-stage AI system for detecting and segmenting oral lichen planus lesions from intraoral dental photographs. Built for single-patient longitudinal monitoring.

---

## Overview

Oral lichen planus is a chronic inflammatory condition affecting the oral mucosa. Early and accurate detection is critical for treatment planning and monitoring disease progression. This system combines a fast lesion detector (YOLOv8) with a precise pixel-level segmentation model (5-fold UNet ensemble) to produce colour-coded overlay maps on uploaded images.

```
Input image
    │
    ▼
┌─────────────┐
│  YOLOv8-seg │  ── detects lesion region(s), draws bounding box
└──────┬──────┘
       │  crop + padding
       ▼
┌──────────────────────────┐
│  UNet 5-fold Ensemble    │  ── segments pixels: lichen / other / normal
│  (EfficientNet-B0)       │
└──────────────────────────┘
       │
       ▼
Colour overlay on original image
  🔴 Red   = Lichen planus
  🟢 Green = Other lesion
  ⬜ None  = Normal mucosa
```

---

## Features

- **Two-stage pipeline** — YOLO acts as a fast gate, UNet provides precise pixel-level classification
- **5-fold ensemble** — averages predictions from 5 independently trained UNet checkpoints for stability
- **Test-time augmentation (TTA)** — horizontal and vertical flip augmentation at inference for ~1% F1 boost
- **Small blob filtering** — removes isolated false-positive predictions below a configurable pixel threshold (200 px for OLP, 80 px for other)
- **Fallback mode** — if YOLO detects nothing, UNet runs on the full image automatically with a stricter OLP threshold (0.80)
- **Adjustable thresholds** — YOLO confidence, lichen probability, minimum blob size all tunable via sidebar
- **Batch upload** — process multiple images at once, displayed in a responsive grid
- **Feedback collection** — clinicians can mark predictions as correct or incorrect, with reasons, saving images and metadata locally for future retraining

---

## Model Details

> The configuration, weights, training/evaluation scripts and results of the released model are in
> [`ml/`](ml/README.md) (git tag `v1.0`). Parameters are defined once in
> [`ml/configs/final_config.yaml`](ml/configs/final_config.yaml).

### Stage 1 — YOLOv8s-seg (Gate Detector)
| Property | Value |
|---|---|
| Architecture | YOLOv8s-seg (start from `yolov8s-seg.pt`) |
| Input size | 640 × 640 |
| Classes | 1 (lesion) |
| Training | 120 epochs, batch size 16, AdamW (lr0 5e-4), patience 25 |
| Training data | 783 images: **598 train + 185 validation** (all lesion images + 20 % background images in train) |
| Labels source | Manual YOLO seg annotations + polygons derived from the pixel masks |
| Inference | confidence **0.15**, ROI padding **40 px** |
| Validation (box) | P 0.735 · R 0.625 · mAP50 0.682 · mAP50-95 0.304 |
| Validation (mask) | P 0.712 · R 0.617 · mAP50 0.640 · mAP50-95 0.269 |

Validation metrics of `models/yolo_best.pt` on the 185 validation images (`ml/eval_yolo_val.py`;
[`ml/results/yolo_val_metrics.json`](ml/results/yolo_val_metrics.json)). The validation split also selects the
checkpoint, so these are not independent test metrics.

### Stage 2 — UNet 5-Fold Ensemble (Segmentation)
| Property | Value |
|---|---|
| Architecture | UNet + EfficientNet-B0 encoder |
| Input size | **384 × 384** |
| Classes | 3 (normal / lichen / other) |
| Training data | 941 images (normal / OLP / other), 5-fold stratified cross-validation |
| Loss | HybridLoss = CrossEntropy(label_smoothing=0.1) + TverskyLoss(α=0.3, β=0.7) |
| CV mean F1 (pixel-level OLP) | 0.631 ± 0.033 |
| OLP threshold | 0.65 (softmax probability); **0.80** when YOLO finds nothing and UNet runs on the whole image |
| Noise filter | components < **200 px** (OLP) and < **80 px** (other) removed |
| Image-level label | OLP and other area both < **0.5 %** → normal, else the larger area wins |

### Independent test set (89 images: 30 normal, 29 OLP, 30 other)
Final pipeline, parameters above (`python ml/evaluate.py`, full tables in
[`ml/results/metrics_summary.md`](ml/results/metrics_summary.md)):

| Image level (one-vs-rest) | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| OLP | 73.0 % | 69.0 % | 75.0 % | 57.1 % | 0.625 |
| Other | 84.3 % | 80.0 % | 86.4 % | 75.0 % | 0.774 |
| Normal | 86.5 % | 66.7 % | 96.6 % | 90.9 % | 0.769 |

3-class accuracy 71.9 %, macro-F1 0.723.

| Pixel level | Dice (mean ± SD) | IoU (mean ± SD) | Specificity on lesion-free images |
|---|---|---|---|
| OLP (29 images) | 0.541 ± 0.361 | 0.448 ± 0.327 | 98.8 % |
| Other (30 images) | 0.538 ± 0.323 | 0.431 ± 0.298 | 99.1 % |

---

## Project Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── feedback.py             # Feedback widget and data saving logic
├── requirements.txt        # Python dependencies of the web app
├── LICENSE                 # AGPL-3.0
├── README.md               # This file
├── ml/                  # Config, training / evaluation code, results (see ml/README.md)
├── models/
│   ├── yolo_best.pt        # YOLOv8s-seg checkpoint
│   ├── unet_folds_v2/
│   │   └── UNet_fold0_best.pth ... UNet_fold4_best.pth   # 5-fold UNet (default in the app)
│   └── unet_folds/         # earlier 5-fold run (v1, selectable in the app) + UNet++ comparison
└── feedback_data/          # git-ignored; local-only fallback, clinical data must never be committed
```

---

## Setup

### 1. Install dependencies

```bash
# Install PyTorch with CUDA (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining packages
pip install -r requirements.txt
```

### 2. Place model checkpoints

```
models/
├── yolo_best.pt                    ← download from Kaggle training output
└── unet_folds_v2/
    ├── UNet_fold0_best.pth         ← from Kaggle: /kaggle/working/UNet_fold0_best.pth
    ├── UNet_fold1_best.pth
    ├── UNet_fold2_best.pth
    ├── UNet_fold3_best.pth
    └── UNet_fold4_best.pth
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Upload** one or more intraoral images (JPG or PNG)
2. Adjust settings in the **sidebar** if needed (defaults work well)
3. The app shows:
   - Original image
   - Crop panel(s) from YOLO detection with UNet overlay
   - Full-image overlay with red (lichen) and green (other) regions
   - Percentage of image area classified as lichen / other
4. Provide **feedback** on each prediction using the widget below each image
5. Download all feedback as a ZIP from the sidebar for future retraining

---

## Sidebar Controls

| Control | Default | Description |
|---|---|---|
| YOLO confidence | 0.15 | Lower catches more lesions, higher reduces false alarms |
| YOLO crop padding | 40 px | Extra margin around detected bbox before UNet |
| Lichen threshold | 0.65 | Min softmax probability to classify a pixel as lichen |
| Min lichen blob (pixels) | 200 | Remove lichen predictions smaller than this area |
| Min other-lesion blob (pixels) | 80 | Remove "other" predictions smaller than this area |
| TTA | On | Test-time augmentation (flip ensemble at inference) |
| YOLO gate | On | Disable to run UNet on full image always |
| Images per row | 3 | Layout control |

---

## Feedback System

Clinicians can mark each prediction as **Correct** or **Incorrect**. If incorrect, they select the reason:

| Reason | Saved to |
|---|---|
| Wrong mask | `feedback_data/UNet_Bad_Mask/` |
| Wrong class | `feedback_data/UNet_Wrong_Class/` |

Each feedback entry saves the original image, overlay image, and a JSON metadata file. Use the **Download Feedback ZIP** button to export all collected data for retraining.

---

## Training

Training, data-split and evaluation code is in [`ml/`](ml/README.md):

| Script | Purpose |
|---|---|
| `ml/prepare_yolo_data.py` | Build the YOLO seg dataset from pixel masks / annotations, 80/20 split (seed 42) |
| `ml/train_yolo.py` | 20 % background filter + YOLOv8s-seg training (120 epochs, batch 16, 640 px) |
| `ml/split_unet_folds.py` | Stratified 5-fold split of the U-Net data (`ml/data/folds.csv`) |
| `ml/train_unet_5fold.py` | 5-fold U-Net (EfficientNet-B0, 384 px) training |
| `ml/evaluate.py` | Dice, IoU, accuracy, sensitivity, specificity, precision, F1 and confusion matrices on the test set |
| `ml/eval_yolo_val.py` | YOLO box/mask precision, recall, mAP on the validation split |
| `ml/olp_pipeline.py` | Standalone inference pipeline (no web dependencies) |

Exact library versions: `ml/requirements-ml.txt`. Data (clinical images) are not distributed, see `ml/data/README.md`.

---

## License

[GNU AGPL-3.0](LICENSE). The detector is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0),
so the code and the weights derived from it are released under the same license. The clinical images, masks and annotations used
to train/evaluate the models are not covered by this license and are not part of the repository.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4 GB (GTX 1050 Ti) | 8 GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB (models + data) | — |

Inference on GTX 1050 Ti: ~1–2 seconds per image (5-model ensemble + TTA).  
CPU-only mode is supported but ~5–10× slower.
