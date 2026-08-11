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
- **Small blob filtering** — removes isolated false-positive predictions below a configurable pixel threshold
- **Fallback mode** — if YOLO detects nothing, UNet runs on the full image automatically
- **Adjustable thresholds** — YOLO confidence, lichen probability, minimum blob size all tunable via sidebar
- **Batch upload** — process multiple images at once, displayed in a responsive grid
- **Feedback collection** — clinicians can mark predictions as correct or incorrect, with reasons, saving images and metadata locally for future retraining

---

## Model Details

### Stage 1 — YOLOv8s-seg (Gate Detector)
| Property | Value |
|---|---|
| Architecture | YOLOv8s-seg |
| Input size | 640 × 640 |
| Classes | 1 (lesion) |
| Training data | ~746 images (lichen + other + normal) |
| Labels source | Manual YOLO seg annotations |
| mAP50 (seg) | 0.640 |
| Precision | 0.735 |
| Recall | 0.625 |

### Stage 2 — UNet 5-Fold Ensemble (Segmentation)
| Property | Value |
|---|---|
| Architecture | UNet + EfficientNet-B0 encoder |
| Input size | 256 × 256 |
| Classes | 3 (normal / lichen / other) |
| Training data | ~932 samples (5-fold cross-validation) |
| Loss | HybridLoss = CrossEntropy(label_smoothing=0.1) + TverskyLoss(α=0.3, β=0.7) |
| CV mean F1 | 0.645 ± 0.025 |
| Production threshold | 0.65 (lichen softmax probability) |
| Min blob filter | 200 pixels |

---

## Project Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── feedback.py             # Feedback widget and data saving logic
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/
│   ├── yolo_best.pt        # YOLOv8s-seg checkpoint
│   └── unet_folds/
│       ├── UNet_fold0_best.pth
│       ├── UNet_fold1_best.pth
│       ├── UNet_fold2_best.pth
│       ├── UNet_fold3_best.pth
│       └── UNet_fold4_best.pth
└── feedback_data/          # Auto-created, stores clinician feedback
    ├── Success_Data/
    ├── YOLO_FN/            # Cases where YOLO missed a lesion
    ├── YOLO_FP/            # Cases where YOLO falsely detected a lesion
    ├── UNet_Bad_Mask/      # Cases where segmentation mask was wrong
    └── UNet_Wrong_Class/   # Cases where lichen/other was misclassified
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
└── unet_folds/
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
| Min blob pixels | 200 | Remove lichen predictions smaller than this area |
| TTA | On | Test-time augmentation (flip ensemble at inference) |
| YOLO gate | On | Disable to run UNet on full image always |
| Images per row | 3 | Layout control |

---

## Feedback System

Clinicians can mark each prediction as **Correct** or **Incorrect**. If incorrect, they select the reason:

| Reason | Saved to |
|---|---|
| YOLO missed lesion | `feedback_data/YOLO_FN/` |
| YOLO false alarm | `feedback_data/YOLO_FP/` |
| Wrong lichen mask | `feedback_data/UNet_Bad_Mask/` |
| Wrong class | `feedback_data/UNet_Wrong_Class/` |

Each feedback entry saves the original image, overlay image, and a JSON metadata file. Use the **Download Feedback ZIP** button to export all collected data for retraining.

---

## Training

Training notebooks are in the parent directory:

| Notebook | Purpose |
|---|---|
| `lichen-unet-training.ipynb` | 5-fold CV training for UNet ensemble (run on Kaggle) |
| `kaggle_yolo_train.ipynb` | YOLOv8s-seg training (run on Kaggle) |
| `evaluate_ensemble.py` | Local threshold tuning, TTA, blob filter evaluation |
| `gen_yolo_labels.py` | Generate YOLO seg labels from pixel masks |
| `check_yolo_labels.py` | Visual QC of YOLO labels vs true masks |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4 GB (GTX 1050 Ti) | 8 GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB (models + data) | — |

Inference on GTX 1050 Ti: ~1–2 seconds per image (5-model ensemble + TTA).  
CPU-only mode is supported but ~5–10× slower.
