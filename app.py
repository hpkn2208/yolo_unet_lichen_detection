"""
Oral Lichen Detection — Stage 3
Pipeline:
    1. YOLOv8-seg  → detect lesion ROI (gate)
    2. UNet 5-fold ensemble → segment & classify (lichen / other / normal)
"""

import os
import io
import base64
import hashlib
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ultralytics import YOLO

from feedback import render_feedback_widget, create_feedback_zip

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Oral Lichen Detector", layout="wide")
st.title("Oral Lichen Detection — Stage 3")
st.write("YOLO gate → 5-fold UNet ensemble segmentation")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path("models")
YOLO_PT     = MODEL_DIR / "yolo_best.pt"
UNET_DIR    = MODEL_DIR / "unet_folds"
N_FOLDS     = 5
IMG_SIZE    = 256
NUM_CLASSES = 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PALETTE = {
    1: np.array([220,  50,  50], dtype=np.uint8),   # lichen — red
    2: np.array([ 50, 200,  80], dtype=np.uint8),   # other  — green
}

val_tf = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

tta_tfs = [
    A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
    A.Compose([A.HorizontalFlip(p=1), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
    A.Compose([A.VerticalFlip(p=1),   A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
]

# ── Sidebar settings ──────────────────────────────────────────────────────────
st.sidebar.header("Settings")

yolo_conf     = st.sidebar.slider("YOLO confidence threshold", 0.05, 0.50, 0.15, 0.05,
                                   help="Lower = catch more lesions (fewer FN), higher = fewer false alarms")
yolo_padding  = st.sidebar.slider("YOLO crop padding (px)", 0, 100, 40, 10,
                                   help="Extra pixels around detected bbox before sending to UNet")
lichen_thresh = st.sidebar.slider("Lichen probability threshold", 0.30, 0.90, 0.65, 0.05,
                                   help="Min lichen softmax prob to colour a pixel red")
min_blob_px   = st.sidebar.slider("Min lesion blob (pixels)", 0, 500, 200, 50,
                                   help="Remove isolated lichen predictions smaller than this")
use_tta       = st.sidebar.checkbox("Test-time augmentation (TTA)", value=True,
                                     help="Average hflip+vflip predictions — ~1% F1 boost")
use_yolo_gate = st.sidebar.checkbox("Enable YOLO gate", value=True,
                                     help="Uncheck to run UNet on full image without YOLO crop")
row_cols      = st.sidebar.selectbox("Images per row", [2, 3, 4], index=1)

# ── Model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_yolo(path):
    if not Path(path).exists():
        return None
    return YOLO(str(path))


@st.cache_resource
def load_unet_ensemble(unet_dir, n_folds, device_str):
    device = torch.device(device_str)
    models = []
    for k in range(n_folds):
        ckpt = Path(unet_dir) / f"UNet_fold{k}_best.pth"
        if not ckpt.exists():
            st.warning(f"UNet fold {k} checkpoint not found: {ckpt}")
            continue
        m = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
            decoder_dropout=0.5,
        ).to(device)
        m.load_state_dict(torch.load(str(ckpt), map_location=device))
        m.eval()
        models.append(m)
    return models


# ── Inference helpers ─────────────────────────────────────────────────────────

def remove_small_blobs(pred_mask, min_pixels):
    if min_pixels <= 0:
        return pred_mask
    out    = pred_mask.copy()
    lichen = (pred_mask == 1).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lichen, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_pixels:
            out[labels == i] = 0
    return out


def predict_unet(models, img_rgb, lichen_threshold, use_tta, device):
    """Return (pred H×W, probs C×H×W) from ensemble average."""
    tfs = tta_tfs if use_tta else [val_tf]
    probs_sum = None
    with torch.no_grad():
        for m in models:
            for i, tf in enumerate(tfs):
                t = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
                p = F.softmax(m(t), dim=1).squeeze(0).cpu()
                if i == 1: p = p.flip(-1)   # undo hflip
                if i == 2: p = p.flip(-2)   # undo vflip
                probs_sum = p if probs_sum is None else probs_sum + p
    probs = (probs_sum / (len(models) * len(tfs))).numpy()  # C×H×W

    pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pred[probs[1] >= lichen_threshold] = 1
    other_m = (probs[2] > probs[0]) & (probs[2] > probs[1])
    pred[other_m & (pred == 0)] = 2
    return pred, probs


def draw_overlay(img_rgb, pred, alpha=0.45):
    out = img_rgb.copy()
    for cls, color in PALETTE.items():
        m = pred == cls
        if m.any():
            out[m] = (img_rgb[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    # draw contour outlines
    for cls, color in PALETTE.items():
        binary = (pred == cls).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color.tolist(), 2)
    return out


def yolo_crop(yolo_model, img_bgr, conf, padding):
    """
    Run YOLO on full image.
    Returns list of (x1,y1,x2,y2) padded bboxes, or None if no detection.
    """
    H, W = img_bgr.shape[:2]
    results = yolo_model(img_bgr, conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    boxes = []
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(W, int(x2) + padding)
        y2 = min(H, int(y2) + padding)
        boxes.append((x1, y1, x2, y2))
    return boxes


def show_image(arr, caption=None):
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    html = "<div style='text-align:center;margin:4px 0'>"
    html += f"<img src='data:image/png;base64,{data}' style='max-width:100%;border-radius:8px'/>"
    if caption:
        html += f"<div style='font-size:12px;color:#aaa;margin-top:3px'>{caption}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def image_id(filename):
    return hashlib.md5(filename.encode()).hexdigest()[:12]


# ── Load models ───────────────────────────────────────────────────────────────
yolo_model   = load_yolo(YOLO_PT)
unet_models  = load_unet_ensemble(str(UNET_DIR), N_FOLDS, str(DEVICE))

with st.sidebar:
    st.divider()
    if yolo_model:
        st.success(f"YOLO loaded ({YOLO_PT.name})")
    else:
        st.warning(f"YOLO not found: {YOLO_PT}")
    st.success(f"UNet: {len(unet_models)}/{N_FOLDS} folds loaded")
    st.info(f"Device: {DEVICE}")

if not unet_models:
    st.error("No UNet checkpoints found. Place them in models/unet_folds/UNet_fold{{k}}_best.pth")
    st.stop()

# ── Feedback download ─────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("### Feedback")
if st.sidebar.button("Download Feedback ZIP"):
    zip_path = create_feedback_zip()
    if zip_path and zip_path.exists():
        with open(zip_path, "rb") as f:
            st.sidebar.download_button(
                "⬇ Download feedback_data.zip",
                data=f.read(),
                file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )
    else:
        st.sidebar.info("No feedback yet.")

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload oral images (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
if not uploaded_files:
    st.info("Upload images to run detection.")
    st.stop()

# ── Process images ────────────────────────────────────────────────────────────
uploaded_files = list(uploaded_files)[::-1]

for i in range(0, len(uploaded_files), row_cols):
    cols = st.columns(row_cols, gap="small", vertical_alignment="top", border=True)
    for j, uf in enumerate(uploaded_files[i:i + row_cols]):
        img_pil  = Image.open(uf).convert("RGB")
        img_rgb  = np.array(img_pil)
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        H, W     = img_rgb.shape[:2]
        iid      = image_id(uf.name)

        col = cols[j]
        with col:
            st.markdown(f"#### {uf.name}")
            show_image(img_rgb, "Original")

            # ── YOLO gate ─────────────────────────────────────────────────
            yolo_boxes    = None
            yolo_detected = False

            if use_yolo_gate and yolo_model:
                yolo_boxes = yolo_crop(yolo_model, img_bgr, yolo_conf, yolo_padding)
                yolo_detected = yolo_boxes is not None
                if yolo_detected:
                    st.markdown(
                        f"<span style='color:#2ecc71;font-weight:bold'>"
                        f"✓ YOLO: {len(yolo_boxes)} lesion region(s) detected</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:#e67e22;font-weight:bold'>"
                        "⚠ YOLO: no lesion detected — running UNet on full image</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<span style='color:#95a5a6'>YOLO gate disabled</span>",
                    unsafe_allow_html=True,
                )

            # ── UNet ensemble ─────────────────────────────────────────────
            # Build canvas at original resolution
            full_pred  = np.zeros((H, W), dtype=np.uint8)
            full_probs = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

            if yolo_boxes:
                # Run UNet on each detected crop, paste back
                for (x1, y1, x2, y2) in yolo_boxes:
                    crop_rgb = img_rgb[y1:y2, x1:x2]
                    crop_256 = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
                    pred_256, probs_256 = predict_unet(
                        unet_models, crop_256, lichen_thresh, use_tta, DEVICE)
                    pred_256 = remove_small_blobs(pred_256, min_blob_px)

                    # Resize pred back to crop size and paste into canvas
                    pred_crop = cv2.resize(
                        pred_256, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    full_pred[y1:y2, x1:x2] = np.maximum(
                        full_pred[y1:y2, x1:x2], pred_crop)

                    # Visualise crop with bounding box
                    vis_crop = draw_overlay(crop_256, pred_256)
                    show_image(vis_crop, f"Crop ({x1},{y1})→({x2},{y2})")
            else:
                # No YOLO detection or gate disabled — full image
                img_256 = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                pred_256, probs_256 = predict_unet(
                    unet_models, img_256, lichen_thresh, use_tta, DEVICE)
                pred_256 = remove_small_blobs(pred_256, min_blob_px)
                full_pred = cv2.resize(
                    pred_256, (W, H), interpolation=cv2.INTER_NEAREST)

            # ── Overlay on original resolution ────────────────────────────
            overlay_rgb = draw_overlay(img_rgb, cv2.resize(
                full_pred, (W, H), interpolation=cv2.INTER_NEAREST))

            lichen_pct = (full_pred == 1).mean() * 100
            other_pct  = (full_pred == 2).mean() * 100

            if lichen_pct > 0.5:
                st.markdown(
                    f"<span style='color:#e74c3c;font-weight:bold'>"
                    f"→ Lichen detected: {lichen_pct:.1f}% of image</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span style='color:#27ae60;font-weight:bold'>"
                    "→ No significant lichen detected</span>",
                    unsafe_allow_html=True,
                )
            if other_pct > 0.5:
                st.markdown(
                    f"<span style='color:#2ecc71'>Other lesion: {other_pct:.1f}%</span>",
                    unsafe_allow_html=True,
                )

            show_image(overlay_rgb, "🔴 Lichen  🟢 Other")

            # ── Feedback ──────────────────────────────────────────────────
            predictions = {
                "yolo_detected":  yolo_detected,
                "yolo_boxes":     yolo_boxes,
                "lichen_pct":     float(lichen_pct),
                "other_pct":      float(other_pct),
                "lichen_thresh":  lichen_thresh,
                "yolo_conf":      yolo_conf,
            }
            models_used = {
                "yolo":  YOLO_PT.name if yolo_model else None,
                "unet":  f"{len(unet_models)}-fold ensemble (efficientnet-b0)",
                "tta":   use_tta,
            }
            render_feedback_widget(
                col, img_rgb, overlay_rgb, iid,
                predictions, uf.name, models_used,
            )

st.success("Done.")
