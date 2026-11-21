"""Standalone inference pipeline: YOLOv8s-seg gate -> 5-fold U-Net ensemble.

This is the same logic as the deployed Streamlit app (../app.py) without any web
dependency. All parameters come from a YAML config (default: configs/final_config.yaml).

    from olp_pipeline import OLPPipeline
    pipe = OLPPipeline()                      # loads configs/final_config.yaml
    out = pipe.predict(img_rgb)               # img_rgb: HxWx3 uint8, RGB
    out["mask"]        # HxW uint8: 0 normal, 1 OLP, 2 other
    out["olp_pct"], out["other_pct"], out["label"], out["yolo_detected"]
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "final_config.yaml"

_NORM = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
_TTA = [
    A.Compose([A.Normalize(**_NORM), ToTensorV2()]),
    A.Compose([A.HorizontalFlip(p=1), A.Normalize(**_NORM), ToTensorV2()]),
    A.Compose([A.VerticalFlip(p=1), A.Normalize(**_NORM), ToTensorV2()]),
]
_PLAIN = [_TTA[0]]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def remove_small_blobs(mask: np.ndarray, min_olp: int, min_other: int) -> np.ndarray:
    """Remove 8-connected components of class 1 / 2 whose area is below the given thresholds."""
    out = mask.copy()
    for cls, min_px in ((1, min_olp), (2, min_other)):
        if min_px <= 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask == cls).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_px:
                out[labels == i] = 0
    return out


def image_label(olp_pct: float, other_pct: float, area_threshold_pct: float) -> str:
    """Image-level label from area percentages (same rule as the app's 'Evaluation label')."""
    if olp_pct < area_threshold_pct and other_pct < area_threshold_pct:
        return "normal"
    return "olp" if olp_pct >= other_pct else "other"


class OLPPipeline:
    def __init__(self, config: str | Path | dict = DEFAULT_CONFIG, device: str | None = None):
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        c = self.cfg
        self.yolo = YOLO(str(REPO_ROOT / c["yolo"]["weights"]))
        self.unets = []
        for k in range(c["unet"]["n_folds"]):
            ckpt = REPO_ROOT / c["unet"]["weights_dir"] / c["unet"]["weights_pattern"].format(k=k)
            m = smp.Unet(encoder_name=c["unet"]["encoder"], encoder_weights=None, in_channels=3,
                         classes=c["unet"]["num_classes"], decoder_dropout=0.5)
            m.load_state_dict(torch.load(str(ckpt), map_location=self.device))
            self.unets.append(m.to(self.device).eval())

    # -- stage 2 on one (already cropped) RGB image ---------------------------------------
    @torch.no_grad()
    def _segment(self, img_rgb: np.ndarray, olp_threshold: float) -> np.ndarray:
        size = self.cfg["unet"]["img_size"]
        img = cv2.resize(img_rgb, (size, size))
        tfs = _TTA if self.cfg["unet"]["tta"] else _PLAIN
        acc = None
        for m in self.unets:
            for i, tf in enumerate(tfs):
                x = tf(image=img)["image"].unsqueeze(0).to(self.device)
                p = F.softmax(m(x), dim=1).squeeze(0).cpu()
                if i == 1:
                    p = p.flip(-1)      # undo hflip
                if i == 2:
                    p = p.flip(-2)      # undo vflip
                acc = p if acc is None else acc + p
        probs = (acc / (len(self.unets) * len(tfs))).numpy()
        pred = np.zeros((size, size), dtype=np.uint8)
        pred[probs[1] >= olp_threshold] = 1
        other = (probs[2] > probs[0]) & (probs[2] > probs[1])
        pred[other & (pred == 0)] = 2
        pp = self.cfg["postprocess"]
        return remove_small_blobs(pred, pp["min_blob_olp_px"], pp["min_blob_other_px"])

    def _yolo_boxes(self, img_bgr: np.ndarray):
        H, W = img_bgr.shape[:2]
        pad = self.cfg["yolo"]["roi_padding_px"]
        res = self.yolo(img_bgr, conf=self.cfg["yolo"]["confidence"], verbose=False,
                        device=self.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = []
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            boxes.append((max(0, int(x1) - pad), max(0, int(y1) - pad),
                          min(W, int(x2) + pad), min(H, int(y2) + pad)))
        return boxes

    def predict(self, img_rgb: np.ndarray) -> dict:
        H, W = img_rgb.shape[:2]
        pp = self.cfg["postprocess"]
        boxes = self._yolo_boxes(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        full = np.zeros((H, W), dtype=np.uint8)
        if boxes:
            for x1, y1, x2, y2 in boxes:
                m = self._segment(img_rgb[y1:y2, x1:x2], pp["olp_threshold"])
                m = cv2.resize(m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                full[y1:y2, x1:x2] = np.maximum(full[y1:y2, x1:x2], m)
        else:   # YOLO found nothing: U-Net on the whole image with the stricter threshold
            m = self._segment(img_rgb, pp["fallback_olp_threshold"])
            full = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        olp_pct = float((full == 1).mean() * 100)
        other_pct = float((full == 2).mean() * 100)
        return {
            "mask": full,
            "olp_pct": olp_pct,
            "other_pct": other_pct,
            "label": image_label(olp_pct, other_pct, self.cfg["image_label"]["area_threshold_pct"]),
            "yolo_detected": bool(boxes),
            "n_boxes": len(boxes),
        }
