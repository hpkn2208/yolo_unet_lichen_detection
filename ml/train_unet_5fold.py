"""Train the 5-fold U-Net (EfficientNet-B0) ensemble — script version of the Kaggle notebook
`5fold-unet.ipynb` that produced models/unet_folds_v2/UNet_fold{0-4}_best.pth (the 5 checkpoints
reproduce the notebook's per-fold validation F1: 0.579 / 0.641 / 0.659 / 0.656 / 0.620).

    python ml/train_unet_5fold.py --data-root /path/to/dataset --out-dir runs/unet

DATA_ROOT layout: {normal,lichen,other}/{images,masks}/ ; mask training target = (mask > 0) * class id.
Final settings: 384x384 input, batch 4, AdamW lr 5e-4 / wd 2e-3, cosine schedule, 60 epochs max,
early stopping patience 15 on the pixel-level OLP F1 of the validation fold, encoder frozen for the
first 5 epochs then unfrozen with lr/5, loss = CE(label smoothing 0.1, inverse-frequency class weights)
+ Tversky(alpha 0.3, beta 0.7), grad clipping 1.0, seed 42, StratifiedKFold(5).

Use --folds 0 1 --epochs 2 --limit 40 for a quick smoke test.
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_unet_folds import CLASSES, N_FOLDS, SEED, collect_samples  # noqa: E402

IMG_SIZE = 384
BATCH_SIZE = 4
EPOCHS = 60
LR = 5e-4
WEIGHT_DECAY = 2e-3
ENCODER_FREEZE_EPOCHS = 5
PATIENCE = 15
NUM_CLASSES = 3
NORM = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.88, 1.12),
             rotate=(-15, 15), p=0.4),
    A.ElasticTransform(alpha=30, sigma=5, p=0.2),
    A.OneOf([
        A.RandomGamma(gamma_limit=(80, 120), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
    ], p=0.8),
    A.CLAHE(clip_limit=(2, 4), tile_grid_size=(8, 8), p=0.4),
    A.OneOf([A.GaussNoise(std_range=(0.01, 0.05), p=0.6), A.GaussianBlur(blur_limit=(3, 5), p=0.4)], p=0.4),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.2),
    # holes are applied to the IMAGE only, the mask keeps the ground truth
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 32), hole_width_range=(16, 32),
                    fill=0, p=0.3),
    A.Normalize(**NORM),
    ToTensorV2(),
])
val_transform = A.Compose([A.Normalize(**NORM), ToTensorV2()])


class SegDataset(Dataset):
    def __init__(self, root: Path, samples, augment):
        self.root, self.samples, self.augment = root, samples, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, cls = self.samples[i]
        img_path = self.root / rel
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask_path = img_path.parent.parent / "masks" / f"{img_path.stem}.png"
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if m is not None:
            m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = (m > 0).astype(np.uint8) * cls
        else:  # missing mask: whole image takes the image class (normal -> 0)
            mask = np.full((IMG_SIZE, IMG_SIZE), cls, dtype=np.uint8)
        out = self.augment(image=img, mask=mask)
        return out["image"], out["mask"].long()


def build_model(arch: str):
    kw = dict(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3,
              classes=NUM_CLASSES, decoder_dropout=0.5)
    if arch == "unet":
        return smp.Unet(**kw)
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(decoder_attention_type="scse", **kw)
    raise ValueError(arch)


class HybridLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        # alpha < beta: false negatives are penalised more than false positives
        self.tversky = smp.losses.TverskyLoss(mode="multiclass", from_logits=True, alpha=0.3, beta=0.7)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.tversky(logits, target)


def class_weights_of(ds: Subset) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in ds.indices:
        _, mask = ds.dataset[i]
        counts += np.bincount(mask.numpy().ravel(), minlength=NUM_CLASSES)
    return torch.tensor(counts.sum() / (NUM_CLASSES * counts + 1e-8), dtype=torch.float32)


def lichen_f1(y_pred: np.ndarray, y_true: np.ndarray, eps=1e-8) -> dict:
    yp, yt = (y_pred == 1).astype(int), (y_true == 1).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    sens, prec = tp / (tp + fn + eps), tp / (tp + fp + eps)
    return {"f1": 2 * prec * sens / (prec + sens + eps), "precision": prec, "sensitivity": sens,
            "specificity": tn / (tn + fp + eps)}


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, y_pred, y_true = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                y_pred.append(logits.argmax(1).cpu().numpy().ravel())
                y_true.append(masks.cpu().numpy().ravel())
            total += loss.item() * imgs.size(0)
    metrics = lichen_f1(np.concatenate(y_pred), np.concatenate(y_true)) if not train else None
    return total / len(loader.dataset), metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/unet"))
    ap.add_argument("--arch", choices=["unet", "unetplusplus"], default="unet")
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="use only N random images (smoke test)")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "UNet" if args.arch == "unet" else "UNet++_+_scSE"

    samples = collect_samples(args.data_root)
    labels = np.array([c for _, c in samples])
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(samples, labels))
    if args.limit:
        keep = set(np.random.RandomState(SEED).choice(len(samples), args.limit, replace=False).tolist())
        splits = [([i for i in tr if i in keep], [i for i in va if i in keep]) for tr, va in splits]
    print(f"{len(samples)} images, classes {dict(zip([n for n, _ in CLASSES], np.bincount(labels)))}")

    train_ds = SegDataset(args.data_root, samples, train_transform)
    val_ds = SegDataset(args.data_root, samples, val_transform)
    summary, history = [], []

    for fold in args.folds:
        tr_idx, va_idx = splits[fold]
        tr, va = Subset(train_ds, tr_idx), Subset(val_ds, va_idx)
        tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"\nFold {fold}  train={len(tr)} val={len(va)}")

        model = build_model(args.arch).to(device)
        for p in model.encoder.parameters():
            p.requires_grad = False
        criterion = HybridLoss(class_weights_of(tr).to(device))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                                weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
        best_f1, bad = -1.0, 0

        for epoch in range(1, args.epochs + 1):
            if epoch == ENCODER_FREEZE_EPOCHS + 1:   # unfreeze encoder, lower lr, restart cosine
                for p in model.encoder.parameters():
                    p.requires_grad = True
                opt = torch.optim.AdamW(model.parameters(), lr=LR / 5, weight_decay=WEIGHT_DECAY)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, args.epochs - epoch), eta_min=1e-6)
            t_loss, _ = run_epoch(model, tr_loader, criterion, device, opt)
            v_loss, m = run_epoch(model, va_loader, criterion, device)
            sched.step()
            history.append(dict(fold=fold, epoch=epoch, train_loss=t_loss, val_loss=v_loss, **m))
            improved = m["f1"] > best_f1 + 1e-4
            if improved:
                best_f1, bad = m["f1"], 0
                torch.save(copy.deepcopy(model.state_dict()), args.out_dir / f"{name}_fold{fold}_best.pth")
            else:
                bad += 1
            print(f"  ep {epoch:3d} train={t_loss:.4f} val={v_loss:.4f} f1={m['f1']:.4f}"
                  f" prec={m['precision']:.3f} sens={m['sensitivity']:.3f}{'  *' if improved else ''}")
            if bad >= PATIENCE:
                print(f"  early stop, best F1={best_f1:.4f}")
                break
        summary.append({"arch": name, "fold": fold, "best_f1": best_f1})

    df = pd.DataFrame(summary)
    df.to_csv(args.out_dir / "cv_results.csv", index=False)
    pd.DataFrame(history).to_csv(args.out_dir / "cv_history.csv", index=False)
    print("\nCV summary (pixel-level OLP F1):", df["best_f1"].agg(["mean", "std", "min", "max"]).round(4).to_dict())


if __name__ == "__main__":
    main()
