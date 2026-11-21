# Training notebooks

The two Kaggle (GPU) notebooks that trained the released models. Script versions of the same
pipelines, with command-line arguments, are in [`../ml/`](../ml/README.md).

| Notebook | Trains | Output |
|---|---|---|
| [`yolov8-training.ipynb`](yolov8-training.ipynb) | YOLOv8s-seg lesion detector (1 class): 20 % background filter on the train split, 120 epochs, batch 16, 640 px, AdamW lr0 5e-4, patience 25 | `yolo_best.pt` → `models/yolo_best.pt` |
| [`5fold-unet.ipynb`](5fold-unet.ipynb) | 5-fold U-Net, EfficientNet-B0 encoder, 3 classes (normal / OLP / other), 384 px, hybrid CE + Tversky loss; mean pixel-level OLP F1 0.631 ± 0.033 over the 5 validation folds | `UNet_fold{0-4}_best.pth` → `models/unet_folds_v2/` |

## Running them

* Dataset paths contain the placeholder `<kaggle-user>`; point them at your own Kaggle dataset.
* YOLO dataset layout: `images/{train,val}`, `labels/{train,val}`, `data.yaml` (build it with `ml/prepare_yolo_data.py`).
* U-Net dataset layout: `{normal,lichen,other}/{images,masks}/`.
* The images, masks and annotations are clinical data and are not distributed (see [`../ml/data/README.md`](../ml/data/README.md)).
* The U-Net notebook keeps its cell outputs (training log, learning curves, cross-validation summary); the YOLO notebook was saved without outputs.
