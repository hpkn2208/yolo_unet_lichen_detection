# Oral Lesion Segmentation Model — Performance Summary

## What the model does

Given a photo of the oral cavity, the model locates and outlines (pixel-by-pixel) two types of lesions — **lichen planus** and **other oral lesions** — and distinguishes them from healthy tissue. The delivered pipeline runs in two stages: a detector locates the lesion region(s) in the photo, then a segmentation model traces the precise lesion boundary within that region.

## Test setup

- Evaluated on an **89-image test set**: 30 healthy (normal) images, 29 lichen planus images, 30 other-lesion images. None of these images is in the training data (checked by image content, see `ml/data/README.md`).
- Model: YOLOv8s-seg detector + 5-model U-Net ensemble (averaged predictions, 384 x 384 input). All parameters are in `ml/configs/final_config.yaml`; the numbers below are produced by `python ml/evaluate.py` (full tables in `ml/results/`).

## Headline results

| Lesion type | Dice (overlap score) | IoU (overlap score) | Specificity on lesion-free images |
|---|---|---|---|
| Lichen planus (29 images) | 0.564 ± 0.324 | 0.457 ± 0.299 | 96.1% |
| Other lesions (30 images) | 0.492 ± 0.347 | 0.395 ± 0.314 | 99.2% |

Deciding which of the three groups a whole photo belongs to (normal / lichen planus / other lesion) gives **66.3% accuracy** (59 of 89 photos):

| Group | Sensitivity (lesion found) | Specificity (no false alarm) |
|---|---|---|
| Lichen planus | 82.8% | 65.0% |
| Other lesions | 73.3% | 88.1% |
| Normal | 43.3% | 96.6% |

**How to read this**: Dice and IoU both measure how closely the model's outlined region matches the true lesion boundary drawn by a clinician — 1.0 would be a perfect pixel-for-pixel match, 0 means no overlap at all. They are only calculated on images that actually contain that lesion type (29 lichen images, 30 other-lesion images), since there is nothing to measure overlap against on a lesion-free image. The ± value is the spread between images. The pixel-level specificity column shows how rarely healthy pixels are marked as lesion; the whole-photo table shows how often a photo is given the right group.

## What can be counted on

- **Lichen planus is found reliably** (82.8% of lichen photos), but the model often marks healthy mucosa as lichen planus: 14 of 30 normal photos were labelled lichen planus (image-level specificity 65.0%). It is best used as a screening aid whose output a clinician reviews.
- **Outline quality is moderate** (Dice 0.56 for lichen planus, 0.49 for other lesions) and varies a lot from photo to photo (± 0.32 to 0.35): some lesions are outlined well, others poorly.
- **Other lesions are a harder target.** This category groups several visually distinct conditions together, so it is inherently harder than lichen planus alone. More training data per condition, or splitting it into finer categories, is the area to improve.

## Limitations to disclose

- Test set size is moderate (29-30 images per lesion type) and no confidence intervals are provided — treat the reported numbers as representative, not to 3-decimal-place precision.
- Images were split by photograph, not by patient (no patient identifiers are available), so the check shows that no image is shared between training and test, not that no patient is.
- "Other lesions" is a mixed category covering multiple distinct conditions; the reported score is an average across that mix, not a guarantee for any specific sub-condition.
- Current inference speed was measured on CPU only (no GPU acceleration); expect substantially faster inference once deployed with GPU.
