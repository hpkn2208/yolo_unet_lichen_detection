# Oral Lesion Segmentation Model — Performance Summary

## What the model does

Given a photo of the oral cavity, the model locates and outlines (pixel-by-pixel) two types of lesions — **lichen planus** and **other oral lesions** — and distinguishes them from healthy tissue. The delivered pipeline runs in two stages: a detector locates the lesion region(s) in the photo, then a segmentation model traces the precise lesion boundary within that region.

## Test setup

- Evaluated on an **89-image held-out test set** never seen during training: 30 healthy (normal) images, 29 lichen planus images, 30 other-lesion images.
- Model: 5-model ensemble (averaged predictions), trained on a separate training set.

## Headline results

| Lesion type | Metric | Whole-image model | **Detection + segmentation pipeline (final)** |
|---|---|---|---|
| Lichen planus | Dice (overlap score) | 0.57 | **0.68** |
| Lichen planus | IoU (overlap score) | 0.47 | **0.56** |
| Other lesions | Dice (overlap score) | 0.37 | **0.46** |
| Other lesions | IoU (overlap score) | 0.28 | **0.36** |

| Lesion type | Specificity (false-alarm avoidance on healthy tissue) |
|---|---|
| Lichen planus | 92.3% |
| Other lesions | 96.4% |

**How to read this**: Dice and IoU both measure how closely the model's outlined region matches the true lesion boundary drawn by a clinician — 1.0 would be a perfect pixel-for-pixel match, 0 means no overlap at all. Dice and IoU are only calculated on images that actually contain that lesion type (29 lichen images, 30 other-lesion images) — this is standard practice, since there's nothing to measure overlap against on a lesion-free image. For lesion-free images, we instead report **Specificity** — how often the model correctly leaves healthy tissue unmarked. Both lesion types score above 92% specificity, meaning false alarms on clean tissue are not a significant problem.

## What can be counted on

- The **detection + segmentation pipeline is the number to cite** — it's a real, measured improvement over the whole-image model (+18% relative Dice for lichen, +24% relative Dice for other), not a projection.
- **Lichen planus segmentation is the stronger result** of the two lesion types (Dice 0.68, IoU 0.56) and is production-ready for use cases where approximate lesion outlining is acceptable.
- **Other-lesion segmentation is usable but weaker** (Dice 0.46). This category groups several visually distinct conditions together, so it's inherently a harder target than lichen planus alone — this is a known limitation, not a defect, and is the area to prioritize for improvement (more training data per condition, or splitting it into finer categories) if higher accuracy is required there.
- Both categories reliably avoid false-positive flags on healthy tissue (>92% specificity).

## Limitations to disclose

- Test set size is moderate (29-30 images per lesion type) — treat the reported numbers as representative, not to 3-decimal-place precision.
- "Other lesions" is a mixed category covering multiple distinct conditions; the reported score is an average across that mix, not a guarantee for any specific sub-condition.
- Current inference speed was measured on CPU only (no GPU acceleration); expect substantially faster inference once deployed with GPU.
