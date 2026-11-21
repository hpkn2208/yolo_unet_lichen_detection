# Evaluation summary (89 images)

## Image-level (label from area rule)

Confusion matrix (rows = ground truth, columns = prediction):

| GT \ Pred | normal | olp | other |
|---|---|---|---|
| normal | 20 | 9 | 1 |
| olp | 2 | 20 | 7 |
| other | 0 | 6 | 24 |

3-class accuracy: **71.9%**, macro-F1: **0.723**

| Class (one-vs-rest) | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| normal | 86.5% | 66.7% | 96.6% | 90.9% | 0.769 |
| olp | 73.0% | 69.0% | 75.0% | 57.1% | 0.625 |
| other | 84.3% | 80.0% | 86.4% | 75.0% | 0.774 |

## Pixel-level segmentation

| Class | n lesion images | Dice (mean ± SD) | IoU (mean ± SD) | Specificity on lesion-free images | Pooled Sens | Pooled Spec | Pooled Prec | Pooled F1 | Pooled Acc |
|---|---|---|---|---|---|---|---|---|---|
| olp | 29 | 0.541 ± 0.361 | 0.448 ± 0.327 | 98.8% | 59.4% | 98.4% | 79.7% | 0.681 | 94.7% |
| other | 30 | 0.538 ± 0.323 | 0.431 ± 0.298 | 99.1% | 69.8% | 97.8% | 61.2% | 0.652 | 96.4% |

Dice/IoU are averaged per image over the images whose ground truth contains the class. Pooled columns use all pixels of all images.
