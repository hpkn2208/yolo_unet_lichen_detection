# Evaluation summary (89 images)

## Image-level (label from area rule)

Confusion matrix (rows = ground truth, columns = prediction):

| GT \ Pred | normal | olp | other |
|---|---|---|---|
| normal | 13 | 14 | 3 |
| olp | 1 | 24 | 4 |
| other | 1 | 7 | 22 |

3-class accuracy: **66.3%**, macro-F1: **0.657**

| Class (one-vs-rest) | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| normal | 78.7% | 43.3% | 96.6% | 86.7% | 0.578 |
| olp | 70.8% | 82.8% | 65.0% | 53.3% | 0.649 |
| other | 83.1% | 73.3% | 88.1% | 75.9% | 0.746 |

## Pixel-level segmentation

| Class | n lesion images | Dice (mean ± SD) | IoU (mean ± SD) | Specificity on lesion-free images | Pooled Sens | Pooled Spec | Pooled Prec | Pooled F1 | Pooled Acc |
|---|---|---|---|---|---|---|---|---|---|
| olp | 29 | 0.564 ± 0.324 | 0.457 ± 0.299 | 96.1% | 57.4% | 96.4% | 62.7% | 0.599 | 92.8% |
| other | 30 | 0.492 ± 0.347 | 0.395 ± 0.314 | 99.2% | 64.9% | 98.0% | 61.8% | 0.633 | 96.4% |

Dice/IoU are averaged per image over the images whose ground truth contains the class. Pooled columns use all pixels of all images.
