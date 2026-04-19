# distressCNN Model Architecture

## Overview

`distressCNN.keras` is a Keras Sequential CNN trained to classify road pavement distress from grayscale images into 13 categories with varying severity levels.

## Input

| Property | Value |
|---|---|
| Shape | `(1, 64, 64, 1)` |
| Color space | Grayscale |
| Pixel range | `[0.0, 1.0]` (normalized by dividing by 255) |

## Layer Stack

```
Input: (64, 64, 1)
│
├── Conv2D — 32 filters, 3×3 kernel, ReLU
├── MaxPooling2D — 2×2 pool
│
├── Conv2D — 64 filters, 3×3 kernel, ReLU
├── MaxPooling2D — 2×2 pool
│
├── Flatten
│
├── Dense — 128 units, ReLU
└── Dense — 13 units, Softmax  ← output probabilities
```

## Output Classes

The final Dense layer outputs a probability distribution over 13 classes:

| Index | Class | Severity |
|---|---|---|
| 0 | Longitudinal-traverse | Low |
| 1 | Longitudinal-traverse | Medium |
| 2 | Longitudinal-traverse | High |
| 3 | Patch | Low |
| 4 | Patch | Medium |
| 5 | Patch | High |
| 6 | Pothole | Low |
| 7 | Pothole | Medium |
| 8 | Pothole | High |
| 9 | Ravelling and weathering | Low |
| 10 | Ravelling and weathering | Medium |
| 11 | Ravelling and weathering | High |
| 12 | Rutting | — |

The predicted class is the index with the highest softmax probability (`np.argmax`).

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | `SparseCategoricalCrossentropy` |
| Metric | Accuracy |
| Epochs | 40 |
| Train/test split | 80% / 20% of full dataset (shuffled, `random_state=40`) |
| Validation split | 10% of the training set (= 8% of full dataset) |

## Dataset

The model was trained on the **Flexible Pavement Distresses** dataset, organized with one subdirectory per class. Images were augmented before training using `augment_dir` in [images.py](images.py), which applies:

- 90° rotation
- 180° rotation
- Horizontal flip
- Brightness increase (factor 1.5×)

## Preprocessing Pipeline

Every inference image must follow this exact pipeline (implemented in [backend.py](backend.py) as `preprocess_image`):

1. Resize to 64×64 pixels
2. Convert to grayscale (PIL `'L'` mode → numpy array)
3. Normalize: divide by `255.0`
4. Reshape to `(1, 64, 64, 1)`
