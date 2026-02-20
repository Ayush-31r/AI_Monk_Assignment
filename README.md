# Aimonk Multilabel Classification

## Files
| File | Purpose |
|------|---------|
| `train.py` | training pipeline → saves `model.pth` + `loss_curve.png` |
| `inference.py` | loads saved weights, predicts attributes for a single image |

---

## Setup

```bash
pip install torch torchvision matplotlib pandas pillow numpy
```

---

## Training

```bash
python train.py
```

Expects:
```
project/
├── data/
│   ├── images/
│   └── labels.txt
├── train.py
└── inference.py
```

Outputs:
- `model.pth` — best model weights (lowest validation loss)
- `loss_curve.png` — iteration vs training loss

---

## Inference

```bash
python inference.py --image data/images/image_0.jpg --model model.pth --threshold 0.5
```

---

## Design Decisions

### Architecture: ResNet-50
- pretrained on ImageNet — fine-tuning only, not training from scratch
- early layers (layer1, layer2) are frozen since they capture generic features like edges and textures that transfer well
- only layer3, layer4, and the new head are trained — gives the model enough capacity to learn without overfitting on a small dataset
- head is `Dropout(0.65) → Linear(2048, 256) → ReLU → Dropout(0.4) → Linear(256, 4)`
- three learning rate groups: layer3 × 0.01, layer4 × 0.1, head × 1.0 — earlier layers get smaller updates since they're already well trained

### Handling NA Labels — Masked Loss
- images with NA for a given attribute are not discarded
- a binary mask tensor travels with each sample (1 = known label, 0 = NA)
- `MaskedBCELoss` zeroes out NA positions before averaging, so they contribute no gradient
- the model still learns from whatever labels it does have for that image

### Class Imbalance — Positive Weighting
- `pos_weight[i] = #negatives / #positives` per attribute, clamped to [0.1, 20]
- passed to `BCEWithLogitsLoss` so the model is penalised more for missing rare positives

### Augmentations
- RandomResizedCrop (scale 0.5–1.0)
- RandomHorizontalFlip, RandomVerticalFlip
- ColorJitter (brightness, contrast, saturation, hue)
- RandomRotation(25°)
- RandomGrayscale
- RandomErasing — randomly blacks out patches to prevent memorisation
- ImageNet normalisation

### Regularisation
- dropout at 0.65 and 0.4 in the head
- weight decay 1e-2 via AdamW
- early stopping with patience of 10 epochs

### Optimiser & Scheduler
- AdamW with weight decay 1e-2
- cosine annealing LR over epochs

---

## What Was Tried

| Experiment | Val Loss | Outcome |
|---|---|---|
| full resnet, no freezing | 0.622 | overfitting |
| freeze all except layer4 | 0.588 | underfitting |
| freeze all except layer3+layer4 | 0.5756 | best result |
| EfficientNet-B0 | 0.5846 | worse than resnet |

---

## Further Improvements (not implemented)
- more training data — biggest potential gain by far
- k-fold cross validation — more reliable evaluation on small datasets
- per-attribute threshold tuning — optimise f1 per attribute instead of using 0.5 globally
- mixup / cutmix — blend two images and labels during training
- test-time augmentation — average predictions over multiple crops and flips
