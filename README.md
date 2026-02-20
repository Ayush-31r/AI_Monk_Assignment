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

## Thought Process

### Why Multilabel and not Multiclass?
each image can have more than one attribute present at the same time. multiclass classification picks one winner, which doesn't work here. multilabel means we treat each attribute as its own independent yes/no question, which is what the problem actually needs.

---

### Algorithm: Binary Cross Entropy per Attribute
since each attribute is independent, we run a sigmoid on each of the 4 outputs and compute binary cross entropy per attribute. this is the standard approach for multilabel problems. we use `BCEWithLogitsLoss` which combines the sigmoid and loss into one numerically stable step.

---

### Model: ResNet-50
we needed a pretrained model that already understands images so we don't have to learn from scratch with limited data. ResNet-50 was chosen because:
- it's pretrained on ImageNet (1 million images), so it already knows how to recognise shapes, textures, edges
- it's well tested and reliable
- it's not too large that it overfits easily on small datasets

we replaced the final fully connected layer with our own head that outputs 4 values (one per attribute).

---

### Why Fine-Tune and not Train from Scratch?
training from scratch requires a huge amount of data. we don't have that. fine-tuning lets us start from a model that already understands images and just teach it our specific task on top of that. this is standard practice when working with small datasets.

---

### Freezing Layers — Finding the Right Balance
this was the trickiest part and required experimentation:

- **freeze nothing** → the whole network updated on a small dataset → memorised training images → overfitting (val loss went up to 0.622)
- **freeze everything except layer4** → too restrictive, model couldn't learn enough → underfitting (val loss stuck at 0.588, stopped improving at epoch 1)
- **freeze everything except layer3 + layer4** → sweet spot, model had enough capacity to learn without memorising → best result (val loss 0.5756)

the intuition is that early layers (layer1, layer2) learn generic things like edges and textures that apply to any image, so we keep those frozen. later layers learn more task-specific things, so we unfreeze those and let them adapt.

---

### Handling NA Labels
some images had NA for certain attributes meaning no one checked whether that attribute was present or not. two options:
1. skip the whole image — wastes data
2. use a mask — keep the image but ignore the NA attributes when computing loss

we went with option 2. a binary mask travels with each image (1 = label known, 0 = NA). the loss is only computed and averaged over known labels. this way every image contributes something to training.

---

### Class Imbalance
some attributes appear in most images, others appear in very few. if we do nothing, the model learns to always predict "no" for rare attributes and still gets a decent loss score — but it's useless. 

the fix is `pos_weight`: for each attribute we compute `#negatives / #positives` and pass it to the loss function. this means getting a rare positive wrong is penalised much more heavily than getting a common negative wrong, forcing the model to pay attention to rare cases.

---

### Overfitting Fixes Applied
after seeing val loss climb while train loss dropped, we applied four fixes together:
- **layer freezing** — biggest impact, explained above
- **dropout** (0.65 + 0.4 in the head) — randomly switches off neurons during training so the model can't rely on memorised patterns
- **weight decay 1e-2** — penalises large weights, keeps the model simple
- **aggressive augmentation** — each image looks slightly different every epoch so the model can't memorise exact pixels

---

### Early Stopping
we added patience-based early stopping (patience = 5 epochs). if val loss doesn't improve for 10 epochs in a row, training stops automatically and the best model is kept. this prevents wasting time and avoids the model degrading after its best point.

---

### Augmentations
augmentation artificially increases dataset variety by applying random transformations to training images. the model sees a different version of each image every epoch:
- random crop and resize — forces the model to recognise objects at different scales
- horizontal and vertical flips — object identity shouldn't depend on orientation
- colour jitter — lighting and colour shouldn't matter
- random rotation — orientation invariance
- random erasing — forces model to not rely on any single part of the image

---

## What Was Tried

| experiment | val loss | outcome |
|---|---|---|
| full resnet, no freezing | 0.622 | overfitting |
| freeze all except layer4 | 0.588 | underfitting |
| freeze all except layer3 + layer4 | **0.5756** | best result ✓ |
| EfficientNet-B0 | 0.5846 | worse than resnet |

---

## Further Improvements (not implemented)
- more training data — the single biggest thing that would improve results
- k-fold cross validation — more reliable evaluation on small datasets
- per-attribute threshold tuning — find the best threshold per attribute using f1 score instead of using 0.5 globally
- mixup / cutmix — blend two images and their labels during training, very effective for multilabel
- test-time augmentation — run inference on multiple crops/flips and average the predictions
