import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


CONFIG = {
    "labels_file":   "data/labels.txt",
    "images_dir":    "data/images",
    "model_save":    "model.pth",
    "num_classes":   4,
    "batch_size":    16,
    "num_epochs":    40,
    "learning_rate": 5e-5,
    "weight_decay":  1e-2,
    "img_size":      224,
    "num_workers":   4,
    "seed":          42,
    "patience":      5,
}

ATTR_NAMES = ["Attr1", "Attr2", "Attr3", "Attr4"]


def parse_labels(labels_file: str) -> pd.DataFrame:
    # each line is: image_name.jpg attr1 attr2 attr3 attr4
    # na means the label is unknown for that attribute
    records = []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"  skipping malformed line: {line!r}")
                continue
            img_name = parts[0]
            attrs = []
            for val in parts[1:]:
                if val.upper() == "NA":
                    attrs.append(np.nan)
                else:
                    # strip any trailing non-numeric chars e.g. "0mobilenet" -> 0
                    clean = "".join(ch for ch in val if ch.isdigit() or ch == ".")
                    attrs.append(float(clean) if clean else np.nan)
            records.append([img_name] + attrs)

    return pd.DataFrame(records, columns=["image_name"] + ATTR_NAMES)


class MultilabelDataset(Dataset):

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
        transforms.RandomRotation(25),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, df: pd.DataFrame, images_dir: str, train: bool = True):
        # skip images that are missing from disk
        exists = df["image_name"].apply(
            lambda name: os.path.isfile(os.path.join(images_dir, name))
        )
        missing = df[~exists]["image_name"].tolist()
        if missing:
            print(f"  skipping {len(missing)} missing image(s): {missing}")
        self.df = df[exists].reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = self.TRAIN_TRANSFORMS if train else self.VAL_TRANSFORMS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.images_dir, row["image_name"])).convert("RGB")
        image = self.transform(image)

        # na labels are set to 0 but masked out so they don't affect the loss
        labels = torch.tensor(
            [row[a] if not pd.isna(row[a]) else 0.0 for a in ATTR_NAMES],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [0.0 if pd.isna(row[a]) else 1.0 for a in ATTR_NAMES],
            dtype=torch.float32,
        )
        return image, labels, mask


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # freeze the whole network first
    for param in model.parameters():
        param.requires_grad = False

    # only fine-tune the last two residual blocks
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.65),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def compute_pos_weights(df: pd.DataFrame) -> torch.Tensor:
    # weight = neg/pos per attribute to handle class imbalance
    weights = []
    for attr in ATTR_NAMES:
        col = df[attr].dropna()
        pos = (col == 1).sum()
        neg = (col == 0).sum()
        weights.append(float(np.clip(neg / max(pos, 1), 0.1, 20.0)))
    return torch.tensor(weights, dtype=torch.float32)


class MaskedBCELoss(nn.Module):

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits, targets, mask):
        loss = self.bce(logits, targets)
        loss = loss * mask  # ignore na positions
        return loss.sum() / mask.sum().clamp(min=1e-6)


def train(config: dict):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    df = parse_labels(config["labels_file"])
    print(f"total samples: {len(df)}")

    df = df.sample(frac=1, random_state=config["seed"]).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    train_loader = DataLoader(
        MultilabelDataset(train_df, config["images_dir"], train=True),
        batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        MultilabelDataset(val_df, config["images_dir"], train=False),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )

    model = build_model(config["num_classes"]).to(device)

    # different learning rates for each group — earlier layers get smaller updates
    optimizer = optim.AdamW([
        {"params": list(model.layer3.parameters()), "lr": config["learning_rate"] * 0.01},
        {"params": list(model.layer4.parameters()), "lr": config["learning_rate"] * 0.1},
        {"params": list(model.fc.parameters()),     "lr": config["learning_rate"]},
    ], weight_decay=config["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    criterion = MaskedBCELoss(compute_pos_weights(df).to(device))

    iteration_losses = []
    global_iter = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = config.get("patience", 10)

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for images, labels, mask in train_loader:
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels, mask)
            loss.backward()
            optimizer.step()
            global_iter += 1
            iteration_losses.append((global_iter, loss.item()))
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, mask in val_loader:
                images, labels, mask = images.to(device), labels.to(device), mask.to(device)
                val_loss += criterion(model(images), labels, mask).item()
        avg_val = val_loss / max(len(val_loader), 1)

        print(f"Epoch [{epoch:3d}/{config['num_epochs']}]  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), config["model_save"])
            print(f"  -> saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  no improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"  early stopping at epoch {epoch}")
                break

    iters, losses = zip(*iteration_losses)
    window = max(1, len(losses) // 50)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(iters, losses, linewidth=0.8, alpha=0.7, label="batch loss")
    plt.plot(range(window, len(losses) + 1), smooth, linewidth=2, color="red", label="smoothed")
    plt.xlabel("iteration_number")
    plt.ylabel("training_loss")
    plt.title("Aimonk_multilabel_problem")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("loss curve saved to loss_curve.png")


if __name__ == "__main__":
    train(CONFIG)
