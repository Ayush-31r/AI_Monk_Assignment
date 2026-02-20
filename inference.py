import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

ATTR_NAMES = ["Attr1", "Attr2", "Attr3", "Attr4"]
NUM_CLASSES = 4

# same preprocessing as val set in train.py
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    # must match train.py exactly otherwise weights won't load
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.65),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def predict(image_path: str, model_path: str, threshold: float = 0.5) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))[0]

    present_attrs = [ATTR_NAMES[i] for i, p in enumerate(probs.cpu().tolist()) if p >= threshold]

    print(f"\nimage : {image_path}")
    print(f"threshold: {threshold}\n")
    for name, prob in zip(ATTR_NAMES, probs.cpu().tolist()):
        flag = "+" if prob >= threshold else "-"
        print(f"  [{flag}] {name}: {prob:.4f}")
    print("\nattributes present:", present_attrs if present_attrs else ["none"])

    return present_attrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--model",     default="model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predict(args.image, args.model, args.threshold)
