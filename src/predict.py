from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mobilenetv2_plant_disease.pth"

def main():
    parser = argparse.ArgumentParser(description="Predict a plant leaf class from an image.")
    parser.add_argument("image_path", help="Path to a leaf image, e.g. sample_leaf.jpg")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    with Image.open(image_path) as im:
        image = im.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, predicted_idx = probabilities.max(dim=0)

    print(f"Image: {image_path}")
    print(f"Predicted class: {class_names[predicted_idx.item()]}")
    print(f"Model confidence: {confidence.item() * 100:.2f}%")
    print("Note: confidence is not the same as probability of being correct, especially on unfamiliar field images.")

if __name__ == "__main__":
    main()
