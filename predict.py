
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_real_resnet18.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {DEVICE}")

    return model, class_names


# --------------------------------------------------
# 4. Predict disease from an image
# --------------------------------------------------

def predict_disease(image_path, model, class_names):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities[0],
            k=min(3, len(class_names))
        )

    results = []

    for probability, index in zip(
        top_probabilities.tolist(),
        top_indices.tolist()
    ):
        results.append({
            "disease": class_names[index],
            "confidence": probability * 100
        })

    return results


# --------------------------------------------------
# 5. Run a test prediction
# --------------------------------------------------

if __name__ == "__main__":

    model, class_names = load_model()

    image_path = input(
        "\nEnter the full path of a plant-leaf image: "
    ).strip().strip('"')

    try:
        results = predict_disease(
            image_path,
            model,
            class_names
        )

        print("\n" + "=" * 45)
        print("PLANT DISEASE PREDICTION")
        print("=" * 45)

        for position, result in enumerate(results, start=1):
            print(f"\n{position}. {result['disease']}")
            print(
                f"   Model confidence: "
                f"{result['confidence']:.2f}%"
            )

        print("\nNote: Confidence is not a guarantee")
        print("that the prediction is correct.")

    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"\nError: {error}")

