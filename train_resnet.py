from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "real data"
MODEL_PATH = BASE_DIR / "best_real_resnet18.pth"
REPORT_PATH = BASE_DIR / "real_training_report.json"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
NUM_WORKERS = 0
SEED = 42

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 2. CHECK DATASET
# =========================================================

print("=" * 65)
print("PLANT DISEASE DETECTION - FULL DATASET TRAINING")
print("=" * 65)

print("Device:", device)
print("Dataset:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_PATH}"
    )


# =========================================================
# 3. IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 4. LOAD ALL NON-EMPTY CLASSES
# =========================================================

print("\nLoading dataset...")

original_dataset = datasets.ImageFolder(
    root=str(DATA_PATH),
    transform=transform,
    allow_empty=True
)

# Count images in every class.
class_counts = {
    class_name: 0
    for class_name in original_dataset.classes
}

for original_label in original_dataset.targets:
    class_name = original_dataset.classes[original_label]
    class_counts[class_name] += 1

# Keep every class that contains at least one image.
class_names = [
    class_name
    for class_name in original_dataset.classes
    if class_counts[class_name] > 0
]

# Map original ImageFolder labels to consecutive model labels.
original_to_model_label = {
    original_dataset.class_to_idx[class_name]: model_label
    for model_label, class_name in enumerate(class_names)
}

# Keep every image belonging to a non-empty class.
eligible_indices = [
    index
    for index, original_label
    in enumerate(original_dataset.targets)
    if original_label in original_to_model_label
]


class FullTrainingDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        label_mapping
    ):
        self.dataset = dataset
        self.indices = indices
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]

        image, original_label = self.dataset[original_index]

        model_label = self.label_mapping[original_label]

        return image, model_label


train_dataset = FullTrainingDataset(
    original_dataset,
    eligible_indices,
    original_to_model_label
)

if len(train_dataset) == 0:
    raise ValueError(
        "No images were found in the dataset."
    )

num_classes = len(class_names)

print("\nDATASET SUMMARY")
print("-" * 65)

print("Total class folders:", len(original_dataset.classes))
print("Classes with images:", num_classes)
print(
    "Empty classes skipped:",
    len(original_dataset.classes) - num_classes
)
print("Total training images:", len(train_dataset))

print("\nCLASS COUNTS")
print("-" * 65)

for class_name in class_names:
    print(
        f"{class_name}: {class_counts[class_name]} images"
    )

print("\nClasses included in training:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# =========================================================
# 5. CREATE DATALOADER
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


# =========================================================
# 6. CREATE RESNET18 MODEL
# =========================================================

print("\nLoading pretrained ResNet18...")

try:
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

except Exception as error:
    raise RuntimeError(
        "Could not load pretrained ResNet18 weights. "
        "Check your internet connection or torchvision "
        "weight cache, then try again."
    ) from error

# Replace the final layer with one output per dataset class.
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Model created successfully.")
print("Number of output classes:", num_classes)


# =========================================================
# 7. TRAINING FUNCTION
# =========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for batch_number, (images, labels) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Calculate loss.
        loss = criterion(outputs, labels)

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Calculate training statistics.
        predictions = torch.argmax(outputs, dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

        total_images += batch_size

        if batch_number % 50 == 0:
            print(
                f"Processed {total_images}/"
                f"{len(train_dataset)} images"
            )

    average_loss = total_loss / total_images

    training_accuracy = total_correct / total_images

    return average_loss, training_accuracy


# =========================================================
# 8. TRAIN ON ALL AVAILABLE IMAGES
# =========================================================

history = []

print("\nStarting training...")
print("Epochs:", EPOCHS)
print("Training images:", len(train_dataset))
print("No train/validation/test split is being used.")

start_time = time.time()

for epoch in range(EPOCHS):

    epoch_start = time.time()

    average_loss, training_accuracy = train_one_epoch()

    epoch_seconds = time.time() - epoch_start

    epoch_result = {
        "epoch": epoch + 1,
        "training_loss": average_loss,
        "training_accuracy": training_accuracy,
        "duration_seconds": round(epoch_seconds, 2)
    }

    history.append(epoch_result)

    print("\n" + "-" * 65)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {average_loss:.4f}")
    print(f"Training accuracy: {training_accuracy * 100:.2f}%")
    print(f"Epoch duration: {epoch_seconds:.2f} seconds")


# =========================================================
# 9. SAVE FINAL MODEL
# =========================================================

print("\nSaving final trained model...")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "num_classes": num_classes,
    "image_size": IMAGE_SIZE,
    "epochs": EPOCHS,
    "training_images": len(train_dataset),
    "training_accuracy": history[-1]["training_accuracy"],
    "training_loss": history[-1]["training_loss"],
    "class_counts": {
        class_name: class_counts[class_name]
        for class_name in class_names
    },
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225]
}

torch.save(checkpoint, MODEL_PATH)

print("Model saved:", MODEL_PATH)


# =========================================================
# 10. SAVE TRAINING REPORT
# =========================================================

total_seconds = time.time() - start_time

report = {
    "project": "Plant Disease Detection",
    "model_architecture": "ResNet18",
    "dataset_path": str(DATA_PATH),
    "device": str(device),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "total_training_images": len(train_dataset),
    "total_class_folders": len(original_dataset.classes),
    "trained_classes": num_classes,
    "empty_classes_skipped": (
        len(original_dataset.classes) - num_classes
    ),
    "class_names": class_names,
    "class_counts": {
        class_name: class_counts[class_name]
        for class_name in class_names
    },
    "final_training_accuracy": history[-1]["training_accuracy"],
    "final_training_loss": history[-1]["training_loss"],
    "history": history,
    "total_training_seconds": round(total_seconds, 2),
    "independent_test_evaluation": False,
    "note": (
        "All images from non-empty classes were used for training. "
        "Training accuracy is not independent test accuracy. "
        "An independent labelled dataset is required to measure "
        "generalization performance."
    ),
    "model_path": str(MODEL_PATH)
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )

print("Training report saved:", REPORT_PATH)


# =========================================================
# 11. COMPLETED
# =========================================================

print("\n" + "=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Training images:", len(train_dataset))
print("Trained classes:", num_classes)

print(
    f"Final training accuracy: "
    f"{history[-1]['training_accuracy'] * 100:.2f}%"
)

print("Model:", MODEL_PATH)
print("Report:", REPORT_PATH)
print(f"Total training time: {total_seconds / 60:.2f} minutes")