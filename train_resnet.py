
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = Path("data/real data")

MODEL_PATH = Path("best_real_resnet18.pth")
SPLIT_PATH = Path("real_data_splits.json")
REPORT_PATH = Path("real_training_report.json")

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 5
MIN_IMAGES_PER_CLASS = 10
SEED = 42
NUM_WORKERS = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. CHECK DATASET PATH
# ==========================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_PATH.resolve()}"
    )


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. LOAD REAL DATASET
# ==========================================

# ImageFolder uses each class folder's name as its label.
# Empty folders are not included in dataset.classes.

full_dataset = datasets.ImageFolder(
    root=str(DATA_PATH),
    transform=transform,
    allow_empty=True
)

print("\nDataset inspection")
print("-" * 50)
print("Images discovered:", len(full_dataset))
print("Non-empty classes:", len(full_dataset.classes))

if len(full_dataset) == 0:
    raise ValueError("No images were found in the dataset.")

all_labels = np.array(full_dataset.targets)

class_counts = np.bincount(
    all_labels,
    minlength=len(full_dataset.classes)
)

print("\nImages per detected class:")

for class_name, count in zip(
    full_dataset.classes, class_counts
):
    print(f"{class_name}: {count}")


# ==========================================
# 5. SELECT CLASSES WITH ENOUGH IMAGES
# ==========================================

eligible_classes = [
    index
    for index, count in enumerate(class_counts)
    if count >= MIN_IMAGES_PER_CLASS
]

excluded_classes = [
    {
        "class_name": full_dataset.classes[index],
        "image_count": int(class_counts[index])
    }
    for index, count in enumerate(class_counts)
    if count < MIN_IMAGES_PER_CLASS
]

eligible_class_names = [
    full_dataset.classes[index]
    for index in eligible_classes
]

print("\nEligible classes:", len(eligible_classes))
print("Excluded classes:", len(excluded_classes))

if excluded_classes:
    print("\nClasses excluded from this experiment:")

    for item in excluded_classes:
        print(
            f"{item['class_name']}: "
            f"{item['image_count']} images"
        )

eligible_indices = np.where(
    np.isin(all_labels, eligible_classes)
)[0]

# Map the original labels to consecutive labels 0, 1, 2...
label_mapping = {
    original_label: new_label
    for new_label, original_label
    in enumerate(eligible_classes)
}

new_labels = np.array([
    label_mapping[all_labels[index]]
    for index in eligible_indices
])

class_names = eligible_class_names
num_classes = len(class_names)

if num_classes < 2:
    raise ValueError(
        "At least two eligible classes are required."
    )

print("\nImages selected for training/evaluation:",
      len(eligible_indices))
print("Number of output classes:", num_classes)


# ==========================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

train_indices, temporary_indices = train_test_split(
    eligible_indices,
    test_size=0.30,
    stratify=new_labels,
    random_state=SEED
)

# Labels corresponding to the temporary subset.
index_to_label = dict(zip(eligible_indices, new_labels))

temporary_labels = np.array([
    index_to_label[index]
    for index in temporary_indices
])

val_indices, test_indices = train_test_split(
    temporary_indices,
    test_size=0.50,
    stratify=temporary_labels,
    random_state=SEED
)

print("\nDataset split")
print("-" * 50)
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))
print("Testing images:", len(test_indices))


# Save split information so we can reproduce the experiment.
split_information = {
    "seed": SEED,
    "dataset_path": str(DATA_PATH),
    "class_names": class_names,
    "excluded_classes": excluded_classes,
    "train_indices": train_indices.tolist(),
    "validation_indices": val_indices.tolist(),
    "test_indices": test_indices.tolist()
}

with open(SPLIT_PATH, "w", encoding="utf-8") as file:
    json.dump(split_information, file, indent=4)

print("Saved split information:", SPLIT_PATH)


# ==========================================
# 7. CREATE DATASETS AND LOADERS
# ==========================================

# Subset indices refer to the original ImageFolder dataset.
# A wrapper remaps the original class labels to our selected classes.

class RemappedSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices, label_mapping):
        self.dataset = dataset
        self.indices = list(indices)
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image_index = self.indices[position]

        image, original_label = self.dataset[image_index]

        new_label = self.label_mapping[original_label]

        return image, new_label


train_dataset = RemappedSubset(
    full_dataset, train_indices, label_mapping
)

val_dataset = RemappedSubset(
    full_dataset, val_indices, label_mapping
)

test_dataset = RemappedSubset(
    full_dataset, test_indices, label_mapping
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ==========================================
# 8. CREATE PRETRAINED RESNET18
# ==========================================

print("\nLoading pretrained ResNet18...")

try:
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
except Exception as error:
    raise RuntimeError(
        "Could not load pretrained ResNet18 weights. "
        "Check your internet connection or cached weights."
    ) from error

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    num_classes
)

model = model.to(device)

print("ResNet18 created successfully!")
print("Output classes:", num_classes)


# ==========================================
# 9. LOSS AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==========================================
# 10. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

    return total_loss / total, correct / total


# ==========================================
# 11. VALIDATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels_batch in loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels_batch
            ).sum().item()

            total += labels_batch.size(0)

    return total_loss / total, correct / total


# ==========================================
# 12. TRAINING LOOP
# ==========================================

best_val_accuracy = -1.0
history = []

print("\nStarting real-data training...\n")

for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        model, train_loader
    )

    val_loss, val_accuracy = evaluate(
        model, val_loader
    )

    epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }

    history.append(epoch_results)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "num_classes": num_classes,
                "best_val_accuracy": best_val_accuracy,
                "epoch": epoch + 1,
                "excluded_classes": excluded_classes
            },
            MODEL_PATH
        )

        print("Best real-data model saved!")


# ==========================================
# 13. SAVE TRAINING REPORT
# ==========================================

report = {
    "device": str(device),
    "dataset_path": str(DATA_PATH),
    "total_images_found": len(full_dataset),
    "eligible_images": len(eligible_indices),
    "num_classes": num_classes,
    "class_names": class_names,
    "excluded_classes": excluded_classes,
    "train_images": len(train_indices),
    "validation_images": len(val_indices),
    "test_images": len(test_indices),
    "best_validation_accuracy": best_val_accuracy,
    "history": history,
    "model_path": str(MODEL_PATH)
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nTraining completed.")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print("Saved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)
print("Test set has been reserved for independent evaluation.")

