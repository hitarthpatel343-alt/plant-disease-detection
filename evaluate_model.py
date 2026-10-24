from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np

from base_model import PlantDiseaseCNN


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = Path("data/raw")
CHECKPOINT_PATH = Path("best_base_model.pth")

IMAGE_SIZE = 128
BATCH_SIZE = 32
MAX_IMAGES = 12000
SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. LOAD DATASET
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(
    root=str(DATA_PATH),
    transform=transform
)

print("Total images:", len(dataset))
print("Total classes:", len(dataset.classes))


# ==========================================
# 3. RECREATE THE SAME DATA SPLIT
# ==========================================

labels = np.array(dataset.targets)
all_indices = np.arange(len(dataset))

if len(dataset) > MAX_IMAGES:
    selected_indices, _ = train_test_split(
        all_indices,
        train_size=MAX_IMAGES,
        stratify=labels,
        random_state=SEED
    )
else:
    selected_indices = all_indices

selected_indices = np.array(selected_indices)
selected_labels = labels[selected_indices]

train_indices, remaining_indices = train_test_split(
    selected_indices,
    test_size=0.30,
    stratify=selected_labels,
    random_state=SEED
)

remaining_labels = labels[remaining_indices]

val_indices, test_indices = train_test_split(
    remaining_indices,
    test_size=0.50,
    stratify=remaining_labels,
    random_state=SEED
)

test_dataset = Subset(dataset, test_indices)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model = PlantDiseaseCNN(
    num_classes=len(dataset.classes)
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

criterion = nn.CrossEntropyLoss()

print("\nTrained model loaded successfully!")


# ==========================================
# 5. EVALUATE TEST DATA
# ==========================================

all_predictions = []
all_true_labels = []

total_loss = 0.0
total_correct = 0
total_samples = 0

with torch.no_grad():

    for images, labels_batch in test_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels_batch)

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        total_correct += (
            predictions == labels_batch
        ).sum().item()

        total_samples += labels_batch.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_true_labels.extend(
            labels_batch.cpu().numpy()
        )


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

test_loss = total_loss / total_samples
test_accuracy = total_correct / total_samples

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy Percentage: {test_accuracy * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        all_true_labels,
        all_predictions,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        zero_division=0
    )
)

print("\n========== CONFUSION MATRIX ==========")

matrix = confusion_matrix(
    all_true_labels,
    all_predictions,
    labels=list(range(len(dataset.classes)))
)

print(matrix)

print("\nEvaluation completed!")
