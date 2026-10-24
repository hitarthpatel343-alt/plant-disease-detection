
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from base_model import PlantDiseaseCNN


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = Path("data/raw")
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 3

# Small experiment first; increase after verification.
MAX_IMAGES = 12000

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_PATH.resolve()}"
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. LOAD THE DATASET
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(
    root=str(DATA_PATH),
    transform=transform
)

num_classes = len(dataset.classes)

print("\nTotal images:", len(dataset))
print("Total classes:", num_classes)

if num_classes != 38:
    raise ValueError(
        f"Expected 38 classes, but found {num_classes}."
    )

print("\nClass mapping:")

for index, class_name in enumerate(dataset.classes):
    print(f"{index}: {class_name}")


# ==========================================
# 3. GET LABELS
# ==========================================

labels = np.array(dataset.targets)
all_indices = np.arange(len(dataset))

# Use all images if the dataset is smaller than MAX_IMAGES.
# Otherwise, select a stratified subset.
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

print("\nImages selected for this experiment:",
      len(selected_indices))


# ==========================================
# 4. SPLIT INTO TRAIN, VALIDATION, TEST
# ==========================================

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

print("\nTraining images:", len(train_indices))
print("Validation images:", len(val_indices))
print("Testing images:", len(test_indices))


# ==========================================
# 5. CREATE DATA LOADERS
# ==========================================

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nData loaders created successfully.")


# ==========================================
# 6. CREATE THE MODEL
# ==========================================

CHECKPOINT_PATH = Path("best_base_model.pth")

model = PlantDiseaseCNN(
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

start_epoch = 0
best_val_accuracy = -1.0

if CHECKPOINT_PATH.exists():

    print("\nLoading saved model checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_val_accuracy = checkpoint.get(
        "best_val_accuracy", -1.0
    )

    print("Saved model weights loaded successfully!")

else:
    print("\nNo saved checkpoint found. Starting from scratch.")

print("Model is ready!")

# ==========================================
# 7. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Make predictions.
        outputs = model(images)

        # Calculate prediction error.
        loss = criterion(outputs, labels_batch)

        # Calculate gradients.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ==========================================
# 8. VALIDATION FUNCTION
# ==========================================

def evaluate(model, loader, criterion, device):

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

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ==========================================
# 9. TRAIN THE MODEL
# ==========================================

best_val_accuracy = -1.0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("-" * 40)

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": dataset.classes,
        "image_size": IMAGE_SIZE,
        "num_classes": num_classes,
        "best_val_accuracy": best_val_accuracy,
        "epoch": epoch + 1,
    },
    "best_base_model.pth"
)

print("Best model saved.")

print("\nTraining completed.")
print("Best validation accuracy:",
      f"{best_val_accuracy:.4f}")

print("\nSaved model: best_base_model.pth")
