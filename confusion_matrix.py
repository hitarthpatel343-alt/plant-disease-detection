import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/raw"
MODEL_PATH = "best_resnet18.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 16
MAX_IMAGES = 12000
SEED = 42

device = torch.device("cpu")


# ==========================================
# 2. TRANSFORMATIONS
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
# 3. LOAD DATASET
# ==========================================

dataset = datasets.ImageFolder(
    root=DATA_PATH,
    transform=transform
)

print("Total images:", len(dataset))
print("Number of classes:", len(dataset.classes))


# ==========================================
# 4. RECREATE THE DATA SPLIT
# ==========================================

all_indices = np.arange(len(dataset))
labels = np.array(dataset.targets)

selected_indices, _ = train_test_split(
    all_indices,
    train_size=MAX_IMAGES,
    stratify=labels,
    random_state=SEED
)

selected_labels = labels[selected_indices]

train_indices, temp_indices = train_test_split(
    selected_indices,
    test_size=0.30,
    stratify=selected_labels,
    random_state=SEED
)

temp_labels = labels[temp_indices]

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)


# ==========================================
# 5. CREATE TEST DATALOADER
# ==========================================

test_dataset = Subset(dataset, test_indices)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 6. LOAD MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

if checkpoint["class_names"] != dataset.classes:
    raise ValueError("Class names do not match!")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(dataset.classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ==========================================
# 7. GET PREDICTIONS
# ==========================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels_batch in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels_batch.numpy()
        )


# ==========================================
# 8. CREATE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(dataset.classes)))
)

print("\nConfusion matrix shape:", cm.shape)


# ==========================================
# 9. DISPLAY MOST COMMON CONFUSIONS
# ==========================================

print("\nTop 15 class confusions:")

confusions = []

for actual_class in range(len(dataset.classes)):

    for predicted_class in range(len(dataset.classes)):

        if actual_class != predicted_class:

            count = cm[
                actual_class,
                predicted_class
            ]

            if count > 0:

                confusions.append(
                    (
                        count,
                        dataset.classes[actual_class],
                        dataset.classes[predicted_class]
                    )
                )

confusions.sort(reverse=True)

for count, actual, predicted in confusions[:15]:

    print(
        f"{actual} --> {predicted}: {count} images"
    )


# ==========================================
# 10. PLOT CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(22, 18))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=dataset.classes,
    yticklabels=dataset.classes
)

plt.title("ResNet18 Plant Disease Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)

plt.tight_layout()

plt.savefig(
    "resnet_confusion_matrix.png",
    dpi=200
)

plt.show()

print("\nSaved: resnet_confusion_matrix.png")