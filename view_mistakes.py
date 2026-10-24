import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split


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
# 2. IMAGE TRANSFORMATIONS
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

all_indices = list(range(len(dataset)))
labels = dataset.targets

selected_indices, _ = train_test_split(
    all_indices,
    train_size=MAX_IMAGES,
    stratify=labels,
    random_state=SEED
)

selected_labels = [labels[i] for i in selected_indices]

train_indices, temp_indices = train_test_split(
    selected_indices,
    test_size=0.30,
    stratify=selected_labels,
    random_state=SEED
)

temp_labels = [labels[i] for i in temp_indices]

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
# 6. LOAD TRAINED MODEL
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
# 7. FIND INCORRECT PREDICTIONS
# ==========================================

mistakes = []

with torch.no_grad():

    for images, actual_labels in test_loader:

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(images)):

            actual = actual_labels[i].item()
            predicted = predictions[i].item()

            if actual != predicted:

                mistakes.append({
                    "image": images[i].clone(),
                    "actual": actual,
                    "predicted": predicted
                })


print("Total incorrect predictions:", len(mistakes))


# ==========================================
# 8. DISPLAY FIRST 9 MISTAKES
# ==========================================

if len(mistakes) == 0:

    print("No incorrect predictions found!")

else:

    number_to_show = min(9, len(mistakes))

    fig, axes = plt.subplots(
        3, 3,
        figsize=(15, 15)
    )

    axes = axes.flatten()

    for i in range(number_to_show):

        item = mistakes[i]

        image = item["image"]

        # Reverse ImageNet normalization for display.
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = image * std + mean
        image = image.clamp(0, 1)

        image = image.permute(1, 2, 0).numpy()

        actual_name = dataset.classes[item["actual"]]
        predicted_name = dataset.classes[item["predicted"]]

        axes[i].imshow(image)

        axes[i].set_title(
            f"Actual: {actual_name}\n"
            f"Predicted: {predicted_name}",
            fontsize=8
        )

        axes[i].axis("off")

    for i in range(number_to_show, 9):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_images.png", dpi=200)
    plt.show()

    print("Saved: misclassified_images.png")