
from pathlib import Path
import json
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "real data"
MODEL_PATH = BASE_DIR / "best_real_resnet18.pth"
SPLIT_PATH = BASE_DIR / "real_data_splits.json"

OUTPUT_DIR = BASE_DIR / "evaluation_results"

BATCH_SIZE = 16
NUM_WORKERS = 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("PLANT DISEASE DETECTION - MODEL EVALUATION")
print("=" * 65)

print("Device:", device)
print("Dataset:", DATA_PATH)
print("Model:", MODEL_PATH)


# =========================================================
# 2. CHECK REQUIRED FILES
# =========================================================

for path in [DATA_PATH, MODEL_PATH, SPLIT_PATH]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file or folder not found: {path}"
        )


# =========================================================
# 3. LOAD MODEL CHECKPOINT
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]

image_size = checkpoint["image_size"]

num_classes = checkpoint["num_classes"]

print("\nMODEL INFORMATION")
print("-" * 65)

print("Number of classes:", num_classes)
print("Image size:", image_size)

print(
    "Best validation accuracy:",
    f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
)

print("Best epoch:", checkpoint["epoch"])


# =========================================================
# 4. LOAD SAVED DATASET SPLIT
# =========================================================

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8"
) as file:

    split_info = json.load(file)


# Verify that the model and split use the same class order.

if class_names != split_info["class_names"]:

    raise ValueError(
        "Class names in the model checkpoint do not match "
        "the saved dataset split."
    )

if num_classes != len(class_names):

    raise ValueError(
        "Number of classes does not match the saved class names."
    )


# =========================================================
# 5. IMAGE PREPROCESSING
# =========================================================

# These transformations match train_resnet.py.

transform = transforms.Compose([

    transforms.Resize(
        (image_size, image_size)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# =========================================================
# 6. LOAD THE ORIGINAL DATASET
# =========================================================

# IMPORTANT:
# The saved training and test indices refer to this original
# ImageFolder dataset, which contains all discovered classes.
#
# Do not filter the dataset before reading the saved indices.

original_dataset = datasets.ImageFolder(
    root=str(DATA_PATH),
    transform=transform,
    allow_empty=True
)

print("\nDATASET INFORMATION")
print("-" * 65)

print("All discovered images:", len(original_dataset))
print("All discovered classes:", len(original_dataset.classes))


# =========================================================
# 7. VERIFY CLASS NAMES
# =========================================================

missing_classes = [

    name
    for name in class_names
    if name not in original_dataset.class_to_idx

]

if missing_classes:

    raise ValueError(
        "The following model classes were not found "
        f"in the current dataset: {missing_classes}"
    )


# =========================================================
# 8. MAP ORIGINAL LABELS TO MODEL LABELS
# =========================================================

# ImageFolder assigns labels using all 100 class folders.
#
# The model was trained using only 20 eligible classes.
#
# Therefore, we map the original labels to the model's
# expected labels, from 0 to 19.

original_to_model_label = {

    original_dataset.class_to_idx[class_name]: model_label

    for model_label, class_name in enumerate(class_names)

}

print("\nCLASS MAPPING")
print("-" * 65)

for model_label, class_name in enumerate(class_names):

    original_label = original_dataset.class_to_idx[class_name]

    print(
        f"Model label {model_label:2d} | "
        f"Original ImageFolder label {original_label:2d} | "
        f"{class_name}"
    )


# =========================================================
# 9. CREATE DATASET WITH ORIGINAL IMAGE INDICES
# =========================================================

# We keep the original image indices but return labels
# that match the model's 20 output classes.

class OriginalIndexDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        label_mapping
    ):

        self.dataset = dataset
        self.indices = list(indices)
        self.label_mapping = label_mapping

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, position):

        original_index = self.indices[position]

        image, original_label = self.dataset[original_index]

        model_label = self.label_mapping[original_label]

        return image, model_label


# =========================================================
# 10. SELECT ORIGINAL TEST INDICES
# =========================================================

saved_test_indices = split_info["test_indices"]

if not saved_test_indices:

    raise ValueError(
        "No test indices were found in real_data_splits.json."
    )

if (
    min(saved_test_indices) < 0
    or max(saved_test_indices) >= len(original_dataset)
):

    raise ValueError(
        "Saved test indices exceed the current dataset size. "
        "The dataset may have changed since training."
    )


# Retain only images from classes that were used for training.
#
# The indices remain indices into original_dataset.

eligible_test_indices = []

for index in saved_test_indices:

    original_label = original_dataset.targets[index]

    if original_label in original_to_model_label:

        eligible_test_indices.append(index)


if not eligible_test_indices:

    raise ValueError(
        "No eligible test images were found. "
        "Check whether the dataset has changed since training."
    )


# =========================================================
# 11. CHECK WHETHER SAVED INDICES ARE STILL VALID
# =========================================================

# The split file saves indices, not image paths.
# If files were added, deleted, or reordered after training,
# the indices might refer to different images.
#
# We cannot prove the original file ordering from indices alone.
# We report the selected test count and proceed only if the
# expected split size is consistent.

test_dataset = OriginalIndexDataset(
    original_dataset,
    eligible_test_indices,
    original_to_model_label
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("\nTEST DATASET")
print("-" * 65)

print("Saved test indices:", len(saved_test_indices))
print("Eligible test images:", len(test_dataset))


# =========================================================
# 12. REBUILD RESNET18
# =========================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("\nModel checkpoint loaded successfully.")


# =========================================================
# 13. RUN MODEL PREDICTIONS
# =========================================================

all_true = []
all_pred = []
all_probabilities = []

print("\nStarting evaluation...")

with torch.no_grad():

    for batch_number, (images, labels) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_true.extend(
            labels.numpy().tolist()
        )

        all_pred.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        if batch_number % 10 == 0:

            print(
                f"Processed {len(all_true)} "
                f"/ {len(test_dataset)} test images"
            )


# =========================================================
# 14. CALCULATE EVALUATION METRICS
# =========================================================

accuracy = accuracy_score(
    all_true,
    all_pred
)

precision_macro = precision_score(
    all_true,
    all_pred,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    all_true,
    all_pred,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    all_true,
    all_pred,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    all_true,
    all_pred,
    labels=list(range(num_classes)),
    average="weighted",
    zero_division=0
)

average_confidence = float(
    np.max(
        np.array(all_probabilities),
        axis=1
    ).mean()
)


# =========================================================
# 15. DISPLAY FINAL RESULTS
# =========================================================

print("\n")
print("=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

print("Test images:", len(all_true))

print(
    f"Accuracy:           {accuracy * 100:.2f}%"
)

print(
    f"Macro precision:    {precision_macro:.4f}"
)

print(
    f"Macro recall:       {recall_macro:.4f}"
)

print(
    f"Macro F1-score:     {f1_macro:.4f}"
)

print(
    f"Weighted F1-score:  {f1_weighted:.4f}"
)

print(
    f"Average confidence: {average_confidence * 100:.2f}%"
)


# =========================================================
# 16. CLASSIFICATION REPORT
# =========================================================

class_report_text = classification_report(
    all_true,
    all_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

class_report_dict = classification_report(
    all_true,
    all_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(class_report_text)


# =========================================================
# 17. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(num_classes))
)

plt.figure(figsize=(20, 16))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Plant Disease Detection - Test Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.yticks(
    rotation=0,
    fontsize=8
)

plt.tight_layout()

confusion_path = (
    OUTPUT_DIR / "test_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved:", confusion_path)


# =========================================================
# 18. SAVE JSON EVALUATION REPORT
# =========================================================

evaluation_report = {

    "model_path": str(MODEL_PATH),

    "dataset_path": str(DATA_PATH),

    "test_images": len(all_true),

    "num_classes": num_classes,

    "class_names": class_names,

    "image_size": image_size,

    "best_epoch": checkpoint["epoch"],

    "best_validation_accuracy":
        checkpoint["best_val_accuracy"],

    "test_accuracy": accuracy,

    "macro_precision": precision_macro,

    "macro_recall": recall_macro,

    "macro_f1": f1_macro,

    "weighted_f1": f1_weighted,

    "average_confidence": average_confidence,

    "classification_report": class_report_dict,

    "confusion_matrix": cm.tolist()

}

report_path = (
    OUTPUT_DIR / "test_evaluation.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        evaluation_report,
        file,
        indent=4
    )

print("Evaluation report saved:", report_path)


# =========================================================
# 19. SAVE INDIVIDUAL PREDICTIONS
# =========================================================

predictions_path = (
    OUTPUT_DIR / "test_predictions.csv"
)

with open(
    predictions_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([

        "dataset_index",

        "actual_class",

        "predicted_class",

        "correct",

        "confidence"

    ])

    for position, (
        true_label,
        pred_label,
        probabilities
    ) in enumerate(
        zip(
            all_true,
            all_pred,
            all_probabilities
        )
    ):

        dataset_index = eligible_test_indices[position]

        confidence = float(
            max(probabilities)
        )

        writer.writerow([

            dataset_index,

            class_names[true_label],

            class_names[pred_label],

            true_label == pred_label,

            round(confidence, 6)

        ])


print(
    "Individual predictions saved:",
    predictions_path
)


# =========================================================
# 20. COMPLETED
# =========================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)

print("Test images:", len(all_true))

print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)

print("Results folder:", OUTPUT_DIR)

