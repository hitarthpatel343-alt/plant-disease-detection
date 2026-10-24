from pathlib import Path
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"  # Put the PlantVillage class folders here.
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # Safe default for Windows. Increase only if you know your setup supports it.

# These are the 17 PlantVillage classes for apple, potato, and tomato.
# Class matching accepts the common original folder names with triple underscores.
TARGET_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_folder_name(name):
    # Normalize small spelling/spacing differences while retaining crop and disease.
    return "".join(ch.lower() for ch in name if ch.isalnum())

def select_class_folders(imagefolder):
    """Match desired class names to folders in the downloaded dataset."""
    folder_by_normalized = {
        normalize_folder_name(name): name for name in imagefolder.classes
    }
    selected = []
    missing = []
    for wanted in TARGET_CLASSES:
        key = normalize_folder_name(wanted)
        found = folder_by_normalized.get(key)
        if found is None:
            missing.append(wanted)
        else:
            selected.append(found)
    if missing:
        print("\nCould not find these expected class folders:")
        for name in missing:
            print("  -", name)
        print("\nFolders detected in your dataset (first 50):")
        for name in imagefolder.classes[:50]:
            print("  -", name)
        raise RuntimeError(
            "Dataset folder names do not match the expected PlantVillage names. "
            "Check that data/raw contains the class folders, not an extra ZIP or parent folder."
        )
    return selected

class LeafDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, remapped integer label)
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label

def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    # ImageFolder scans class subfolders and assigns a numeric ID to each folder.
    scanned = ImageFolder(DATA_DIR)
    selected_folder_names = select_class_folders(scanned)
    selected_set = set(selected_folder_names)

    # Create a stable, compact label mapping for the 17 selected classes only.
    class_names = selected_folder_names
    label_map = {name: i for i, name in enumerate(class_names)}
    folder_to_original_index = scanned.class_to_idx

    all_samples = []
    for path, original_label in scanned.samples:
        folder_name = scanned.classes[original_label]
        if folder_name in selected_set:
            all_samples.append((path, label_map[folder_name]))

    if not all_samples:
        raise RuntimeError("No selected images found. Check the dataset directory.")

    labels = np.array([label for _, label in all_samples])
    indices = np.arange(len(all_samples))
    train_idx, temporary_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=labels
    )
    temp_labels = labels[temporary_idx]
    val_idx, test_idx = train_test_split(
        temporary_idx, test_size=0.50, random_state=SEED, stratify=temp_labels
    )

    # ImageNet pretrained MobileNetV2 expects normalized RGB images.
    weights = MobileNet_V2_Weights.DEFAULT
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Separate dataset objects let training use augmentation and evaluation stay deterministic.
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    train_loader = DataLoader(
        LeafDataset(train_samples, train_transform), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        LeafDataset(val_samples, eval_transform), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        LeafDataset(test_samples, eval_transform), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Selected classes: {len(class_names)}")
    print(f"Images total: {len(all_samples)} | train: {len(train_samples)} | "
          f"validation: {len(val_samples)} | test: {len(test_samples)}")
    print("\nClass counts:")
    for label, class_name in enumerate(class_names):
        count = int((labels == label).sum())
        print(f"  {class_name}: {count}")

    # Load ImageNet-learned visual features, freeze them, and replace the 1000-class head.
    model = mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_path = MODEL_DIR / "mobilenetv2_plant_disease.pth"
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss_sum += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total += targets.size(0)
        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, train acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, val acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": len(class_names),
                "architecture": "mobilenet_v2",
            }, best_path)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed / 60:.1f} minutes.")
    print(f"Best checkpoint saved to: {best_path}")

    # Reload best validation checkpoint before touching the held-out test set.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            predictions = model(images).argmax(1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(targets.numpy().tolist())

    metrics = {
        "dataset": "PlantVillage",
        "architecture": "MobileNetV2 transfer learning; frozen feature extractor",
        "num_classes": len(class_names),
        "total_images": len(all_samples),
        "train_images": len(train_samples),
        "validation_images": len(val_samples),
        "test_images": len(test_samples),
        "split": "70% train / 15% validation / 15% test, stratified, seed 42",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "training_minutes": round(elapsed / 60, 2),
        "device": str(device),
        "class_names": class_names,
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(16, 14))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(ax=ax, xticks_rotation=90, colorbar=False, values_format="d")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\n=== HELD-OUT TEST RESULTS (actual run) ===")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Macro precision:  {metrics['precision_macro']:.4f}")
    print(f"Macro recall:     {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score:   {metrics['f1_macro']:.4f}")
    print("\nPer-class report saved to reports/classification_report.txt")
    print("Confusion matrix saved to reports/confusion_matrix.png")
    print("Metrics saved to reports/metrics.json")
    print("\nIMPORTANT: These are PlantVillage test-set metrics, not proof of real-world field accuracy.")

if __name__ == "__main__":
    main()
