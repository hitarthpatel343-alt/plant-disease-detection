from pathlib import Path

# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

# Automatically find the folder containing this Python file.
BASE_DIR = Path(__file__).resolve().parent

# Dataset location:
# plant_disease_detection/
# ├── check_dataset.py
# └── data/
#     └── real data/
#         ├── Apple___Apple_scab/
#         ├── Apple___healthy/
#         └── ...

DATASET_PATH = BASE_DIR / "data" / "real data"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# Minimum recommended number of images per class for reporting
MIN_IMAGES_PER_CLASS = 10


# ============================================================
# 2. CHECK WHETHER THE DATASET EXISTS
# ============================================================

print("\n" + "=" * 60)
print("          PLANT DISEASE DATASET CHECKER")
print("=" * 60)

print(f"\nProject folder : {BASE_DIR}")
print(f"Dataset folder : {DATASET_PATH}")

if not DATASET_PATH.exists():
    print("\nERROR: Dataset folder was not found!")
    print("\nPlease check that your project structure is:")
    print("plant_disease_detection/")
    print("    check_dataset.py")
    print("    data/")
    print("        real data/")
    raise SystemExit(1)

if not DATASET_PATH.is_dir():
    print("\nERROR: The dataset path is not a directory.")
    raise SystemExit(1)


# ============================================================
# 3. FIND CLASS FOLDERS
# ============================================================

class_folders = sorted(
    [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

if not class_folders:
    print("\nERROR: No class folders were found in the dataset.")
    raise SystemExit(1)


# ============================================================
# 4. COUNT IMAGES IN EACH CLASS
# ============================================================

total_classes = 0
total_images = 0

empty_classes = []
small_classes = []
healthy_classes = []
class_results = []

print("\n" + "=" * 60)
print("                 CLASS-WISE REPORT")
print("=" * 60)

for class_folder in class_folders:

    # Count supported image files, including images in subfolders.
    image_files = [
        file
        for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_count = len(image_files)

    total_classes += 1
    total_images += image_count

    class_results.append(
        {
            "class_name": class_folder.name,
            "image_count": image_count,
        }
    )

    print(f"\nClass name : {class_folder.name}")
    print(f"Image count: {image_count}")

    if image_count == 0:
        empty_classes.append(class_folder.name)
        print("Status     : EMPTY")

    elif image_count < MIN_IMAGES_PER_CLASS:
        small_classes.append(
            (class_folder.name, image_count)
        )
        print("Status     : FEWER THAN 10 IMAGES")

    else:
        healthy_classes.append(class_folder.name)
        print("Status     : OK")


# ============================================================
# 5. DISPLAY DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("                  DATASET SUMMARY")
print("=" * 60)

print(f"Total class folders          : {total_classes}")
print(f"Total image files            : {total_images}")
print(f"Classes with images          : {len(healthy_classes) + len(small_classes)}")
print(f"Empty classes                : {len(empty_classes)}")
print(f"Classes with 1-9 images      : {len(small_classes)}")
print(f"Classes with 10+ images      : {len(healthy_classes)}")


# ============================================================
# 6. DISPLAY EMPTY CLASSES
# ============================================================

print("\n" + "=" * 60)
print("                    EMPTY CLASSES")
print("=" * 60)

if empty_classes:
    for class_name in empty_classes:
        print(f"- {class_name}")
else:
    print("No empty classes found.")


# ============================================================
# 7. DISPLAY CLASSES WITH FEWER THAN 10 IMAGES
# ============================================================

print("\n" + "=" * 60)
print("             CLASSES WITH FEWER THAN 10 IMAGES")
print("=" * 60)

if small_classes:
    for class_name, count in small_classes:
        print(f"- {class_name}: {count} images")
else:
    print("No classes with fewer than 10 images found.")


# ============================================================
# 8. DATASET VALIDATION RESULT
# ============================================================

print("\n" + "=" * 60)
print("                 VALIDATION RESULT")
print("=" * 60)

if total_images == 0:
    print("ERROR: No supported image files were found.")
    print("Please check the dataset folders and image formats.")

elif empty_classes:
    print("WARNING: Some class folders contain no images.")
    print("These classes cannot be trained as normal image classes.")

elif small_classes:
    print("WARNING: Some classes have fewer than 10 images.")
    print("Review these classes before training.")

else:
    print("SUCCESS: All class folders contain at least 10 images.")

print("\nDataset checking completed.")
print("=" * 60)

