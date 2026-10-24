
from pathlib import Path
from collections import Counter

# Path to your manager's real dataset
DATA_DIR = Path("data/real data")

# Image formats we will look for
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


def main():
    if not DATA_DIR.exists():
        print(f"Dataset folder not found: {DATA_DIR.resolve()}")
        print("Please check the folder path.")
        return

    class_folders = sorted(
        folder for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    print("=" * 60)
    print("REAL PLANT DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset location: {DATA_DIR.resolve()}")
    print(f"Number of classes: {len(class_folders)}")

    total_images = 0
    empty_folders = []
    image_counts = {}

    for folder in class_folders:
        images = [
            file for file in folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        count = len(images)
        image_counts[folder.name] = count
        total_images += count

        if count == 0:
            empty_folders.append(folder.name)

        print(f"{folder.name}: {count} images")

    print("\n" + "=" * 60)
    print(f"TOTAL CLASSES: {len(class_folders)}")
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 60)

    if empty_folders:
        print("\nFolders with no supported images:")
        for folder in empty_folders:
            print(folder)

    if image_counts:
        counts = list(image_counts.values())
        print(f"\nSmallest class: {min(counts)} images")
        print(f"Largest class: {max(counts)} images")


if __name__ == "__main__":
    main()

