
from pathlib import Path
from PIL import Image, UnidentifiedImageError

DATA_DIR = Path("data/real data")

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}


def main():
    if not DATA_DIR.exists():
        print(f"Dataset not found: {DATA_DIR.resolve()}")
        return

    total_images = 0
    valid_images = 0
    corrupted_images = []
    class_counts = {}

    class_folders = sorted(
        folder for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    print("Validating real plant images...")
    print("This may take a few minutes.\n")

    for class_folder in class_folders:
        count = 0

        for image_path in class_folder.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            total_images += 1
            count += 1

            try:
                with Image.open(image_path) as image:
                    image.verify()

                valid_images += 1

            except (OSError, ValueError, UnidentifiedImageError):
                corrupted_images.append(str(image_path))

        if count > 0:
            class_counts[class_folder.name] = count

    print("=" * 55)
    print("IMAGE VALIDATION RESULTS")
    print("=" * 55)

    print(f"Image files found: {total_images}")
    print(f"Valid images: {valid_images}")
    print(f"Invalid/corrupted images: {len(corrupted_images)}")
    print(f"Classes containing images: {len(class_counts)}")

    print("\nImages per class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    if corrupted_images:
        print("\nFirst 20 invalid image paths:")
        for image_path in corrupted_images[:20]:
            print(image_path)

    print("\nValidation complete.")
    print("No files have been deleted or modified.")


if __name__ == "__main__":
    main()

