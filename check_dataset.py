
import os

# Change this path to your actual dataset folder
dataset_path = r"C:\Users\Phita\Desktop\plant_disease_project\dataset"

if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    print("Please check your dataset path.")

else:
    total_images = 0
    total_classes = 0

    print("\n========== DATASET REPORT ==========\n")

    for class_name in sorted(os.listdir(dataset_path)):

        class_path = os.path.join(dataset_path, class_name)

        if os.path.isdir(class_path):
            total_classes += 1

            image_count = 0

            for filename in os.listdir(class_path):
                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    image_count += 1

            print(f"Class: {class_name}")
            print(f"Images: {image_count}")
            print("-" * 40)

            total_images += image_count

    print("\n========== SUMMARY ==========")
    print("Total classes:", total_classes)
    print("Total images:", total_images)
