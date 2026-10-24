# Plant Disease Detection — MobileNetV2 Baseline

This starter project trains a 17-class Apple/Potato/Tomato leaf image classifier from PlantVillage using PyTorch and transfer learning.

## Classes included

**Apple (4):** Apple scab, Black rot, Cedar apple rust, healthy  
**Potato (3):** Early blight, Late blight, healthy  
**Tomato (10):** Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted spider mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, healthy.

Folder labels follow the common original PlantVillage names such as `Apple___Apple_scab` and `Tomato___healthy`.

## 1. Download the dataset

Use the original PlantVillage repository:
https://github.com/spMohanty/PlantVillage-Dataset

Download the repository ZIP from GitHub, extract it, and locate the `raw/color/` folder. Copy the 38 class subfolders from `raw/color/` into this project's `data/raw/` folder. You can copy all classes; the script selects only the 17 listed above.

Expected shape:
```
plant_disease_detection/
  data/
    raw/
      Apple___Apple_scab/
      Apple___Black_rot/
      ...
      Tomato___healthy/
```
Important: `data/raw/` should contain the class folders directly. Do not leave an extra `PlantVillage-Dataset-master/raw/color/` folder level inside it.

## 2. Create a Windows virtual environment

Open a terminal in the project folder:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If your local computer is slow, Google Colab with a GPU can be used; upload this project and the dataset there.

## 3. Train and evaluate

```powershell
python src/train.py
```
The first run downloads ImageNet-pretrained MobileNetV2 weights, so an internet connection is required. The script:
- finds the selected class folders;
- creates stratified 70% train / 15% validation / 15% test splits;
- trains the new classifier head while freezing MobileNetV2's feature extractor;
- selects the best checkpoint by validation loss;
- evaluates once on the held-out test set;
- saves the model and metrics.

Outputs:
- `models/mobilenetv2_plant_disease.pth`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

Do not write accuracy in a report until you run the script and read the printed test metrics.

## 4. Predict a leaf image

Put a leaf photo anywhere you like and run:
```powershell
python src/predict.py "C:\path\to\your_leaf.jpg"
```
The model returns a class and a confidence score. A high confidence score is not a guarantee the prediction is correct.

## Important methodology notes

- This is a baseline, not a production diagnosis tool.
- PlantVillage images often have controlled backgrounds. Real field images may perform much worse.
- The random split is stratified by class, but if near-duplicate or related images exist across splits, scores can be optimistic. For stronger validation, use duplicate detection or group-aware splitting if image provenance allows it.
- Use macro precision/recall/F1 alongside accuracy because class counts are imbalanced.
- Test separately on PlantDoc or locally collected field images before claiming real-world performance.
- The model only predicts among these 17 classes. It is not designed to say “unknown crop,” “not a leaf,” or diseases outside the list.
- Verify the dataset license and follow attribution/share-alike requirements applicable to your use.

## Suggested manager update

“I'm building a baseline using MobileNetV2 transfer learning and the 17 Apple/Potato/Tomato health/disease classes from PlantVillage. The script uses a stratified 70/15/15 train/validation/test split, saves the best checkpoint based on validation loss, and reports test accuracy, macro precision/recall/F1, per-class scores, and a confusion matrix. I will share the measured results after the run completes. A key limitation is that PlantVillage images are relatively controlled, so real-world field performance must be validated separately.”
