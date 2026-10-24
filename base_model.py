import torch
import torch.nn as nn


# Step 1: Create our CNN model
class PlantDiseaseCNN(nn.Module):

    def __init__(self, num_classes=38):
        super().__init__()

        # Step 2: Extract features from plant images
        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second convolution block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Third convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Step 3: Convert feature maps into a vector
        self.flatten = nn.Flatten()

        # Step 4: Predict the plant disease class
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    # Step 5: Define how images pass through the model
    def forward(self, x):

        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


# Step 6: Test the model
if __name__ == "__main__":

    model = PlantDiseaseCNN(num_classes=38)

    # Create four dummy RGB images of size 128 x 128
    sample_images = torch.randn(4, 3, 128, 128)

    # Pass the images through the model
    predictions = model(sample_images)

    print("MODEL ARCHITECTURE")
    print(model)

    print("\nINPUT SHAPE:")
    print(sample_images.shape)

    print("\nOUTPUT SHAPE:")
    print(predictions.shape)

    # Count all trainable parameters
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTRAINABLE PARAMETERS:")
    print(f"{total_parameters:,}")



