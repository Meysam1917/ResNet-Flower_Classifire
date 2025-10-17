import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = datasets.Flowers102(root="./data", split="test", download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = test_dataset.classes

# -------------------------------
# Load best model
# -------------------------------
best_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = best_model.fc.in_features
best_model.fc = nn.Linear(num_features, 102)
best_model.load_state_dict(torch.load("best_resnet_finetuned.pth", map_location=device))
best_model = best_model.to(device)
best_model.eval()

print(" Best model loaded successfully")

# -------------------------------
# Evaluate accuracy
# -------------------------------
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

acc = 100 * correct / total
print(f"Test Accuracy: {acc:.2f}%")

# -------------------------------
# Show sample predictions
# -------------------------------
def imshow(img, title=None):
    img = img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")

# Display predictions for a few samples
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = best_model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    imshow(images[i])
    plt.title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}")
plt.show()
