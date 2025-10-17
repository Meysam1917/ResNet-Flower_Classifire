import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch.optim.lr_scheduler import StepLR




# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced transformations for training data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation and test should NOT be augmented
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.Flowers102(root="./data", split="train", download=False, transform=train_transform)
val_dataset   = datasets.Flowers102(root="./data", split="val", download=False, transform=test_transform)
test_dataset  = datasets.Flowers102(root="./data", split="test", download=False, transform=test_transform)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
print("Labels:", labels[:5])


# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers so we only train the last one (for now)
for param in model.parameters():
    param.requires_grad = True

# Replace the final layer (fc = fully connected)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 102)  # 102 flower classes

# Move to GPU/CPU
model = model.to(device)

print(model.fc)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = StepLR(optimizer, step_size=3, gamma=0.1)



num_epochs = 15  # you can increase later
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 20)

    # ---- TRAINING PHASE ----
    model.train()  # enable training mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # clear previous gradients

        outputs = model(inputs)  # forward pass
        _, preds = torch.max(outputs, 1)  # predicted class
        loss = criterion(outputs, labels)  # compute loss

        loss.backward()       # backward pass
        optimizer.step()      # update weights

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    # ---- VALIDATION PHASE ----
    model.eval()  # turn off dropout/batchnorm
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

    # ---- SAVE BEST MODEL ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_resnet_finetuned.pth")

        print("OK Best model saved!\n")

print("Training complete.")
print(f"Best validation accuracy: {best_acc:.4f}")


# Load the best saved model
best_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final layer again to match 102 classes
num_features = best_model.fc.in_features
best_model.fc = nn.Linear(num_features, 102)

# Load weights
best_model.load_state_dict(torch.load("best_resnet_finetuned.pth", map_location=device))
best_model = best_model.to(device)
best_model.eval()

print("Best model loaded and ready for testing.")


correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")
