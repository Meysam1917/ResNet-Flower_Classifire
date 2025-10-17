import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from torchvision.datasets import Flowers102
classes = Flowers102(root="./data", split="train").classes
# --------------------------
# Basic setup
# --------------------------
st.title("🌸 Flower Classifier")
st.write("Upload a flower image and let the model predict its class.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Load model
# --------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 102)
    model.load_state_dict(torch.load("best_resnet_flower.pth", map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# --------------------------
# Define image transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------
# Upload image
# --------------------------
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    st.markdown("---")
    st.subheader(f"🌼 Predicted Flower: {classes[class_idx].capitalize()}")

    st.write("Note: This index corresponds to the class ID from the Flowers102 dataset.")

