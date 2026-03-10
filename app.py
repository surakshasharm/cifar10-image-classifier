import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("cifar10_model.pth"))
model.eval()

st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = transform(image).unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output,1)

    st.write("Prediction:", classes[predicted.item()])