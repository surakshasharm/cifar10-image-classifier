# cifar10-image-classifier
Image classification using PyTorch and CIFAR-10 dataset with a Streamlit web interface.

# CIFAR-10 Image Classification using PyTorch

## Overview
This project implements an image classification model trained on the CIFAR-10 dataset using PyTorch.  
The model learns to classify images into 10 categories such as airplanes, cats, dogs, and trucks.

## Dataset
CIFAR-10 dataset contains:
- 60,000 images
- 10 classes
- Image size: 32x32 pixels

Classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model
Pretrained ResNet18 convolutional neural network.

Modifications:
- Final layer adjusted for 10 classes
- Trained using CrossEntropyLoss
- Optimizer: Adam

## Project Structure
train.py        -> trains the model
predict.py      -> predicts class for images
evaluate.py     -> calculates model accuracy
cifar10_model.pth -> trained model

## How to Run

Install dependencies:

pip install torch torchvision matplotlib

Train the model:

python train.py

Test predictions:

python predict.py

Evaluate accuracy:

python evaluate.py

## Results

Model achieved approximately 70–80% accuracy on the CIFAR-10 test dataset.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib#
- 
