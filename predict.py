import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.ToTensor()
])

# load dataset
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

image, label = testset[0]

plt.imshow(image.permute(1,2,0))
plt.title("Actual: " + classes[label])
plt.show()

# load model
model = torchvision.models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(torch.load("cifar10_model.pth"))
model.eval()

image = image.unsqueeze(0)

output = model(image)
_, predicted = torch.max(output,1)

print("Predicted:", classes[predicted.item()])