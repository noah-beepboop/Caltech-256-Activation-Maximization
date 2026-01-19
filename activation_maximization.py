import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from PIL import Image

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, image):
        original = image

        output = self.conv1(image)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        output += original
        output = self.relu(output)

        return output


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.inbn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.ResBlock = ResBlock(out_channels)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, image):
        image = self.inconv(image)
        image = self.inbn(image)
        image = self.relu(image)

        image = self.ResBlock(image)

        image = self.pool(image)

        return image

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ResConvBlock(3, 64)
        self.convblock2 = ResConvBlock(64, 128)
        self.convblock3 = ResConvBlock(128, 256)
        self.convblock4 = ResConvBlock(256, 256)
        self.convblock5 = ResConvBlock(256, 512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.3)
        self.fcl = nn.Linear(512, 257)
    
    def forward(self, image):
        image = self.convblock1(image)
        image = self.convblock2(image)
        image = self.convblock3(image)
        image = self.convblock4(image)
        image = self.convblock5(image)

        image = self.pool(image)
        image = torch.flatten(image, 1)

        image = self.dropout(image)
        image = self.fcl(image)

        return image

model = Model()

device = torch.device("cuda")
model = model.to(device)

model.load_state_dict(torch.load(r"")) # Put model path here

model.eval()

image = torch.randn(1, 3, 256, 256, device=device, requires_grad=True)

target_class = 77
iterations = 500
learning_rate = 1e-2

optimizer = torch.optim.Adam([image], lr=learning_rate)

for iteration in range(iterations):
    optimizer.zero_grad()
    output = model(image)
    activation = output[0, target_class]

    loss = -activation # The optimizer will try to minimize it so you gotta reverse-minimize
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        image.clamp_(0, 1)

vutils.save_image(image, "activation_maximization.png", normalize=True)
finished_image = Image.open(r"") # Put image path here
finished_image.show()