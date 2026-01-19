import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
# from torchinfo import summary

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    ), # these are Imagenet normalizations
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(root=r"") # Put dataset path here

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_dataset = TransformSubset(train_dataset, train_transforms)
test_dataset = TransformSubset(test_dataset, test_transforms)

NUM_WORKERS = 4

train_load = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_load = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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
# print(summary(model, input_size=(1, 3, 256, 256)))

device = torch.device("cuda")
model = model.to(device)

if __name__ == "__main__":

    epochs = 200

    criterion = nn.CrossEntropyLoss()  # for classification
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(datetime.now())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_load, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()         
            outputs = model(images)      
            loss = criterion(outputs, labels)
            loss.backward()               
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            blank, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_load.dataset)
        epoch_acc = correct / total
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        scheduler.step()

        if epoch % 10 == 0:

            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0

            with torch.no_grad():
                for images, labels in test_load:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * images.size(0)
                    blank, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_loss /= len(test_load.dataset)
            test_acc = test_correct / test_total
            print(f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            torch.save(model.state_dict(), f"model_{epoch}_{test_acc:.4f}.pth")
            print(f"Saved model at epoch {epoch}")

    print(datetime.now())