
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import set_seed, get_device, save_output


# Configuration
set_seed(42)

DEVICE = get_device()
BATCH_SIZE = 64
EPOCHS = int(os.getenv("DL_EPOCHS", "10"))
TRAIN_LIMIT = int(os.getenv("DL_TRAIN_LIMIT", "6000"))
LEARNING_RATE = 0.001

output_dir = save_output(__file__)

# Training transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Validation transformations: no random augmentation
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Create two versions of the same Fashion-MNIST dataset
train_full = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=train_transform
)

validation_full = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=validation_transform
)

total_size = min(TRAIN_LIMIT, len(train_full))

if TRAIN_LIMIT <= 0:
    total_size = len(train_full)

indices = torch.randperm(
    total_size,
    generator=torch.Generator().manual_seed(42)
).tolist()

train_size = int(0.8 * total_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(validation_full, val_indices)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# MLP model
class AugmentationMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


model = AugmentationMLP().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)


# Training function
def train():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


# Validation function
def validate():

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


# Training loop
history = []

for epoch in range(EPOCHS):

    train_loss, train_acc = train()
    val_loss, val_acc = validate()

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Train Accuracy: {train_acc:.2f}% | "
        f"Validation Accuracy: {val_acc:.2f}%"
    )


# Save CSV
df = pd.DataFrame(history)

df.to_csv(
    os.path.join(output_dir, "augmentation_results.csv"),
    index=False
)


# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Training Loss")
plt.plot(df["epoch"], df["validation_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Data Augmentation - Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "augmentation_loss.png"))
plt.close()


# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_accuracy"], label="Training Accuracy")
plt.plot(df["epoch"], df["validation_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Data Augmentation - Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "augmentation_accuracy.png"))
plt.close()

print("\nData Augmentation experiment completed!")
print("Results saved in:", output_dir)
