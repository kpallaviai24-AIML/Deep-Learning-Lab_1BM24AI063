
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
L2_LAMBDA = 0.0001

output_dir = save_output(__file__)

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

dataset = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

if TRAIN_LIMIT > 0:
    dataset = torch.utils.data.Subset(
        dataset, range(min(TRAIN_LIMIT, len(dataset)))
    )

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# MLP model
class RegularizedMLP(nn.Module):

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


model = RegularizedMLP().to(DEVICE)

criterion = nn.CrossEntropyLoss()

# L2 regularization is applied using weight_decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=L2_LAMBDA
)


# Training
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


# Validation
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


# Save results
df = pd.DataFrame(history)

df.to_csv(
    os.path.join(output_dir, "l2_results.csv"),
    index=False
)


# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Training Loss")
plt.plot(df["epoch"], df["validation_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("L2 Regularization - Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "l2_loss.png"))
plt.close()


# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_accuracy"], label="Training Accuracy")
plt.plot(df["epoch"], df["validation_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("L2 Regularization - Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "l2_accuracy.png"))
plt.close()

print("\nL2 Regularization experiment completed!")
print("Results saved in:", output_dir)
