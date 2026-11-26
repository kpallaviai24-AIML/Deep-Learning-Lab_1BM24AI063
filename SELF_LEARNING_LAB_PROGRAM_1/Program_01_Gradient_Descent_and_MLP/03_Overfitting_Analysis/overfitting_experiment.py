
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import set_seed, get_device, save_output


# -------------------------------
# 1. Configuration
# -------------------------------

set_seed(42)

DEVICE = get_device()
BATCH_SIZE = 64
EPOCHS = int(os.getenv("DL_EPOCHS", "15"))
TRAIN_LIMIT = int(os.getenv("DL_TRAIN_LIMIT", "6000"))
LEARNING_RATE = 0.001

output_dir = save_output(__file__)

# -------------------------------
# 2. Load Fashion-MNIST
# -------------------------------

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

# Use a subset to reduce execution time
if TRAIN_LIMIT > 0:
    dataset = torch.utils.data.Subset(
        dataset,
        range(min(TRAIN_LIMIT, len(dataset)))
    )

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------------
# 3. Define a Large MLP
# -------------------------------

class LargeMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


model = LargeMLP().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -------------------------------
# 4. Training Function
# -------------------------------

def train_one_epoch():

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

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# -------------------------------
# 5. Validation Function
# -------------------------------

def validate():

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# -------------------------------
# 6. Train the Model
# -------------------------------

history = []

print("\nStarting Overfitting Experiment...\n")

for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# -------------------------------
# 7. Save Results
# -------------------------------

df = pd.DataFrame(history)

csv_path = os.path.join(
    output_dir,
    "overfitting_results.csv"
)

df.to_csv(csv_path, index=False)

# -------------------------------
# 8. Plot Training and Validation Loss
# -------------------------------

plt.figure(figsize=(8, 5))

plt.plot(df["epoch"], df["train_loss"], label="Training Loss")
plt.plot(df["epoch"], df["validation_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, "training_validation_loss.png")
)

plt.close()

# -------------------------------
# 9. Plot Accuracy
# -------------------------------

plt.figure(figsize=(8, 5))

plt.plot(df["epoch"], df["train_accuracy"], label="Training Accuracy")
plt.plot(df["epoch"], df["validation_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, "training_validation_accuracy.png")
)

plt.close()

print("\nExperiment completed successfully!")
print("Results saved in:", output_dir)
print("CSV file:", csv_path)
