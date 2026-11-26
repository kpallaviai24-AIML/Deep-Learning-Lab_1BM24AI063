
import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import set_seed, get_device, save_output


# -------------------- CONFIGURATION --------------------

SEED = 42
EPOCHS = int(os.getenv("DL_EPOCHS", "15"))
TRAIN_LIMIT = int(os.getenv("DL_TRAIN_LIMIT", "6000"))
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 3

set_seed(SEED)
device = get_device()
output_dir = save_output(__file__)


# -------------------- DATASET --------------------

transform = transforms.ToTensor()

dataset = datasets.FashionMNIST(
    root="../../data",
    train=True,
    download=True,
    transform=transform
)

subset_size = min(TRAIN_LIMIT, len(dataset))
dataset, _ = random_split(
    dataset,
    [subset_size, len(dataset) - subset_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

validation_loader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# -------------------- MODEL --------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -------------------- TRAINING AND VALIDATION --------------------

history = []
best_validation_loss = float("inf")
best_epoch = 0
patience_counter = 0

best_model_path = os.path.join(output_dir, "best_early_stopping_model.pth")

print("\nStarting Early Stopping Experiment")

for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = 100 * train_correct / train_total

    # Validation
    model.eval()
    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            validation_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            validation_correct += (predictions == labels).sum().item()
            validation_total += labels.size(0)

    validation_loss /= validation_total
    validation_accuracy = 100 * validation_correct / validation_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Validation Acc: {validation_accuracy:.2f}%"
    )

    # Early Stopping Check
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save(model.state_dict(), best_model_path)
        print("Validation loss improved. Best model saved.")

    else:
        patience_counter += 1
        print(f"No improvement: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered!")
        break


# -------------------- SAVE RESULTS --------------------

results = pd.DataFrame(history)
results.to_csv(
    os.path.join(output_dir, "early_stopping_results.csv"),
    index=False
)

# Load the best model checkpoint
model.load_state_dict(torch.load(best_model_path, map_location=device))

print("\nBest Epoch:", best_epoch)
print("Best Validation Loss:", round(best_validation_loss, 4))
print("Training stopped after:", len(history), "epochs")


# -------------------- PLOT LOSS --------------------

plt.figure(figsize=(8, 5))
plt.plot(results["epoch"], results["train_loss"], marker="o",
         label="Training Loss")
plt.plot(results["epoch"], results["validation_loss"], marker="o",
         label="Validation Loss")

plt.axvline(best_epoch, linestyle="--", label="Best Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Early Stopping - Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "early_stopping_loss.png"))
plt.close()


# -------------------- PLOT ACCURACY --------------------

plt.figure(figsize=(8, 5))
plt.plot(results["epoch"], results["train_accuracy"], marker="o",
         label="Training Accuracy")
plt.plot(results["epoch"], results["validation_accuracy"], marker="o",
         label="Validation Accuracy")

plt.axvline(best_epoch, linestyle="--", label="Best Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Early Stopping - Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "early_stopping_accuracy.png"))
plt.close()


print("\nEarly Stopping experiment completed!")
print("Results saved in:", output_dir)
