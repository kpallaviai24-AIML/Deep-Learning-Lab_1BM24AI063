
import os
import sys
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
EPOCHS = int(os.getenv("DL_EPOCHS", "10"))
TRAIN_LIMIT = int(os.getenv("DL_TRAIN_LIMIT", "6000"))
BATCH_SIZE = 64
LEARNING_RATE = 0.001

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


# -------------------- PLAIN MLP MODEL --------------------

class PlainMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


# -------------------- RESIDUAL BLOCK --------------------

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        output = self.block(x)

        # Skip connection
        output = output + identity

        return self.relu(output)


# -------------------- RESIDUAL MLP MODEL --------------------

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.input_layer = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.residual_block = ResidualBlock(256)

        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.residual_block(x)
        x = self.output_layer(x)

        return x


# -------------------- TRAINING FUNCTION --------------------

def train_model(model, model_name):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    print(f"\nStarting: {model_name}")

    for epoch in range(EPOCHS):

        # Training
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

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

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                validation_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                validation_correct += (
                    predictions == labels
                ).sum().item()

                validation_total += labels.size(0)

        validation_loss /= validation_total
        validation_accuracy = (
            100 * validation_correct / validation_total
        )

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

    # Save trained model
    model_path = os.path.join(
        output_dir,
        model_name.lower().replace(" ", "_") + ".pth"
    )

    torch.save(model.state_dict(), model_path)

    return history


# -------------------- RUN BOTH MODELS --------------------

# Reset the random seed before each model for a more controlled comparison.
set_seed(SEED)
plain_model = PlainMLP()
plain_history = train_model(plain_model, "Plain MLP")

set_seed(SEED)
residual_model = ResidualMLP()
residual_history = train_model(residual_model, "Residual MLP")


# -------------------- SAVE CSV RESULTS --------------------

plain_df = pd.DataFrame(plain_history)
residual_df = pd.DataFrame(residual_history)

plain_df.to_csv(
    os.path.join(output_dir, "plain_mlp_results.csv"),
    index=False
)

residual_df.to_csv(
    os.path.join(output_dir, "residual_mlp_results.csv"),
    index=False
)

comparison = pd.DataFrame({
    "epoch": plain_df["epoch"],
    "plain_train_accuracy": plain_df["train_accuracy"],
    "residual_train_accuracy": residual_df["train_accuracy"],
    "plain_validation_accuracy": plain_df["validation_accuracy"],
    "residual_validation_accuracy": residual_df["validation_accuracy"],
    "plain_train_loss": plain_df["train_loss"],
    "residual_train_loss": residual_df["train_loss"],
    "plain_validation_loss": plain_df["validation_loss"],
    "residual_validation_loss": residual_df["validation_loss"]
})

comparison.to_csv(
    os.path.join(output_dir, "plain_vs_residual_comparison.csv"),
    index=False
)


# -------------------- ACCURACY COMPARISON GRAPH --------------------

plt.figure(figsize=(9, 5))

plt.plot(
    plain_df["epoch"],
    plain_df["validation_accuracy"],
    marker="o",
    label="Plain MLP"
)

plt.plot(
    residual_df["epoch"],
    residual_df["validation_accuracy"],
    marker="o",
    label="Residual MLP"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Plain MLP vs Residual MLP - Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "plain_vs_residual_accuracy.png")
)
plt.close()


# -------------------- LOSS COMPARISON GRAPH --------------------

plt.figure(figsize=(9, 5))

plt.plot(
    plain_df["epoch"],
    plain_df["validation_loss"],
    marker="o",
    label="Plain MLP"
)

plt.plot(
    residual_df["epoch"],
    residual_df["validation_loss"],
    marker="o",
    label="Residual MLP"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Plain MLP vs Residual MLP - Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "plain_vs_residual_loss.png")
)
plt.close()


# -------------------- FINAL COMPARISON --------------------

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    f"Plain MLP Final Validation Accuracy: "
    f"{plain_df['validation_accuracy'].iloc[-1]:.2f}%"
)

print(
    f"Residual MLP Final Validation Accuracy: "
    f"{residual_df['validation_accuracy'].iloc[-1]:.2f}%"
)

print(
    f"Plain MLP Final Validation Loss: "
    f"{plain_df['validation_loss'].iloc[-1]:.4f}"
)

print(
    f"Residual MLP Final Validation Loss: "
    f"{residual_df['validation_loss'].iloc[-1]:.4f}"
)

print("\nPlain MLP vs Residual MLP experiment completed!")
print("All results saved in:", output_dir)
