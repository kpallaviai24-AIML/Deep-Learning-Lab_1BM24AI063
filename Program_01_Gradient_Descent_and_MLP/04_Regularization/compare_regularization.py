
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


# --------------------------------
# 1. Configuration
# --------------------------------

set_seed(42)

DEVICE = get_device()

BATCH_SIZE = 64
EPOCHS = int(os.getenv("DL_EPOCHS", "10"))
TRAIN_LIMIT = int(os.getenv("DL_TRAIN_LIMIT", "6000"))
LEARNING_RATE = 0.001

output_dir = save_output(__file__)


# --------------------------------
# 2. Dataset and transformations
# --------------------------------

normal_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

augmentation_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Create datasets with different transformations
normal_train_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=normal_transform
)

augmented_train_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=augmentation_transform
)

validation_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=normal_transform
)

total_size = min(TRAIN_LIMIT, len(normal_train_data))

if TRAIN_LIMIT <= 0:
    total_size = len(normal_train_data)

indices = torch.randperm(
    total_size,
    generator=torch.Generator().manual_seed(42)
).tolist()

train_size = int(0.8 * total_size)

train_indices = indices[:train_size]
validation_indices = indices[train_size:]

normal_train_loader = DataLoader(
    Subset(normal_train_data, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True
)

augmented_train_loader = DataLoader(
    Subset(augmented_train_data, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    Subset(validation_data, validation_indices),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------
# 3. Define MLP model
# --------------------------------

class RegularizationMLP(nn.Module):

    def __init__(self, use_dropout=False):
        super().__init__()

        if use_dropout:
            self.network = nn.Sequential(
                nn.Flatten(),

                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 10)
            )

        else:
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


# --------------------------------
# 4. Training and validation
# --------------------------------

criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

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

    return total_loss / total, 100 * correct / total


def validate(model):

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

    return total_loss / total, 100 * correct / total


# --------------------------------
# 5. Run one experiment
# --------------------------------

def run_experiment(name, use_dropout=False,
                   weight_decay=0.0, augmentation=False):

    print("\nStarting:", name)

    # Reset seed for a reproducible initialization
    set_seed(42)

    model = RegularizationMLP(
        use_dropout=use_dropout
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )

    train_loader = (
        augmented_train_loader
        if augmentation
        else normal_train_loader
    )

    history = []

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer
        )

        val_loss, val_acc = validate(model)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc
        })

        print(
            f"{name} | Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2f}% | "
            f"Validation Accuracy: {val_acc:.2f}%"
        )

    return history


# --------------------------------
# 6. Compare all techniques
# --------------------------------

experiments = {
    "Without Regularization": {
        "use_dropout": False,
        "weight_decay": 0.0,
        "augmentation": False
    },

    "Dropout": {
        "use_dropout": True,
        "weight_decay": 0.0,
        "augmentation": False
    },

    "L2 Regularization": {
        "use_dropout": False,
        "weight_decay": 0.0001,
        "augmentation": False
    },

    "Data Augmentation": {
        "use_dropout": False,
        "weight_decay": 0.0,
        "augmentation": True
    }
}

all_results = {}
comparison = []

for name, settings in experiments.items():

    history = run_experiment(
        name,
        **settings
    )

    all_results[name] = history

    final_result = history[-1]

    comparison.append({
        "method": name,
        "final_train_accuracy": final_result["train_accuracy"],
        "final_validation_accuracy": final_result["validation_accuracy"],
        "final_train_loss": final_result["train_loss"],
        "final_validation_loss": final_result["validation_loss"]
    })


# --------------------------------
# 7. Save comparison CSV
# --------------------------------

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    os.path.join(output_dir, "regularization_comparison.csv"),
    index=False
)

print("\nFinal Comparison:")
print(comparison_df.to_string(index=False))


# --------------------------------
# 8. Plot validation accuracy
# --------------------------------

plt.figure(figsize=(10, 6))

for name, history in all_results.items():

    epochs = [item["epoch"] for item in history]

    accuracy = [
        item["validation_accuracy"]
        for item in history
    ]

    plt.plot(
        epochs,
        accuracy,
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Comparison of Regularization Techniques")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, "regularization_accuracy_comparison.png")
)

plt.close()


# --------------------------------
# 9. Plot validation loss
# --------------------------------

plt.figure(figsize=(10, 6))

for name, history in all_results.items():

    epochs = [item["epoch"] for item in history]

    losses = [
        item["validation_loss"]
        for item in history
    ]

    plt.plot(
        epochs,
        losses,
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, "regularization_loss_comparison.png")
)

plt.close()


# --------------------------------
# 10. Final output
# --------------------------------

print("\nRegularization comparison completed!")
print("All graphs and CSV results saved in:", output_dir)
print("\n========== FINAL EXPERIMENT RESULTS ==========")

for name, history in all_results.items():

    result = history[-1]

    print(f"\nExperiment: {name}")
    print(f"Training Accuracy: {result['train_accuracy']:.2f}%")
    print(f"Validation Accuracy: {result['validation_accuracy']:.2f}%")
    print(f"Training Loss: {result['train_loss']:.4f}")
    print(f"Validation Loss: {result['validation_loss']:.4f}")

print("\n==============================================")