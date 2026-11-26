
import os
import sys
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import set_seed, get_device, save_output


def main():
    set_seed(42)
    device = get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    limit = int(os.getenv("DL_TRAIN_LIMIT", "6000"))
    if limit > 0:
        train_data = torch.utils.data.Subset(
            train_data, range(min(limit, len(train_data)))
        )

    loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    epochs = int(os.getenv("DL_EPOCHS", "10"))
    losses = []

    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    elapsed = time.time() - start

    output_dir = save_output(__file__)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker="o", label="Momentum")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Momentum Optimizer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "momentum_results.png"))
    plt.close()

    pd.DataFrame({
        "Epoch": range(1, epochs+1),
        "Training_Loss": losses
    }).to_csv(
        os.path.join(output_dir, "momentum_results.csv"),
        index=False
    )

    print("Execution time:", round(elapsed, 2), "seconds")


if __name__ == "__main__":
    main()
