
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Subset
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

    dataset = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    limit = int(os.getenv("DL_TRAIN_LIMIT", "6000"))
    if limit > 0:
        dataset = Subset(dataset, range(min(limit, len(dataset))))

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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

        average_loss = total_loss / total_samples
        losses.append(average_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    execution_time = time.time() - start
    output_dir = save_output(__file__)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Mini-Batch Gradient Descent")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mini_batch_loss.png"))
    plt.close()

    pd.DataFrame({
        "Epoch": range(1, epochs + 1),
        "Training_Loss": losses
    }).to_csv(os.path.join(output_dir, "mini_batch_results.csv"), index=False)

    print("Execution time:", round(execution_time, 2), "seconds")


if __name__ == "__main__":
    main()
