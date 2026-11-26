
import os
import sys
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import set_seed, get_device, save_output


def create_model(device):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)


def train_method(dataset, batch_size, method, device, epochs):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = create_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

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

        print(f"{method} | Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    elapsed = time.time() - start

    return {
        "method": method,
        "losses": losses,
        "execution_time": elapsed,
        "final_loss": losses[-1]
    }


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

    epochs = int(os.getenv("DL_EPOCHS", "10"))

    results = [
        train_method(dataset, len(dataset), "Batch GD", device, epochs),
        train_method(dataset, 1, "Stochastic GD", device, epochs),
        train_method(dataset, 64, "Mini-Batch GD", device, epochs)
    ]

    output_dir = save_output(__file__)

    plt.figure(figsize=(9, 6))

    for result in results:
        plt.plot(
            range(1, epochs + 1),
            result["losses"],
            marker="o",
            label=result["method"]
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Comparison of Gradient Descent Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_descent_comparison.png"))
    plt.close()

    for result in results:
        filename = result["method"].lower().replace(" ", "_") + "_loss.png"

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), result["losses"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title(result["method"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    rows = []

    for result in results:
        rows.append({
            "Method": result["method"],
            "Final_Training_Loss": result["final_loss"],
            "Execution_Time_Seconds": result["execution_time"]
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "comparison_results.csv"),
        index=False
    )

    print("\nFinal Comparison:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
