
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Reduce learning rate after every 3 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    epochs = int(os.getenv("DL_EPOCHS", "10"))
    losses = []
    learning_rates = []

    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0

        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

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

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Learning Rate: {current_lr:.6f}"
        )

        scheduler.step()

    execution_time = time.time() - start
    output_dir = save_output(__file__)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Scheduled Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scheduled_learning_rate.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), learning_rates, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_rate_schedule.png"))
    plt.close()

    pd.DataFrame({
        "Epoch": range(1, epochs + 1),
        "Training_Loss": losses,
        "Learning_Rate": learning_rates
    }).to_csv(
        os.path.join(output_dir, "scheduled_learning_rate_results.csv"),
        index=False
    )

    print("Execution time:", round(execution_time, 2), "seconds")


if __name__ == "__main__":
    main()
