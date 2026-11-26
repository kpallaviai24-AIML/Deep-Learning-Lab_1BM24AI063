
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


def run_experiment(loader, optimizer_name, device, epochs, scheduled=False):
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Momentum":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=0.001
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001
        )

    scheduler = None

    if scheduled:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.5
        )

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

        if scheduler is not None:
            scheduler.step()

        print(
            f"{optimizer_name} | "
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
        )

    execution_time = time.time() - start

    return losses, execution_time


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

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    epochs = int(os.getenv("DL_EPOCHS", "10"))

    results = []
    curves = {}

    # Compare three optimizers
    for name in ["Momentum", "Adam", "RMSprop"]:
        losses, elapsed = run_experiment(
            loader, name, device, epochs
        )

        curves[name] = losses

        results.append({
            "Experiment": name,
            "Final_Training_Loss": losses[-1],
            "Execution_Time_Seconds": elapsed
        })

    # Compare constant and scheduled learning rates
    constant_losses, constant_time = run_experiment(
        loader, "Constant LR", device, epochs, scheduled=False
    )

    scheduled_losses, scheduled_time = run_experiment(
        loader, "Scheduled LR", device, epochs, scheduled=True
    )

    output_dir = save_output(__file__)

    # Optimizer comparison graph
    plt.figure(figsize=(9, 6))

    for name, losses in curves.items():
        plt.plot(
            range(1, epochs + 1),
            losses,
            marker="o",
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Optimizer Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "optimizer_comparison.png")
    )
    plt.close()

    # Learning rate comparison graph
    plt.figure(figsize=(9, 6))

    plt.plot(
        range(1, epochs + 1),
        constant_losses,
        marker="o",
        label="Constant Learning Rate"
    )

    plt.plot(
        range(1, epochs + 1),
        scheduled_losses,
        marker="o",
        label="Scheduled Learning Rate"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "learning_rate_comparison.png")
    )
    plt.close()

    # Combined training loss graph
    plt.figure(figsize=(9, 6))

    for name, losses in curves.items():
        plt.plot(range(1, epochs + 1), losses, label=name)

    plt.plot(
        range(1, epochs + 1),
        constant_losses,
        label="Constant LR"
    )

    plt.plot(
        range(1, epochs + 1),
        scheduled_losses,
        label="Scheduled LR"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "training_loss.png")
    )
    plt.close()

    results.extend([
        {
            "Experiment": "Constant Learning Rate",
            "Final_Training_Loss": constant_losses[-1],
            "Execution_Time_Seconds": constant_time
        },
        {
            "Experiment": "Scheduled Learning Rate",
            "Final_Training_Loss": scheduled_losses[-1],
            "Execution_Time_Seconds": scheduled_time
        }
    ])

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "optimizer_results.csv"),
        index=False
    )

    print("\nFinal Optimizer Comparison:")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
