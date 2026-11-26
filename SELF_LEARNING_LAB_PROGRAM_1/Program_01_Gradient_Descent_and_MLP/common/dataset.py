
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_fashion_mnist(
    data_dir,
    batch_size=64,
    train=True,
    download=True
):
    transform = transforms.ToTensor()

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return dataset, loader
