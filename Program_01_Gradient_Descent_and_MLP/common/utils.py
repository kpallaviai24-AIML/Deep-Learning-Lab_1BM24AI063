
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    return device


def save_output(script_file):
    folder = os.path.dirname(os.path.abspath(script_file))

    output_dir = os.path.join(folder, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir
