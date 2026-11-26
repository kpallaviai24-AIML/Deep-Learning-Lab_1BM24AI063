
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)
