
import torch
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "best_early_stopping_model.pth"
)

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

print("Model loaded successfully!")
print("\nSaved model parameters:")

for name, parameter in checkpoint.items():
    print(name, parameter.shape)
