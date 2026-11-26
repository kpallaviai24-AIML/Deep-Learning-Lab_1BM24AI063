
import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)

csv_path = os.path.join(
    output_dir,
    "overfitting_results.csv"
)

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        "Run overfitting_experiment.py first to generate overfitting_results.csv"
    )

df = pd.read_csv(csv_path)

# Training vs Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(df["epoch"], df["train_loss"], marker="o", label="Training Loss")
plt.plot(df["epoch"], df["validation_loss"], marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting: Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, "training_validation_loss.png")
)

plt.show()
plt.close()

# Training vs Validation Accuracy
plt.figure(figsize=(8, 5))

plt.plot(df["epoch"], df["train_accuracy"], marker="o", label="Training Accuracy")
plt.plot(df["epoch"], df["validation_accuracy"], marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Overfitting: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, "training_validation_accuracy.png")
)

plt.show()
plt.close()

print("Graphs generated successfully!")
