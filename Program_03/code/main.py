
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ---------------------------------
# STEP 1: CREATE OUTPUT DIRECTORY
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------
# STEP 2: LOAD IRIS DATASET
# ---------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("\nIRIS DATASET")
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)
print("Dataset shape:", X.shape)

# ---------------------------------
# STEP 3: SPLIT DATASET
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------
# STEP 4: CREATE MODEL
# ---------------------------------

model = LogisticRegression(max_iter=500)

# ---------------------------------
# STEP 5: TRAIN MODEL
# ---------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")

# ---------------------------------
# STEP 6: MAKE PREDICTIONS
# ---------------------------------

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

# ---------------------------------
# STEP 7: MODEL ACCURACY
# ---------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# ---------------------------------
# STEP 8: CONFUSION MATRIX
# ---------------------------------

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot(cmap="Blues")

plt.title("Iris Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png")
)

plt.close()

# ---------------------------------
# STEP 9: SAVE RESULTS
# ---------------------------------

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv(
    os.path.join(OUTPUT_DIR, "classification_results.csv"),
    index=False
)

# ---------------------------------
# STEP 10: COMPLETION
# ---------------------------------

print("\nProgram 03 executed successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Generated files are saved in:", OUTPUT_DIR)

