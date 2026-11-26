
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# STEP 1: CREATE OUTPUT DIRECTORY
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------
# STEP 2: CREATE SAMPLE DATASET
# ---------------------------------

data = {
    "Student": ["A", "B", "C", "D", "E"],
    "Maths": [85, 90, 78, 92, 88],
    "Science": [80, 85, 82, 95, 90],
    "English": [75, 88, 80, 90, 85]
}

df = pd.DataFrame(data)

print("\nSTUDENT DATASET")
print(df)

# ---------------------------------
# STEP 3: DATA ANALYSIS
# ---------------------------------

df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nSTUDENT AVERAGES")
print(df[["Student", "Average"]])

# ---------------------------------
# STEP 4: SAVE CSV
# ---------------------------------

csv_path = os.path.join(OUTPUT_DIR, "analysis_results.csv")
df.to_csv(csv_path, index=False)

# ---------------------------------
# STEP 5: BAR CHART
# ---------------------------------

plt.figure(figsize=(8, 5))

plt.bar(df["Student"], df["Average"])

plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.title("Student Average Marks")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "bar_chart.png"))
plt.close()

# ---------------------------------
# STEP 6: LINE CHART
# ---------------------------------

plt.figure(figsize=(8, 5))

plt.plot(df["Student"], df["Maths"], marker="o", label="Maths")
plt.plot(df["Student"], df["Science"], marker="o", label="Science")
plt.plot(df["Student"], df["English"], marker="o", label="English")

plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Subject-wise Performance")

plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "line_chart.png"))
plt.close()

# ---------------------------------
# STEP 7: DISPLAY COMPLETION
# ---------------------------------

print("\nProgram 02 executed successfully!")
print("Generated files are saved in:", OUTPUT_DIR)

