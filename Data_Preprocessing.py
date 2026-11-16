"""
Data Mining Lab - Experiment 5: Data Preprocessing
Duplicate removal, missing-value handling, noise reduction,
categorical transformation, normalization, and data reduction.
"""

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Student": ["A", "B", "C", "D", "E", "F", "F"],
    "Age": [19, 20, None, 21, 22, 20, 20],
    "Marks": [78, 85, 91, None, 88, 150, 150],
    "Attendance": [90, 95, 88, 92, None, 60, 60],
    "Department": ["IT", "CS", "IT", "CS", None, "IT", "IT"]
}
df = pd.DataFrame(data)
print("ORIGINAL DATASET")
print(df.to_string(index=False))
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()
for column in ["Age", "Marks", "Attendance"]:
    df[column] = df[column].fillna(df[column].mean())
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

# Marks are valid on a 0-100 scale; cap the invalid noisy value.
df["Marks_Cleaned"] = df["Marks"].apply(lambda value: 100 if value > 100 else value)
df["Department_Code"] = df["Department"].map({"CS": 0, "IT": 1})

minimum = df["Marks_Cleaned"].min()
maximum = df["Marks_Cleaned"].max()
if maximum == minimum:
    df["Marks_MinMax"] = 0.0
else:
    df["Marks_MinMax"] = (df["Marks_Cleaned"] - minimum) / (maximum - minimum)

reduced = df[["Student", "Marks_Cleaned", "Attendance", "Department"]]
print("\nAFTER PREPROCESSING")
print(df.to_string(index=False))
print("\nREDUCED DATASET")
print(reduced.to_string(index=False))
print("\nRows after duplicate removal:", len(df))
print("Missing values after cleaning:")
print(df.isnull().sum().to_string())

plt.bar(reduced["Student"], reduced["Marks_Cleaned"])
plt.title("Cleaned Student Marks")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("23610070_A5_cleaned_marks.png", dpi=150)
plt.show()
