"""
Data Mining Lab - Experiment 1
Study of Data Types, Dataset Characteristics and Data Visualization

"""
import pandas as pd
import matplotlib.pyplot as plt

# Small, self-contained student performance dataset
data = {
    "Student": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "Math": [85, 78, 92, 70, 88, 76, 95, 82],
    "Science": [80, 75, 90, 72, 85, 79, 93, 84],
    "Attendance": [92, 85, 96, 78, 88, 81, 98, 90]
}
df = pd.DataFrame(data)

print("DATASET")
print(df.to_string(index=False))
print("\nRows:", len(df))
print("Columns:", list(df.columns))
print("\nData types:")
print(df.dtypes)
print("\nBasic dataset characteristics:")
print("Math minimum:", df["Math"].min(), "maximum:", df["Math"].max())
print("Science minimum:", df["Science"].min(), "maximum:", df["Science"].max())
print("Missing values by column:")
print(df.isnull().sum())

# Bar chart: compare Math and Science for each student
df.plot(x="Student", y=["Math", "Science"], kind="bar")
plt.title("Student Marks by Subject")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("A1_bar_chart.png", dpi=150)
plt.show()

# Line chart: compare subject trends across the student sequence
plt.figure()
plt.plot(df["Student"], df["Math"], marker="o", label="Math")
plt.plot(df["Student"], df["Science"], marker="s", label="Science")
plt.title("Marks Across Students")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("A1_line_chart.png", dpi=150)
plt.show()

# Pie chart: share of total marks in both subjects
plt.figure()
plt.pie([df["Math"].sum(), df["Science"].sum()],
        labels=["Math", "Science"], autopct="%1.1f%%", startangle=90)
plt.title("Share of Total Marks")
plt.tight_layout()
plt.savefig("A1_pie_chart.png", dpi=150)
plt.show()

# Histogram: distribution of Math marks
plt.figure()
plt.hist(df["Math"], bins=5, edgecolor="black")
plt.title("Distribution of Mathematics Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("A1_histogram.png", dpi=150)
plt.show()

# Scatter plot: relationship between Math and Science marks
plt.figure()
plt.scatter(df["Math"], df["Science"])
for _, row in df.iterrows():
    plt.annotate(row["Student"], (row["Math"], row["Science"]),
                 xytext=(4, 3), textcoords="offset points")
plt.title("Math vs Science Marks")
plt.xlabel("Math")
plt.ylabel("Science")
plt.tight_layout()
plt.savefig("A1_scatter_plot.png", dpi=150)
plt.show()
