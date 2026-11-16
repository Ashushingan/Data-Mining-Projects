"""
Data Mining Lab - Experiment 2
Five-Number Summary and Box Plot Analysis

"""
import matplotlib.pyplot as plt

marks = [42, 55, 61, 67, 72, 75, 78, 80, 83, 85,
         88, 91, 94, 97, 100, 65, 74, 82, 89, 58]

def median_manual(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("Cannot calculate median of an empty list.")
    if n % 2 == 0:
        return (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return ordered[n // 2]

def five_number_summary(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("Dataset cannot be empty.")
    minimum = ordered[0]
    maximum = ordered[-1]
    median = median_manual(ordered)
    if n % 2 == 0:
        lower_half = ordered[:n // 2]
        upper_half = ordered[n // 2:]
    else:
        lower_half = ordered[:n // 2]
        upper_half = ordered[n // 2 + 1:]
    q1 = median_manual(lower_half)
    q3 = median_manual(upper_half)
    return minimum, q1, median, q3, maximum, ordered

minimum, q1, median, q3, maximum, ordered = five_number_summary(marks)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = [x for x in marks if x < lower_fence or x > upper_fence]

print("Original marks:", marks)
print("Sorted marks:", ordered)
print("\nFive-number summary")
print("Minimum:", minimum)
print("Q1:", q1)
print("Median (Q2):", median)
print("Q3:", q3)
print("Maximum:", maximum)
print("IQR (Q3 - Q1):", iqr)
print("Lower outlier fence:", lower_fence)
print("Upper outlier fence:", upper_fence)
print("Potential outliers:", outliers if outliers else "None")

plt.boxplot(marks, vert=False)
plt.title("Box Plot of Student Marks")
plt.xlabel("Marks")
plt.tight_layout()
plt.savefig("A2_box_plot.png", dpi=150)
plt.show()
