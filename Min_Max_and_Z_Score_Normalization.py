"""
Data Mining Lab - Experiment 3
Data Normalization: Min-Max and Z-score

"""
import matplotlib.pyplot as plt

values = [12, 15, 18, 20, 22, 25, 30, 35]
n = len(values)
minimum = min(values)
maximum = max(values)
mean = sum(values) / n

squared_deviations = []
for value in values:
    squared_deviations.append((value - mean) ** 2)
variance = sum(squared_deviations) / n
std_dev = variance ** 0.5

if maximum == minimum:
    minmax_values = [0.0 for _ in values]
else:
    minmax_values = [(value - minimum) / (maximum - minimum)
                     for value in values]

if std_dev == 0:
    zscore_values = [0.0 for _ in values]
else:
    zscore_values = [(value - mean) / std_dev for value in values]

print("Original values:", values)
print("Minimum:", minimum, "Maximum:", maximum)
print("Mean:", round(mean, 4))
print("Population standard deviation:", round(std_dev, 4))
print("\nOriginal\tMin-Max\t\tZ-score")
for i in range(n):
    print(f"{values[i]}\t\t{minmax_values[i]:.4f}\t\t{zscore_values[i]:.4f}")

plt.plot(values, minmax_values, marker="o")
plt.title("Min-Max Normalization")
plt.xlabel("Original value")
plt.ylabel("Normalized value [0, 1]")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("A3_minmax.png", dpi=150)
plt.show()

plt.bar([str(v) for v in values], zscore_values)
plt.axhline(0, linewidth=1)
plt.title("Z-score Normalization")
plt.xlabel("Original value")
plt.ylabel("Z-score")
plt.tight_layout()
plt.savefig("A3_zscore.png", dpi=150)
plt.show()
