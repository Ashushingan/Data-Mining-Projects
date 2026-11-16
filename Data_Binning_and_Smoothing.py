"""
Data Mining Lab - Experiment 4
Data Binning: Equal-width, Equal-frequency and Smoothing

"""
import matplotlib.pyplot as plt

values = [4, 8, 9, 15, 21, 22, 25, 28, 30, 33, 35, 40]
sorted_values = sorted(values)
number_of_bins = 3

def median_manual(items):
    ordered = sorted(items)
    n = len(ordered)
    if n % 2 == 0:
        return (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return ordered[n // 2]

# Equal-frequency (equal-depth): each bin has the same number of values.
bin_size = len(sorted_values) // number_of_bins
equal_frequency_bins = []
for i in range(number_of_bins):
    start = i * bin_size
    end = (i + 1) * bin_size if i < number_of_bins - 1 else len(sorted_values)
    equal_frequency_bins.append(sorted_values[start:end])

print("Sorted data:", sorted_values)
print("\nEqual-frequency bins:")
for index, bin_values in enumerate(equal_frequency_bins, start=1):
    print(f"Bin {index}: {bin_values}")
    mean_value = sum(bin_values) / len(bin_values)
    median_value = median_manual(bin_values)
    lower_boundary = bin_values[0]
    upper_boundary = bin_values[-1]
    mean_smoothed = [round(mean_value, 2)] * len(bin_values)
    median_smoothed = [median_value] * len(bin_values)
    boundary_smoothed = [
        lower_boundary if abs(value - lower_boundary) <= abs(value - upper_boundary)
        else upper_boundary for value in bin_values
    ]
    print("  Mean smoothing:", mean_smoothed)
    print("  Median smoothing:", median_smoothed)
    print("  Boundary smoothing:", boundary_smoothed)

# Equal-width bins: interval width = (maximum - minimum) / number of bins.
minimum = sorted_values[0]
maximum = sorted_values[-1]
width = (maximum - minimum) / number_of_bins
equal_width_bins = [[] for _ in range(number_of_bins)]
for value in sorted_values:
    bin_index = int((value - minimum) / width)
    if bin_index >= number_of_bins:
        bin_index = number_of_bins - 1
    equal_width_bins[bin_index].append(value)

print("\nEqual-width bins (interval width =", round(width, 2), "):")
for index, bin_values in enumerate(equal_width_bins, start=1):
    print(f"Bin {index}: {bin_values}")

plt.bar([f"Bin {i}" for i in range(1, number_of_bins + 1)],
        [len(b) for b in equal_width_bins])
plt.title("Equal-Width Binning: Frequency per Bin")
plt.xlabel("Bin")
plt.ylabel("Number of values")
plt.tight_layout()
plt.savefig("A4_equal_width_bins.png", dpi=150)
plt.show()
