
# ============================================================
# ASSIGNMENT 15
# Part A: Visualisation (Exp 1) - Cocoa
#         (attribute types, dataset characteristics, 5 charts)
# Part B: Decision Tree (Exp 10) - Breast Cancer Wisconsin (Diagnostic)
#         (Gini tree vs Entropy tree, max_depth = 4)
#
# Part A: statistics are calculated manually, Matplotlib draws the charts.
# Part B: Scikit-learn is used for the decision trees (as the experiment
#         requires criterion='gini' and criterion='entropy').
# ============================================================
import csv
import math
import matplotlib.pyplot as plt

# -------------------- CSV HELPERS ---------------------------
def load_csv(filename):
    """Read a CSV file. Returns (headers, rows); every cell is a stripped string."""
    with open(filename, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        all_rows = []
        for r in reader:
            has_data = False
            for cell in r:
                if cell.strip() != "":
                    has_data = True
            if has_data:
                all_rows.append(r)
    headers = [h.strip() for h in all_rows[0]]
    rows = []
    for r in all_rows[1:]:
        r = [cell.strip() for cell in r]
        while len(r) < len(headers):
            r.append("")
        rows.append(r)
    return headers, rows

def norm_name(text):
    """'Cocoa Percent' -> 'cocoapercent' (used to match column names)."""
    out = ""
    for ch in text.lower():
        if ch.isalnum():
            out += ch
    return out

def find_column(headers, aliases):
    """Index of the first header that matches one of the aliases."""
    names = [norm_name(h) for h in headers]
    for alias in aliases:                      # exact match first
        if alias in names:
            return names.index(alias)
    for alias in aliases:                      # then partial match
        for i in range(len(names)):
            if alias in names[i]:
                return i
    raise KeyError("Column not found. Tried " + str(aliases) +
                   " in headers " + str(headers))

MISSING_TEXT = ("", "na", "n/a", "nan", "null", "none", "?", "-")

def to_float(text):
    """Text -> float. Returns None when the cell is missing / not numeric."""
    t = text.strip().replace("%", "")
    if t.lower() in MISSING_TEXT:
        return None
    try:
        return float(t)
    except ValueError:
        return None

def load_numeric_columns(filename, wanted):
    """wanted = {"Name": [aliases...]}. Returns {"Name": [float or None, ...]}."""
    headers, rows = load_csv(filename)
    columns = {}
    for name in wanted:
        idx = find_column(headers, wanted[name])
        print("Column", name, "<-", headers[idx])
        columns[name] = [to_float(r[idx]) for r in rows]
    return columns

# -------------------- MANUAL STATISTICS ---------------------
# -------------------- OWN VERSIONS OF min / max / sum / sort --
# (no built-in min(), max(), sum(), sorted(), abs() are used in this program)
def my_sum(values):
    total = 0
    for v in values:
        total += v
    return total

def my_min(values, key=None):
    """Smallest item (compared by key(item) when a key function is given)."""
    best = None
    best_key = None
    for v in values:
        k = v if key is None else key(v)
        if best_key is None or k < best_key:
            best = v
            best_key = k
    return best

def my_max(values, key=None):
    """Largest item (compared by key(item) when a key function is given)."""
    best = None
    best_key = None
    for v in values:
        k = v if key is None else key(v)
        if best_key is None or k > best_key:
            best = v
            best_key = k
    return best

def my_abs(x):
    if x < 0:
        return -x
    return x

def my_ceil(x):
    whole = int(x)
    if x > whole:
        return whole + 1
    return whole

def my_sort(values, key=None, reverse=False):
    """Insertion sort (stable). Returns a new sorted list."""
    items = []
    for v in values:
        items.append(v)
    keys = []
    for v in items:
        keys.append(v if key is None else key(v))
    for i in range(1, len(items)):
        item = items[i]
        k = keys[i]
        j = i - 1
        while j >= 0 and ((keys[j] < k) if reverse else (keys[j] > k)):
            items[j + 1] = items[j]
            keys[j + 1] = keys[j]
            j = j - 1
        items[j + 1] = item
        keys[j + 1] = k
    return items

def my_unique(values):
    """Distinct values, in order of first appearance."""
    result = []
    for v in values:
        found = False
        for r in result:
            if r == v:
                found = True
                break
        if not found:
            result.append(v)
    return result

def calculate_mean(values):
    if len(values) == 0:
        return 0.0
    total = 0.0
    for v in values:
        total += v
    return total / len(values)

def calculate_std(values, sample=False):
    """Standard deviation (population by default, sample if sample=True)."""
    n = len(values)
    if n < 2:
        return 0.0
    m = calculate_mean(values)
    total = 0.0
    for v in values:
        total += (v - m) ** 2
    divisor = (n - 1) if sample else n
    return (total / divisor) ** 0.5

def manual_sort(values):
    """Insertion sort."""
    arr = values[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j = j - 1
        arr[j + 1] = key
    return arr

def find_median(sorted_values):
    n = len(sorted_values)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

def five_number_summary(values):
    """Returns (minimum, Q1, median, Q3, maximum) - median-of-halves method."""
    s = manual_sort(values)
    n = len(s)
    median = find_median(s)
    if n % 2 == 0:
        lower_half = s[:n // 2]
        upper_half = s[n // 2:]
    else:
        lower_half = s[:n // 2]
        upper_half = s[n // 2 + 1:]
    return s[0], find_median(lower_half), median, find_median(upper_half), s[n - 1]

def percentile(sorted_values, q):
    """q-th quantile (0..1) with linear interpolation."""
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = low + 1
    if high > len(sorted_values) - 1:
        high = len(sorted_values) - 1
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)

# ============================================================
# PART A: VISUALISATION - COCOA
# ============================================================
print("=" * 65)
print("PART A: VISUALISATION OF THE COCOA DATASET")
print("=" * 65)

cocoa_file = "Cocoa.csv"
headers, rows = load_csv(cocoa_file)
n_rows = len(rows)

# -------------------- ATTRIBUTE TYPES -----------------------
print("\nDATASET CHARACTERISTICS")
print("-" * 65)
print("Type of data     : record (tabular) data - one row per chocolate bar review")
print("Number of records:", n_rows)
print("Dimensionality   :", len(headers), "attributes")

print("\nATTRIBUTE TYPES")
print("Attribute".ljust(24), "Missing".rjust(8), "Unique".rjust(7), "  Type")
for j in range(len(headers)):
    cells = [r[j] for r in rows]
    missing = len([c for c in cells if c.strip().lower() in MISSING_TEXT])
    present = [c for c in cells if c.strip().lower() not in MISSING_TEXT]
    unique = len(my_unique(present))
    numeric = [c for c in present if to_float(c) is not None]
    name = norm_name(headers[j])
    if len(present) > 0 and len(numeric) >= 0.95 * len(present):
        if "date" in name or "year" in name:
            kind = "Numeric - interval (year)"
        elif "rating" in name:
            kind = "Numeric - ordinal / interval (score scale)"
        elif name in ("ref", "id"):
            kind = "Numeric - identifier (nominal)"
        else:
            kind = "Numeric - ratio (percentage)" if "%" in "".join(present[:50]) or "percent" in name \
                   else "Numeric - ratio"
    else:
        kind = "Categorical - nominal (text)"
    print(headers[j].ljust(24), str(missing).rjust(8), str(unique).rjust(7), " ", kind)

# -------------------- EXTRACT THE COLUMNS USED --------------
i_year = find_column(headers, ["reviewdate", "year", "reviewyear"])
i_rating = find_column(headers, ["rating"])
i_cocoa = find_column(headers, ["cocoapercent", "cocoa", "percent"])
i_loc = find_column(headers, ["companylocation", "location"])

records = []
for r in rows:
    year = to_float(r[i_year])
    rating = to_float(r[i_rating])
    cocoa = to_float(r[i_cocoa])
    location = r[i_loc].strip()
    records.append({"year": None if year is None else int(year), "rating": rating,
                    "cocoa": cocoa, "location": location})
if my_max([r["cocoa"] for r in records if r["cocoa"] is not None]) <= 1.0:
    for r in records:                       # 0.70 -> 70 (%)
        if r["cocoa"] is not None:
            r["cocoa"] *= 100

# Rating class (same definition in all cocoa assignments)
def rating_class(rating):
    if rating < 3.0:
        return "Low (< 3.0)"
    elif rating <= 3.5:
        return "Medium (3.0 - 3.5)"
    return "High (> 3.5)"

# -------------------- CHART 1: LINE - REVIEWS PER YEAR ------
per_year = {}
for r in records:
    if r["year"] is not None:
        per_year[r["year"]] = per_year.get(r["year"], 0) + 1
years = my_sort(list(per_year))
plt.figure(figsize=(9, 5))
plt.plot(years, [per_year[y] for y in years], marker="o")
plt.title("Number of Reviews per Year")
plt.xlabel("Review Date (year)")
plt.ylabel("Number of reviews")
plt.grid(True)
plt.show()

# -------------------- CHART 2: BAR - TOP 10 LOCATIONS -------
loc_ratings = {}
for r in records:
    if r["rating"] is not None and r["location"] != "":
        loc_ratings.setdefault(r["location"], []).append(r["rating"])
loc_mean = []
for loc in loc_ratings:
    if len(loc_ratings[loc]) >= 20:                      # minimum 20 reviews
        loc_mean.append((loc, calculate_mean(loc_ratings[loc]), len(loc_ratings[loc])))
loc_mean = my_sort(loc_mean, key=lambda t: -t[1])
top10 = loc_mean[:10]
print("\nTOP 10 COMPANY LOCATIONS BY MEAN RATING (minimum 20 reviews)")
for loc, m, cnt in top10:
    print("  ", loc.ljust(20), "mean rating =", round(m, 4), "| reviews =", cnt)
plt.figure(figsize=(10, 6))
plt.barh([t[0] for t in top10][::-1], [t[1] for t in top10][::-1], color="saddlebrown")
plt.title("10 Highest-Rated Company Locations (min. 20 reviews)")
plt.xlabel("Mean Rating")
plt.ylabel("Company Location")
plt.grid(axis="x")
plt.show()

# -------------------- CHART 3: PIE - RATING CLASS -----------
class_counts = {}
for r in records:
    if r["rating"] is not None:
        c = rating_class(r["rating"])
        class_counts[c] = class_counts.get(c, 0) + 1
order = ["Low (< 3.0)", "Medium (3.0 - 3.5)", "High (> 3.5)"]
print("\nRATING CLASS COUNTS:", {k: class_counts.get(k, 0) for k in order})
plt.figure(figsize=(7, 7))
plt.pie([class_counts.get(k, 0) for k in order], labels=order, autopct="%1.1f%%",
        startangle=90)
plt.title("Rating Class")
plt.show()

# -------------------- CHART 4: HISTOGRAM - COCOA % ----------
cocoa_values = [r["cocoa"] for r in records if r["cocoa"] is not None]
plt.figure(figsize=(9, 5))
plt.hist(cocoa_values, bins=15, color="chocolate", edgecolor="black")
plt.title("Histogram of Cocoa %")
plt.xlabel("Cocoa %")
plt.ylabel("Frequency")
plt.grid(axis="y")
plt.show()
print("\nCocoa %: min =", my_min(cocoa_values), "| max =", my_max(cocoa_values),
      "| mean =", round(calculate_mean(cocoa_values), 2),
      "| median =", find_median(my_sort(cocoa_values)))

# -------------------- CHART 5: SCATTER - YEAR vs MEAN RATING
year_ratings = {}
for r in records:
    if r["year"] is not None and r["rating"] is not None:
        year_ratings.setdefault(r["year"], []).append(r["rating"])
xs = my_sort(list(year_ratings))
ys = [calculate_mean(year_ratings[y]) for y in xs]
plt.figure(figsize=(9, 5))
plt.scatter(xs, ys, s=70, color="darkgreen")
plt.title("Review Date vs Mean Rating")
plt.xlabel("Review Date (year)")
plt.ylabel("Mean Rating")
plt.grid(True)
plt.show()
for x, y in zip(xs, ys):
    print("  ", x, "mean rating =", round(y, 4))

# ============================================================
# PART B: DECISION TREE - BREAST CANCER WISCONSIN (DIAGNOSTIC)
# ============================================================
print("\n")
print("=" * 65)
print("PART B: DECISION TREES - BREAST CANCER WISCONSIN (DIAGNOSTIC)")
print("=" * 65)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

cancer = load_breast_cancer()
X = cancer.data
feature_names = list(cancer.feature_names)
# In scikit-learn, target 0 = malignant and 1 = benign.
# We define y = 1 for MALIGNANT (M) and y = 0 for BENIGN (B).
y = (cancer.target == 0).astype(int)
malignant_count = 0
for label in y:
    if label == 1:
        malignant_count += 1
print("Records:", X.shape[0], "| Features:", X.shape[1],
      "| Malignant (M):", malignant_count, "| Benign (B):", len(y) - malignant_count)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print("Training records:", len(X_train), "| Test records:", len(X_test))

trees = {}
for criterion in ["gini", "entropy"]:
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=4, random_state=42)
    tree.fit(X_train, y_train)
    trees[criterion] = tree

# -------------------- PLOT BOTH TREES -----------------------
fig, axes = plt.subplots(1, 2, figsize=(30, 11))
for ax, criterion in zip(axes, ["gini", "entropy"]):
    plot_tree(trees[criterion], feature_names=feature_names,
              class_names=["Benign", "Malignant"], filled=True, rounded=True,
              fontsize=8, ax=ax)
    ax.set_title("Decision tree, criterion = '" + criterion + "', max_depth = 4")
plt.tight_layout()
plt.show()

# -------------------- COMPARE THE TREES ---------------------
scores = {}
for criterion in ["gini", "entropy"]:
    tree = trees[criterion]
    y_pred = tree.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)          # rows = actual, cols = predicted
    scores[criterion] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "cm": cm,
    }
    root = feature_names[tree.tree_.feature[0]]
    print("\n" + "-" * 65)
    print("CRITERION =", criterion.upper())
    print("-" * 65)
    print("Root attribute :", root, "(threshold <=", round(tree.tree_.threshold[0], 4), ")")
    print("Tree depth     :", tree.get_depth(), "| leaves:", tree.get_n_leaves())
    print("Top 5 feature importances:")
    ranked = my_sort(list(range(len(feature_names))), key=lambda i: -tree.feature_importances_[i])
    for i in ranked[:5]:
        print("   ", feature_names[i].ljust(24), round(tree.feature_importances_[i], 4))
    print("Accuracy  :", round(scores[criterion]["accuracy"], 4))
    print("Precision :", round(scores[criterion]["precision"], 4), "(malignant = positive)")
    print("Recall    :", round(scores[criterion]["recall"], 4), "(malignant = positive)")
    print("F1-score  :", round(scores[criterion]["f1"], 4))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                 Pred Benign  Pred Malignant")
    print("   Actual Benign   ", str(cm[0][0]).rjust(6), str(cm[0][1]).rjust(12))
    print("   Actual Malignant", str(cm[1][0]).rjust(6), str(cm[1][1]).rjust(12))

print("\nSIDE-BY-SIDE COMPARISON")
print("Measure".ljust(12), "Gini".rjust(10), "Entropy".rjust(10))
for key in ["accuracy", "precision", "recall", "f1"]:
    print(key.ljust(12), str(round(scores["gini"][key], 4)).rjust(10),
          str(round(scores["entropy"][key], 4)).rjust(10))
print("Root attribute: gini =", feature_names[trees["gini"].tree_.feature[0]],
      "| entropy =", feature_names[trees["entropy"].tree_.feature[0]])

# -------------------- WHICH ONE DO WE PREFER? ---------------
if scores["gini"]["recall"] > scores["entropy"]["recall"]:
    best = "gini"
    reason = "it finds more of the malignant cases (higher recall)."
elif scores["entropy"]["recall"] > scores["gini"]["recall"]:
    best = "entropy"
    reason = "it finds more of the malignant cases (higher recall)."
elif scores["entropy"]["accuracy"] > scores["gini"]["accuracy"]:
    best = "entropy"
    reason = "recall is equal, but its accuracy / precision / F1 are higher."
elif scores["gini"]["accuracy"] > scores["entropy"]["accuracy"]:
    best = "gini"
    reason = "recall is equal, but its accuracy / precision / F1 are higher."
else:
    best = "gini"
    reason = "both trees perform the same; Gini is cheaper (no logarithm to compute)."
print("\nPREFERRED TREE:", best.upper())
print("Reason:", reason)
print("\nWHY RECALL FOR MALIGNANT CASES MATTERS MOST")
print("Recall = TP / (TP + FN) = share of real cancers that the model finds.")
print("A false negative means a malignant tumour is called benign: the patient is")
print("sent home untreated and the disease may spread. A false positive only leads")
print("to extra tests that clear the patient. Missing a cancer is far more costly")
print("than a false alarm, so recall for the malignant class is the key measure.")

print("\nASSIGNMENT 15 EXECUTED SUCCESSFULLY")
