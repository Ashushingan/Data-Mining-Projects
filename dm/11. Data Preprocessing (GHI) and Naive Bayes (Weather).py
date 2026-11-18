
# ============================================================
# ASSIGNMENT 11
# Part A: Data Preprocessing (Exp 5) - GHI dataset
#         (missing values, IQR outlier capping, Z-score, PCA)
# Part B: Naive Bayes (Exp 11) - Weather (numeric) dataset
#         (mixed Naive Bayes + leave-one-out cross-validation)
#
# All calculations (median, IQR, Z-score, PCA, Naive Bayes,
# cross-validation) are implemented manually.
# No NumPy, Pandas or Scikit-learn used.
# Matplotlib is used only for plotting.
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
# -------------------- WEATHER (NUMERIC) DATASET -------------
# The standard 14-record weather.numeric dataset.
# Columns: outlook, temperature, humidity, windy, play
WEATHER = [
    ("sunny",    85, 85, "FALSE", "no"),
    ("sunny",    80, 90, "TRUE",  "no"),
    ("overcast", 83, 86, "FALSE", "yes"),
    ("rainy",    70, 96, "FALSE", "yes"),
    ("rainy",    68, 80, "FALSE", "yes"),
    ("rainy",    65, 70, "TRUE",  "no"),
    ("overcast", 64, 65, "TRUE",  "yes"),
    ("sunny",    72, 95, "FALSE", "no"),
    ("sunny",    69, 70, "FALSE", "yes"),
    ("rainy",    75, 80, "FALSE", "yes"),
    ("sunny",    75, 70, "TRUE",  "yes"),
    ("overcast", 72, 90, "TRUE",  "yes"),
    ("overcast", 81, 75, "FALSE", "yes"),
    ("rainy",    71, 91, "TRUE",  "no"),
]
weather = []
for w in WEATHER:
    weather.append({"outlook": w[0], "temperature": float(w[1]),
                    "humidity": float(w[2]), "windy": w[3], "play": w[4]})

# ============================================================
# PART A: DATA PREPROCESSING - GHI
# ============================================================
print("=" * 65)
print("PART A: DATA PREPROCESSING (GHI)")
print("=" * 65)

filename = "GHI_Report.csv"
features = ["Economy", "Fam", "Health", "Freedom"]
aliases = {
    "Economy": ["economy", "economygdppercapita", "gdp"],
    "Fam":     ["fam", "family", "socialsupport"],
    "Health":  ["health", "healthlifeexpectancy", "lifeexpectancy"],
    "Freedom": ["freedom"],
}
columns = load_numeric_columns(filename, aliases)
n = len(columns["Economy"])
print("Total records:", n)

# -------------------- STEP 1: 0 VALUES -> MISSING -> MEDIAN --
print("\nSTEP 1: Treat 0 as missing and impute with the column median")
print("-" * 65)
for name in features:
    valid = []
    for v in columns[name]:
        if v is not None and v != 0:
            valid.append(v)
    median_value = find_median(manual_sort(valid))
    replaced = 0
    for i in range(n):
        if columns[name][i] is None or columns[name][i] == 0:
            columns[name][i] = median_value
            replaced += 1
    print(name.ljust(8), "| zeros/missing replaced:", str(replaced).rjust(3),
          "| median used:", round(median_value, 5))

# -------------------- STEP 2: IQR OUTLIER CAPPING -----------
print("\nSTEP 2: Detect and cap outliers with the IQR rule")
print("-" * 65)
for name in features:
    minimum, q1, median, q3, maximum = five_number_summary(columns[name])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    capped = 0
    for i in range(n):
        v = columns[name][i]
        if v < lower_fence:
            columns[name][i] = lower_fence
            capped += 1
        elif v > upper_fence:
            columns[name][i] = upper_fence
            capped += 1
    print(name.ljust(8), "| Q1 =", round(q1, 4), "| Q3 =", round(q3, 4),
          "| fences = [" + str(round(lower_fence, 4)) + ", " +
          str(round(upper_fence, 4)) + "]", "| capped:", capped)

# -------------------- STEP 3: Z-SCORE NORMALISATION ---------
print("\nSTEP 3: Z-score normalisation  z = (x - mean) / std")
print("-" * 65)
zdata = {}
for name in features:
    m = calculate_mean(columns[name])
    s = calculate_std(columns[name])
    zdata[name] = [(v - m) / s for v in columns[name]]
    print(name.ljust(8), "| mean =", round(m, 5), "| std =", round(s, 5),
          "| after Z-score: mean =", round(calculate_mean(zdata[name]), 5),
          ", std =", round(calculate_std(zdata[name]), 5))

# -------------------- STEP 4: PCA (DATA REDUCTION) ----------
print("\nSTEP 4: PCA - reduce 4 factors to 2 principal components")
print("-" * 65)
data_matrix = []
for i in range(n):
    data_matrix.append([zdata[name][i] for name in features])

d = len(features)
# Covariance matrix of the Z-scored data (means are already 0);
# it equals the correlation matrix of the four factors.
cov = [[0.0] * d for _ in range(d)]
for a in range(d):
    for b in range(d):
        total = 0.0
        for i in range(n):
            total += data_matrix[i][a] * data_matrix[i][b]
        cov[a][b] = total / n

print("Covariance matrix (= correlation matrix for Z-scored data):")
for row in cov:
    print("  ", [round(x, 4) for x in row])

def jacobi_eigen(matrix, max_sweeps=100):
    """Eigenvalues / eigenvectors of a symmetric matrix (Jacobi method)."""
    size = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for sweep in range(max_sweeps):
        off_diagonal = 0.0
        for i in range(size):
            for j in range(i + 1, size):
                off_diagonal += a[i][j] ** 2
        if off_diagonal < 1e-20:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if my_abs(a[p][q]) < 1e-18:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                if theta >= 0:
                    t = 1.0 / (theta + (1.0 + theta * theta) ** 0.5)
                else:
                    t = -1.0 / (-theta + (1.0 + theta * theta) ** 0.5)
                c = 1.0 / ((1.0 + t * t) ** 0.5)
                s = t * c
                for k in range(size):          # rotate columns p, q
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(size):          # rotate rows p, q
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(size):          # accumulate eigenvectors
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    values = [a[i][i] for i in range(size)]
    vectors = [[v[k][i] for k in range(size)] for i in range(size)]
    return values, vectors

eigenvalues, eigenvectors = jacobi_eigen(cov)
order = my_sort(list(range(d)), key=lambda i: -eigenvalues[i])   # largest first
eigenvalues = [eigenvalues[i] for i in order]
eigenvectors = [eigenvectors[i] for i in order]

total_variance = 0.0
for ev in eigenvalues:
    total_variance += ev

print("\nEigenvalues, explained variance ratio:")
cumulative = 0.0
for i in range(d):
    ratio = eigenvalues[i] / total_variance
    cumulative += ratio
    print("  PC" + str(i + 1), "| eigenvalue =", round(eigenvalues[i], 5),
          "| explained variance ratio =", round(ratio, 5),
          "| cumulative =", round(cumulative, 5))

print("\nPrincipal component loadings (weights of each factor):")
for i in range(2):
    print("  PC" + str(i + 1), {features[j]: round(eigenvectors[i][j], 4)
                                for j in range(d)})

# Project the data on the first two principal components.
scores = []
for i in range(n):
    row = []
    for pc in range(2):
        value = 0.0
        for j in range(d):
            value += data_matrix[i][j] * eigenvectors[pc][j]
        row.append(value)
    scores.append(row)

print("\nFirst 5 records reduced to 2 dimensions (PC1, PC2):")
for i in range(5 if n > 5 else n):
    print("  record", i + 1, [round(x, 4) for x in scores[i]])

ratio_2pc = (eigenvalues[0] + eigenvalues[1]) / total_variance
print("\nEXPLAINED VARIANCE RATIO of the 2 principal components:",
      round(ratio_2pc, 5), "(" + str(round(ratio_2pc * 100, 2)) + " %)")

plt.figure(figsize=(8, 6))
plt.scatter([s[0] for s in scores], [s[1] for s in scores], alpha=0.7)
plt.title("GHI data reduced to 2 principal components")
plt.xlabel("PC1 (" + str(round(eigenvalues[0] / total_variance * 100, 1)) + " %)")
plt.ylabel("PC2 (" + str(round(eigenvalues[1] / total_variance * 100, 1)) + " %)")
plt.grid(True)
plt.show()

# ============================================================
# PART B: MIXED NAIVE BAYES - WEATHER (NUMERIC)
# ============================================================
print("\n")
print("=" * 65)
print("PART B: MIXED NAIVE BAYES CLASSIFIER (WEATHER)")
print("=" * 65)

CLASSES = ["yes", "no"]
OUTLOOK_VALUES = ["sunny", "overcast", "rainy"]
WINDY_VALUES = ["TRUE", "FALSE"]

def gaussian_pdf(x, mean, std):
    return (1.0 / (((2 * math.pi) ** 0.5) * std)) * \
           math.exp(-((x - mean) ** 2) / (2 * std * std))

def train_nb(records):
    """Learn priors, Laplace-smoothed tables and Gaussian parameters."""
    model = {"prior": {}, "class_count": {}, "outlook": {}, "outlook_count": {},
             "windy": {}, "windy_count": {}, "temperature": {}, "humidity": {}}
    for c in CLASSES:
        rows_c = [r for r in records if r["play"] == c]
        nc = len(rows_c)
        model["class_count"][c] = nc
        model["prior"][c] = nc / len(records)
        # Laplace smoothing: (count + 1) / (class size + number of values)
        model["outlook"][c] = {}
        model["outlook_count"][c] = {}
        for v in OUTLOOK_VALUES:
            cnt = len([r for r in rows_c if r["outlook"] == v])
            model["outlook_count"][c][v] = cnt
            model["outlook"][c][v] = (cnt + 1) / (nc + len(OUTLOOK_VALUES))
        model["windy"][c] = {}
        model["windy_count"][c] = {}
        for v in WINDY_VALUES:
            cnt = len([r for r in rows_c if r["windy"] == v])
            model["windy_count"][c][v] = cnt
            model["windy"][c][v] = (cnt + 1) / (nc + len(WINDY_VALUES))
        # Gaussian parameters (sample standard deviation, n - 1)
        for attribute in ["temperature", "humidity"]:
            vals = [r[attribute] for r in rows_c]
            std = calculate_std(vals, sample=True)
            if std < 1e-6:
                std = 1e-6
            model[attribute][c] = (calculate_mean(vals), std)
    return model

def classify(model, day, verbose=False):
    """Return (predicted class, normalised posterior probabilities)."""
    scores = {}
    for c in CLASSES:
        p_prior = model["prior"][c]
        p_outlook = model["outlook"][c][day["outlook"]]
        p_windy = model["windy"][c][day["windy"]]
        t_mean, t_std = model["temperature"][c]
        h_mean, h_std = model["humidity"][c]
        p_temp = gaussian_pdf(day["temperature"], t_mean, t_std)
        p_hum = gaussian_pdf(day["humidity"], h_mean, h_std)
        scores[c] = p_prior * p_outlook * p_windy * p_temp * p_hum
        if verbose:
            nc = model["class_count"][c]
            oc = model["outlook_count"][c][day["outlook"]]
            wc = model["windy_count"][c][day["windy"]]
            print("\n  Class =", c)
            print("   P(" + c + ")                    =", nc, "/", len(weather),
                  "=", round(p_prior, 6))
            print("   P(outlook=" + day["outlook"] + " | " + c + ")  = (" + str(oc),
                  "+ 1) / (" + str(nc), "+ 3) =", round(p_outlook, 6))
            print("   P(windy=" + day["windy"] + " | " + c + ")    = (" + str(wc),
                  "+ 1) / (" + str(nc), "+ 2) =", round(p_windy, 6))
            print("   P(temp=" + str(day["temperature"]) + " | " + c + ")      "
                  "= Gaussian(mean=" + str(round(t_mean, 3)) + ", std=" +
                  str(round(t_std, 3)) + ") =", round(p_temp, 6))
            print("   P(humidity=" + str(day["humidity"]) + " | " + c + ")  "
                  "= Gaussian(mean=" + str(round(h_mean, 3)) + ", std=" +
                  str(round(h_std, 3)) + ") =", round(p_hum, 6))
            print("   Score = product of the five terms =", scores[c])
    total = 0.0
    for c in CLASSES:
        total += scores[c]
    posterior = {}
    for c in CLASSES:
        posterior[c] = scores[c] / total if total > 0 else 0.0
    best = CLASSES[0]
    for c in CLASSES:
        if posterior[c] > posterior[best]:
            best = c
    return best, posterior

# -------------------- PROBABILITY TABLES --------------------
model = train_nb(weather)

print("\nPRIOR PROBABILITIES")
for c in CLASSES:
    print("  P(play=" + c + ") =", model["class_count"][c], "/", len(weather),
          "=", round(model["prior"][c], 4))

print("\nOUTLOOK  (Laplace smoothed: (count + 1) / (class size + 3))")
print("  value      | yes count  P(.|yes) | no count  P(.|no)")
for v in OUTLOOK_VALUES:
    print("  " + v.ljust(10), "|", str(model["outlook_count"]["yes"][v]).rjust(6),
          str(round(model["outlook"]["yes"][v], 4)).rjust(10), "|",
          str(model["outlook_count"]["no"][v]).rjust(6),
          str(round(model["outlook"]["no"][v], 4)).rjust(10))

print("\nWINDY  (Laplace smoothed: (count + 1) / (class size + 2))")
print("  value      | yes count  P(.|yes) | no count  P(.|no)")
for v in WINDY_VALUES:
    print("  " + v.ljust(10), "|", str(model["windy_count"]["yes"][v]).rjust(6),
          str(round(model["windy"]["yes"][v], 4)).rjust(10), "|",
          str(model["windy_count"]["no"][v]).rjust(6),
          str(round(model["windy"]["no"][v], 4)).rjust(10))

print("\nGAUSSIAN PARAMETERS (mean, sample std) per class")
print("  attribute    | yes mean   yes std | no mean    no std")
for attribute in ["temperature", "humidity"]:
    ym, ys = model[attribute]["yes"]
    nm, ns = model[attribute]["no"]
    print("  " + attribute.ljust(12), "|", str(round(ym, 3)).rjust(8),
          str(round(ys, 3)).rjust(8), "|", str(round(nm, 3)).rjust(8),
          str(round(ns, 3)).rjust(8))

# -------------------- CLASSIFY THE NEW DAY ------------------
new_day = {"outlook": "sunny", "temperature": 66.0, "humidity": 90.0,
           "windy": "TRUE"}
print("\n")
print("CLASSIFYING THE NEW DAY:", new_day)
print("-" * 65)
prediction, posterior = classify(model, new_day, verbose=True)
print("\n  Normalised posterior probabilities:")
for c in CLASSES:
    print("   P(play=" + c + " | new day) =", round(posterior[c], 4))
print("  PREDICTION: play =", prediction)

# -------------------- LEAVE-ONE-OUT CROSS-VALIDATION --------
print("\n")
print("LEAVE-ONE-OUT CROSS-VALIDATION (14 folds)")
print("-" * 65)
tp = fp = tn = fn = 0           # positive class = "yes"
print("fold | actual | predicted | P(yes)  | result")
for i in range(len(weather)):
    training = [weather[j] for j in range(len(weather)) if j != i]
    test_record = weather[i]
    fold_model = train_nb(training)
    predicted, post = classify(fold_model, test_record)
    actual = test_record["play"]
    if actual == "yes" and predicted == "yes":
        tp += 1
    elif actual == "no" and predicted == "yes":
        fp += 1
    elif actual == "no" and predicted == "no":
        tn += 1
    else:
        fn += 1
    print(str(i + 1).rjust(4), "|", actual.ljust(6), "|", predicted.ljust(9), "|",
          str(round(post["yes"], 4)).ljust(7), "|",
          "correct" if actual == predicted else "WRONG")

total_records = tp + fp + tn + fn
accuracy = (tp + tn) / total_records
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("\nCONFUSION MATRIX (positive class = yes)")
print("                 Predicted yes   Predicted no")
print("  Actual yes  |", str(tp).rjust(8), "(TP)", str(fn).rjust(8), "(FN)")
print("  Actual no   |", str(fp).rjust(8), "(FP)", str(tn).rjust(8), "(TN)")
print("\nAccuracy  =", "(TP + TN) / total =", round(accuracy, 4))
print("Precision =", "TP / (TP + FP)    =", round(precision, 4))
print("Recall    =", "TP / (TP + FN)    =", round(recall, 4))
print("F1-score  =", "2PR / (P + R)     =", round(f1, 4))

print("\nASSIGNMENT 11 EXECUTED SUCCESSFULLY")
