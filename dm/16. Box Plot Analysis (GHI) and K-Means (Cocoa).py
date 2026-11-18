
# ============================================================
# ASSIGNMENT 16
# Part A: Box plot analysis (Exp 2) - GHI
#         (own five-number-summary function, box plots, IQR outliers)
# Part B: K-Means (Exp 13) - Cocoa
#         (company-level clustering, k = 3, Silhouette Coefficient)
#
# Calculations and K-Means are implemented manually.
# No NumPy, Pandas, Scikit-learn or statistics library used.
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
# -------------------- K-MEANS (IMPLEMENTED MANUALLY) --------
def euclidean_distance(p1, p2):
    total = 0.0
    for i in range(len(p1)):
        total += (p1[i] - p2[i]) ** 2
    return total ** 0.5

def nearest_centroid(point, centroids):
    best_index = 0
    best_distance = euclidean_distance(point, centroids[0])
    for i in range(1, len(centroids)):
        d = euclidean_distance(point, centroids[i])
        if d < best_distance:
            best_distance = d
            best_index = i
    return best_index

def kmeans(points, k, max_iterations=100):
    n = len(points)
    # Deterministic farthest-point initialisation (first point, then the
    # point farthest from its nearest chosen centroid).
    centroids = [points[0][:]]
    while len(centroids) < k:
        farthest_point = None
        farthest_distance = -1
        for point in points:
            nearest = euclidean_distance(point, centroids[0])
            for c in centroids:
                d = euclidean_distance(point, c)
                if d < nearest:
                    nearest = d
            if nearest > farthest_distance:
                farthest_distance = nearest
                farthest_point = point
        centroids.append(farthest_point[:])
    labels = [-1] * n
    for iteration in range(max_iterations):
        new_labels = [nearest_centroid(p, centroids) for p in points]
        new_centroids = []
        for c in range(k):
            members = [points[i] for i in range(n) if new_labels[i] == c]
            if len(members) == 0:
                new_centroids.append(centroids[c][:])
            else:
                new_centroids.append([calculate_mean([m[d] for m in members])
                                      for d in range(len(points[0]))])
        converged = (new_labels == labels)
        labels = new_labels
        centroids = new_centroids
        if converged:
            break
    return centroids, labels

def calculate_silhouette(points, labels, k):
    """Mean silhouette coefficient and the value for every point."""
    n = len(points)
    values = []
    for i in range(n):
        own = labels[i]
        own_sum = 0.0
        own_count = 0
        other_sum = [0.0] * k
        other_count = [0] * k
        for j in range(n):
            if i == j:
                continue
            d = euclidean_distance(points[i], points[j])
            if labels[j] == own:
                own_sum += d
                own_count += 1
            else:
                other_sum[labels[j]] += d
                other_count[labels[j]] += 1
        if own_count == 0:
            values.append(0.0)
            continue
        a = own_sum / own_count
        b = float("inf")
        for c in range(k):
            if c != own and other_count[c] > 0:
                avg = other_sum[c] / other_count[c]
                if avg < b:
                    b = avg
        bigger = a if a > b else b
        if b == float("inf") or bigger == 0:
            values.append(0.0)
        else:
            values.append((b - a) / bigger)
    return calculate_mean(values), values

# ============================================================
# PART A: BOX PLOT ANALYSIS - GHI
# ============================================================
print("=" * 65)
print("PART A: FIVE-NUMBER SUMMARY AND BOX PLOTS (GHI)")
print("=" * 65)

filename = "GHI_Report.csv"
aliases = {
    "Economy": ["economy", "economygdppercapita", "gdp"],
    "Fam":     ["fam", "family", "socialsupport"],
    "Health":  ["health", "healthlifeexpectancy", "lifeexpectancy"],
    "Freedom": ["freedom"],
}
raw = load_numeric_columns(filename, aliases)
factors = ["Economy", "Fam", "Health", "Freedom"]
column_values = {}
for name in factors:
    column_values[name] = [v for v in raw[name] if v is not None]
print("Records:", len(column_values["Economy"]))

summary = {}
for name in factors:
    values = column_values[name]
    minimum, q1, median, q3, maximum = five_number_summary(values)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = [v for v in values if v < lower_fence or v > upper_fence]
    summary[name] = {"min": minimum, "q1": q1, "median": median, "q3": q3,
                     "max": maximum, "iqr": iqr, "lower": lower_fence,
                     "upper": upper_fence, "outliers": outliers}
    print("\nColumn:", name)
    print("-" * 45)
    print("Minimum :", round(minimum, 5))
    print("Q1      :", round(q1, 5))
    print("Median  :", round(median, 5))
    print("Q3      :", round(q3, 5))
    print("Maximum :", round(maximum, 5))
    print("IQR     :", round(iqr, 5))
    print("Lower IQR Fence:", round(lower_fence, 5))
    print("Upper IQR Fence:", round(upper_fence, 5))
    print("Number of outliers:", len(outliers))
    print("Outlier values:", [round(x, 5) for x in outliers])

# -------------------- SIDE-BY-SIDE BOX PLOTS ----------------
plt.figure(figsize=(10, 6))
plt.boxplot([column_values[name] for name in factors], tick_labels=factors,
            showfliers=True)
plt.title("Box Plots of Economy, Fam, Health and Freedom")
plt.xlabel("Factors")
plt.ylabel("Values")
plt.grid(axis="y")
plt.show()

# -------------------- COMPARE THE SPREAD --------------------
print("\n")
print("COMPARISON OF THE SPREAD OF THE FOUR FACTORS")
print("-" * 65)
print("Factor".ljust(9), "Range".rjust(8), "IQR".rjust(8), "Std".rjust(8),
      "IQR/Median".rjust(11), "Outliers".rjust(9))
for name in factors:
    s = summary[name]
    values = column_values[name]
    print(name.ljust(9), str(round(s["max"] - s["min"], 4)).rjust(8),
          str(round(s["iqr"], 4)).rjust(8),
          str(round(calculate_std(values, sample=True), 4)).rjust(8),
          str(round(s["iqr"] / s["median"], 4)).rjust(11),
          str(len(s["outliers"])).rjust(9))

widest_abs = my_max(factors, key=lambda f: summary[f]["iqr"])
narrow_abs = my_min(factors, key=lambda f: summary[f]["iqr"])
widest_rel = my_max(factors, key=lambda f: summary[f]["iqr"] / summary[f]["median"])
narrow_rel = my_min(factors, key=lambda f: summary[f]["iqr"] / summary[f]["median"])
print("\nWidest spread by IQR (absolute)       :", widest_abs)
print("Narrowest spread by IQR (absolute)    :", narrow_abs)
print("Widest relative spread (IQR / median) :", widest_rel)
print("Narrowest relative spread             :", narrow_rel)
print("Note: the factors are on different scales, so IQR / median is the fairer")
print("comparison. A box that is long relative to its median means countries differ")
print("a lot on that factor; the outliers show countries far from the typical value.")
with_outliers = [f for f in factors if len(summary[f]["outliers"]) > 0]
print("Factors that contain outliers:", with_outliers if with_outliers else "none")

# ============================================================
# PART B: K-MEANS - COCOA (COMPANY LEVEL)
# ============================================================
print("\n")
print("=" * 65)
print("PART B: K-MEANS CLUSTERING OF COCOA COMPANIES (k = 3)")
print("=" * 65)

cocoa_file = "Cocoa.csv"
headers, rows = load_csv(cocoa_file)
i_company = find_column(headers, ["company", "maker"])
i_rating = find_column(headers, ["rating"])
i_cocoa = find_column(headers, ["cocoapercent", "cocoa", "percent"])
print("Company column:", headers[i_company], "| Rating:", headers[i_rating],
      "| Cocoa %:", headers[i_cocoa])

# -------------------- AGGREGATE TO COMPANY LEVEL ------------
groups = {}
for r in rows:
    company = " ".join(r[i_company].split())
    rating = to_float(r[i_rating])
    cocoa = to_float(r[i_cocoa])
    if company == "" or rating is None or cocoa is None:
        continue
    groups.setdefault(company, []).append((rating, cocoa))

# Cocoa % written as a fraction (0.70) is converted to 70.
all_cocoa = []
for g in groups.values():
    for (_, c) in g:
        all_cocoa.append(c)
biggest = my_max(all_cocoa)
scale = 100.0 if biggest <= 1.0 else 1.0

MIN_REVIEWS = 5
companies = []
for company in groups:
    g = groups[company]
    if len(g) >= MIN_REVIEWS:
        companies.append({"name": company,
                          "rating": calculate_mean([x[0] for x in g]),
                          "cocoa": calculate_mean([x[1] for x in g]) * scale,
                          "reviews": float(len(g))})
print("Companies in the data:", len(groups),
      "| with at least", MIN_REVIEWS, "reviews:", len(companies))

feature_keys = ["rating", "cocoa", "reviews"]
feature_labels = ["Mean Rating", "Mean Cocoa %", "Number of reviews"]
print("\nFirst 5 companies:")
for c in companies[:5]:
    print("  ", c["name"].ljust(22), "rating =", round(c["rating"], 3),
          "| cocoa % =", round(c["cocoa"], 2), "| reviews =", int(c["reviews"]))

# -------------------- MIN-MAX NORMALISATION -----------------
mins = [my_min([c[k] for c in companies]) for k in feature_keys]
maxs = [my_max([c[k] for c in companies]) for k in feature_keys]
points = []
for c in companies:
    point = []
    for j in range(len(feature_keys)):
        point.append((c[feature_keys[j]] - mins[j]) / (maxs[j] - mins[j]))
    points.append(point)
print("\nNormalised the 3 features to [0, 1] with Min-Max (so no feature dominates).")

# -------------------- K-MEANS -------------------------------
k = 3
centroids, labels = kmeans(points, k)
print("\nK =", k, "| Features used:", feature_labels)

print("\n")
print("=" * 65)
print("CLUSTER SIZES AND CENTRES (original units)")
print("=" * 65)
overall = {key: calculate_mean([c[key] for c in companies]) for key in feature_keys}
overall_std = {key: calculate_std([c[key] for c in companies]) for key in feature_keys}
cluster_info = []
for cluster in range(k):
    members = [companies[i] for i in range(len(companies)) if labels[i] == cluster]
    info = {"size": len(members), "members": members}
    for key in feature_keys:
        info[key] = calculate_mean([m[key] for m in members]) if members else 0.0
    cluster_info.append(info)
    print("\nCluster", cluster + 1)
    print("Cluster size:", info["size"])
    print("Mean Rating  :", round(info["rating"], 4))
    print("Mean Cocoa % :", round(info["cocoa"], 4))
    print("Mean reviews :", round(info["reviews"], 2))
    print("Centre (normalised):", [round(x, 4) for x in centroids[cluster]])

# -------------------- PLOT ----------------------------------
colors = ["red", "blue", "green"]
plt.figure(figsize=(10, 6))
for cluster in range(k):
    xs = [companies[i]["cocoa"] for i in range(len(companies)) if labels[i] == cluster]
    ys = [companies[i]["rating"] for i in range(len(companies)) if labels[i] == cluster]
    plt.scatter(xs, ys, color=colors[cluster], alpha=0.75,
                label="Cluster " + str(cluster + 1))
    plt.scatter(cluster_info[cluster]["cocoa"], cluster_info[cluster]["rating"],
                marker="X", s=200, color="black")
plt.title("K-Means (k = 3): Mean Cocoa % vs Mean Rating")
plt.xlabel("Mean Cocoa %")
plt.ylabel("Mean Rating")
plt.legend()
plt.grid(True)
plt.show()

# -------------------- SILHOUETTE COEFFICIENT ----------------
silhouette, individual = calculate_silhouette(points, labels, k)
print("\n")
print("=" * 65)
print("SILHOUETTE COEFFICIENT")
print("=" * 65)
print("Overall Silhouette Coefficient:", round(silhouette, 5))
for cluster in range(k):
    vals = [individual[i] for i in range(len(points)) if labels[i] == cluster]
    if vals:
        print("  Cluster", cluster + 1, "average silhouette:", round(calculate_mean(vals), 4))
if silhouette > 0.7:
    print("Interpretation: Strongly separated clusters.")
elif silhouette > 0.5:
    print("Interpretation: Reasonably well-separated clusters.")
elif silhouette > 0.25:
    print("Interpretation: Weak to moderate cluster separation.")
else:
    print("Interpretation: Overlapping or poorly separated clusters.")

# -------------------- DESCRIBE EACH CLUSTER -----------------
def level(value, key):
    z = (value - overall[key]) / overall_std[key] if overall_std[key] > 0 else 0
    if z > 0.4:
        return "high"
    elif z < -0.4:
        return "low"
    return "average"

print("\n")
print("=" * 65)
print("DESCRIPTION OF EACH CLUSTER (compared with all companies)")
print("=" * 65)
print("All companies: mean rating =", round(overall["rating"], 3),
      "| mean cocoa % =", round(overall["cocoa"], 2),
      "| mean reviews =", round(overall["reviews"], 1))
for cluster in range(k):
    info = cluster_info[cluster]
    if info["size"] == 0:
        continue
    r_lv = level(info["rating"], "rating")
    c_lv = level(info["cocoa"], "cocoa")
    v_lv = level(info["reviews"], "reviews")
    print("\nCluster", cluster + 1, "(" + str(info["size"]), "companies)")
    print("  Rating level:", r_lv, "| Cocoa % level:", c_lv, "| Review volume:", v_lv)
    top = my_sort(info["members"], key=lambda m: -m["reviews"])[:5]
    print("  Largest members:", [m["name"] + " (" + str(int(m["reviews"])) + ")" for m in top])
    sentence = "  -> Companies whose rating is " + r_lv + ", cocoa content is " + c_lv + \
               " and review volume is " + v_lv
    if v_lv == "high":
        sentence += " (large, prolific producers)."
    elif v_lv == "low":
        sentence += " (small producers with few reviews)."
    else:
        sentence += "."
    print(sentence)

print("\nASSIGNMENT 16 EXECUTED SUCCESSFULLY")
