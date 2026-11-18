
# ============================================================
# ASSIGNMENT 12
# Part A: Generalisation / Concept Hierarchies (Exp 6) - GHI
# Part B: Attribute Selection (Exp 9) - Weather (numeric)
#         (Information Gain and Gini index)
#
# All calculations are implemented manually.
# No NumPy, Pandas or Scikit-learn used.
# ============================================================
import csv
import math

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
# PART A: GENERALISATION - CONCEPT HIERARCHIES (GHI)
# ============================================================
print("=" * 65)
print("PART A: CONCEPT HIERARCHIES (GHI)")
print("=" * 65)

filename = "GHI_Report.csv"
aliases = {
    "H_Score": ["hscore", "happinessscore", "score"],
    "Economy": ["economy", "economygdppercapita", "gdp"],
    "Fam":     ["fam", "family", "socialsupport"],
    "Health":  ["health", "healthlifeexpectancy", "lifeexpectancy"],
    "Freedom": ["freedom"],
}
raw = load_numeric_columns(filename, aliases)
names = ["H_Score", "Economy", "Fam", "Health", "Freedom"]

# Keep only complete records.
records = []
for i in range(len(raw["H_Score"])):
    row = {}
    complete = True
    for name in names:
        if raw[name][i] is None:
            complete = False
        row[name] = raw[name][i]
    if complete:
        records.append(row)
print("Complete records used:", len(records))

# -------------------- CONCEPT HIERARCHY 1: H_Score ----------
# Fixed cut-offs 6.5, 5.5, 4.5
def h_level(score):
    if score >= 6.5:
        return "Very High"
    elif score >= 5.5:
        return "High"
    elif score >= 4.5:
        return "Medium"
    return "Low"

# -------------------- CONCEPT HIERARCHY 2: Economy ----------
# Terciles: bottom third = Poor, middle third = Middle, top third = Rich
economy_sorted = my_sort([r["Economy"] for r in records])
tercile_1 = percentile(economy_sorted, 1 / 3)
tercile_2 = percentile(economy_sorted, 2 / 3)
print("\nEconomy tercile cut-offs: T1 =", round(tercile_1, 4),
      "| T2 =", round(tercile_2, 4))

def economy_level(value):
    if value <= tercile_1:
        return "Poor"
    elif value <= tercile_2:
        return "Middle"
    return "Rich"

for r in records:
    r["H_Level"] = h_level(r["H_Score"])
    r["E_Level"] = economy_level(r["Economy"])

h_levels = ["Very High", "High", "Medium", "Low"]
e_levels = ["Rich", "Middle", "Poor"]

# -------------------- MEAN OF EACH FACTOR PER H_Score LEVEL --
print("\nMEAN OF EACH FACTOR PER H_SCORE LEVEL")
print("-" * 65)
print("H_Score level".ljust(14), "Count".rjust(5), "H_Score".rjust(9),
      "Economy".rjust(9), "Fam".rjust(9), "Health".rjust(9), "Freedom".rjust(9))
level_means = {}
for level in h_levels:
    members = [r for r in records if r["H_Level"] == level]
    level_means[level] = {"count": len(members)}
    line = level.ljust(14) + str(len(members)).rjust(6)
    for name in names:
        if len(members) > 0:
            m = calculate_mean([r[name] for r in members])
            level_means[level][name] = m
            line += str(round(m, 4)).rjust(10)
        else:
            level_means[level][name] = None
            line += "n/a".rjust(10)
    print(line)

# -------------------- ECONOMY LEVEL x H_Score LEVEL TABLE ---
print("\nCOUNT TABLE: ECONOMY LEVEL (rows) x H_SCORE LEVEL (columns)")
print("-" * 65)
header = "Economy level".ljust(14)
for level in h_levels:
    header += level.rjust(11)
header += "Total".rjust(8)
print(header)
cross = {}
for e in e_levels:
    cross[e] = {}
    line = e.ljust(14)
    row_total = 0
    for h in h_levels:
        cnt = len([r for r in records if r["E_Level"] == e and r["H_Level"] == h])
        cross[e][h] = cnt
        row_total += cnt
        line += str(cnt).rjust(11)
    cross[e]["Total"] = row_total
    line += str(row_total).rjust(8)
    print(line)
line = "Total".ljust(14)
for h in h_levels:
    line += str(len([r for r in records if r["H_Level"] == h])).rjust(11)
line += str(len(records)).rjust(8)
print(line)

# -------------------- COMMENT ON THE SUMMARY ----------------
print("\nWHAT THE SUMMARY REVEALS")
print("-" * 65)
filled = [lv for lv in h_levels if level_means[lv]["count"] > 0]
for name in ["Economy", "Fam", "Health", "Freedom"]:
    values = [level_means[lv][name] for lv in filled]
    increasing = True
    for i in range(len(values) - 1):
        if values[i] < values[i + 1]:      # list runs from Very High to Low
            increasing = False
    first, last = filled[0], filled[-1]
    print("-", name, ": mean goes from", round(level_means[first][name], 3),
          "(" + first + ") to", round(level_means[last][name], 3),
          "(" + last + ") ->",
          "rises steadily with happiness." if increasing
          else "does NOT rise steadily with happiness.")

def share(group, targets):
    total = cross[group]["Total"]
    hit = 0
    for t in targets:
        hit += cross[group][t]
    return 100.0 * hit / total if total > 0 else 0.0

print("- Rich countries that are Very High/High happiness :",
      round(share("Rich", ["Very High", "High"]), 1), "%")
print("- Poor countries that are Low/Medium happiness     :",
      round(share("Poor", ["Low", "Medium"]), 1), "%")
print("- Poor countries that are Very High/High happiness :",
      round(share("Poor", ["Very High", "High"]), 1), "%")

rich_high = share("Rich", ["Very High", "High"])
poor_high = share("Poor", ["Very High", "High"])
if rich_high > poor_high:
    print("Interpretation: richer countries (by Economy) are more often in the top")
    print("happiness levels than poorer ones, so economic strength goes with happiness.")
else:
    print("Interpretation: Economy level does not separate the happiness levels here.")

# Which factor changes least between the happiest and the least happy level?
smallest_name = None
smallest_change = None
for name in ["Economy", "Fam", "Health", "Freedom"]:
    top = level_means[filled[0]][name]
    bottom = level_means[filled[-1]][name]
    biggest_side = my_max([my_abs(top), my_abs(bottom), 1e-9])
    change = my_abs(top - bottom) / biggest_side
    if smallest_change is None or change < smallest_change:
        smallest_change = change
        smallest_name = name
print("The factor that differs least between the", filled[0], "and", filled[-1],
      "levels is", smallest_name, "(relative change", str(round(smallest_change * 100, 1)) + " %),")
print("so it explains the difference in happiness the least.")

# ============================================================
# PART B: ATTRIBUTE SELECTION - WEATHER (NUMERIC)
# ============================================================
print("\n")
print("=" * 65)
print("PART B: ATTRIBUTE SELECTION - INFORMATION GAIN AND GINI")
print("=" * 65)

def entropy(counts):
    total = 0
    for c in counts:
        total += c
    if total == 0:
        return 0.0
    result = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            result -= p * math.log2(p)
    return result

def gini(counts):
    total = 0
    for c in counts:
        total += c
    if total == 0:
        return 0.0
    result = 1.0
    for c in counts:
        result -= (c / total) ** 2
    return result

def class_counts(rows):
    yes = len([r for r in rows if r["play"] == "yes"])
    return [yes, len(rows) - yes]

parent_counts = class_counts(weather)
parent_entropy = entropy(parent_counts)
parent_gini = gini(parent_counts)
print("Whole dataset: yes =", parent_counts[0], ", no =", parent_counts[1])
print("Entropy(D) =", round(parent_entropy, 4), "| Gini(D) =", round(parent_gini, 4))

def evaluate_split(groups):
    """groups = list of row lists. Returns (info gain, gini gain, weighted entropy, weighted gini)."""
    n_total = 0
    for g in groups:
        n_total += len(g)
    w_entropy = 0.0
    w_gini = 0.0
    for g in groups:
        w_entropy += len(g) / n_total * entropy(class_counts(g))
        w_gini += len(g) / n_total * gini(class_counts(g))
    return parent_entropy - w_entropy, parent_gini - w_gini, w_entropy, w_gini

results = []      # (attribute, split description, info gain, gini gain)

# -------------------- CATEGORICAL ATTRIBUTES ----------------
for attribute in ["outlook", "windy"]:
    values = my_sort(my_unique([r[attribute] for r in weather]))
    groups = [[r for r in weather if r[attribute] == v] for v in values]
    ig, gg, we, wg = evaluate_split(groups)
    print("\n" + attribute.upper(), "(categorical)")
    for v, g in zip(values, groups):
        c = class_counts(g)
        print("   ", v.ljust(9), "yes =", c[0], "no =", c[1],
              "| entropy =", round(entropy(c), 4), "| gini =", round(gini(c), 4))
    print("    weighted entropy =", round(we, 4), "-> Information Gain =", round(ig, 4))
    print("    weighted gini    =", round(wg, 4), "-> Gini gain        =", round(gg, 4))
    results.append((attribute, "multiway split", ig, gg, wg))

# -------------------- NUMERIC ATTRIBUTES --------------------
working_attribute = "humidity"         # attribute whose full table is printed
best_numeric = {}
for attribute in ["temperature", "humidity"]:
    distinct = my_sort(my_unique([r[attribute] for r in weather]))
    thresholds = []
    for i in range(len(distinct) - 1):
        thresholds.append((distinct[i] + distinct[i + 1]) / 2)
    show = (attribute == working_attribute)
    if show:
        print("\nFULL WORKING TABLE FOR", attribute.upper(),
              "(split: <= threshold | > threshold)")
        print("thr   | <=:yes no | >:yes no | H(left) H(right) | wH     IG     |"
              " wGini  GiniGain")
    best = None
    for t in thresholds:
        left = [r for r in weather if r[attribute] <= t]
        right = [r for r in weather if r[attribute] > t]
        ig, gg, we, wg = evaluate_split([left, right])
        if show:
            lc = class_counts(left)
            rc = class_counts(right)
            print(str(t).ljust(5), "|", str(lc[0]).rjust(5), str(lc[1]).rjust(2), "|",
                  str(rc[0]).rjust(4), str(rc[1]).rjust(2), "|",
                  str(round(entropy(lc), 3)).rjust(6), str(round(entropy(rc), 3)).rjust(7), "|",
                  str(round(we, 4)).rjust(6), str(round(ig, 4)).rjust(7), "|",
                  str(round(wg, 4)).rjust(6), str(round(gg, 4)).rjust(7))
        if best is None or ig > best[1]:
            best = (t, ig, gg, wg)
    # Best threshold by Gini (lowest weighted Gini) may differ from best by IG
    best_gini = None
    for t in thresholds:
        left = [r for r in weather if r[attribute] <= t]
        right = [r for r in weather if r[attribute] > t]
        ig, gg, we, wg = evaluate_split([left, right])
        if best_gini is None or gg > best_gini[2]:
            best_gini = (t, ig, gg, wg)
    best_numeric[attribute] = (best, best_gini)
    results.append((attribute, "<= " + str(best[0]), best[1], best_gini[2], best_gini[3]))
    print("\n" + attribute.upper(), "(numeric): best threshold by IG   = <=", best[0],
          "(IG =", round(best[1], 4), ")")
    print(" " * (len(attribute) + 10), "best threshold by Gini = <=", best_gini[0],
          "(Gini gain =", round(best_gini[2], 4), ")")

# -------------------- SUMMARY TABLE -------------------------
print("\n")
print("SUMMARY OF ALL ATTRIBUTES")
print("-" * 65)
print("Attribute".ljust(13), "Split".ljust(16), "Info Gain".rjust(10),
      "Gini gain".rjust(10), "Rank(IG)".rjust(9), "Rank(Gini)".rjust(11))
by_ig = my_sort(results, key=lambda r: -r[2])
by_gini = my_sort(results, key=lambda r: -r[3])
for r in results:
    print(r[0].ljust(13), r[1].ljust(16), str(round(r[2], 4)).rjust(10),
          str(round(r[3], 4)).rjust(10), str(by_ig.index(r) + 1).rjust(9),
          str(by_gini.index(r) + 1).rjust(11))

print("\nSplitting attribute by Information Gain:", by_ig[0][0].upper(),
      "(IG =", round(by_ig[0][2], 4), ")")
print("Splitting attribute by Gini index      :", by_gini[0][0].upper(),
      "(Gini gain =", round(by_gini[0][3], 4), ")")
if by_ig[0][0] == by_gini[0][0]:
    print("Information Gain and Gini AGREE on the splitting attribute.")
else:
    print("Information Gain and Gini DISAGREE on the splitting attribute.")
order_ig = [r[0] for r in by_ig]
order_gini = [r[0] for r in by_gini]
print("Full ranking by IG  :", order_ig)
print("Full ranking by Gini:", order_gini)
print("Rankings identical  :", order_ig == order_gini)

print("\nASSIGNMENT 12 EXECUTED SUCCESSFULLY")
