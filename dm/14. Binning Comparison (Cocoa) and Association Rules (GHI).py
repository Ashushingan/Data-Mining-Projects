
# ============================================================
# ASSIGNMENT 14
# Part A: Binning comparison (Exp 4) - Cocoa
#         (equal-width vs equal-frequency, smoothing by means,
#          medians and boundaries)
# Part B: Association rules (Exp 8) - GHI
#         (rules with consequent H_Score=High, ranked by lift)
#
# Binning, Apriori and rule generation are implemented manually.
# No NumPy, Pandas or Scikit-learn used.
# Matplotlib is used only for plotting.
# ============================================================
import csv
import math
from itertools import combinations
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
# -------------------- APRIORI (IMPLEMENTED MANUALLY) --------
def apriori(transactions, min_support):
    """
    Level-wise Apriori.
    transactions : list of sets of items
    Returns a list of dictionaries: levels[0] = L1, levels[1] = L2, ...
    Each dictionary maps frozenset(itemset) -> support (fraction).
    """
    n = len(transactions)
    min_count = min_support * n - 1e-9

    # ---- L1: count single items ----
    counts = {}
    for t in transactions:
        for item in t:
            key = frozenset([item])
            counts[key] = counts.get(key, 0) + 1
    current = {}
    for key in counts:
        if counts[key] >= min_count:
            current[key] = counts[key] / n

    levels = []
    k = 2
    while len(current) > 0:
        levels.append(current)

        # ---- join step: merge (k-1)-itemsets sharing k-2 items ----
        previous = my_sort([tuple(my_sort(s)) for s in current])
        candidates = []
        for i in range(len(previous)):
            for j in range(i + 1, len(previous)):
                a = previous[i]
                b = previous[j]
                if a[:k - 2] != b[:k - 2]:
                    break
                candidate = frozenset(a) | frozenset(b)
                # ---- prune step: every (k-1)-subset must be frequent ----
                ok = True
                for item in candidate:
                    if (candidate - frozenset([item])) not in current:
                        ok = False
                        break
                if ok:
                    candidates.append(candidate)

        # ---- count the candidates in the transactions ----
        cand_counts = {}
        for c in candidates:
            cand_counts[c] = 0
        for t in transactions:
            for c in candidates:
                if c <= t:
                    cand_counts[c] += 1
        current = {}
        for c in cand_counts:
            if cand_counts[c] >= min_count:
                current[c] = cand_counts[c] / n
        k += 1
    return levels

def count_support(transactions, itemset):
    """Direct counting of one itemset (used for verification)."""
    count = 0
    for t in transactions:
        if set(itemset) <= t:
            count += 1
    return count

def show_levels(levels, n_transactions):
    for i in range(len(levels)):
        print("\nL" + str(i + 1), "- frequent itemsets of size", i + 1,
              "(" + str(len(levels[i])) + " found)")
        ordered = my_sort(levels[i].items(), key=lambda kv: (-kv[1], my_sort(kv[0])))
        for itemset, support in ordered:
            print("  ", my_sort(itemset), "support =", round(support, 4),
                  "(count =", round(support * n_transactions), ")")

# -------------------- GHI TRANSACTIONS (ASSIGNMENT 13 PART B) --
# Each country becomes one transaction of 5 items, e.g. "Economy=High".
def equal_frequency_cutoffs(values):
    """Upper limits of the Low and Medium groups (3 equal-frequency groups)."""
    s = my_sort(values)
    n = len(s)
    low_limit = s[my_ceil(n / 3) - 1]
    medium_limit = s[my_ceil(2 * n / 3) - 1]
    return low_limit, medium_limit

def build_ghi_transactions(filename):
    wanted = {
        "H_Score": ["hscore", "happinessscore", "score"],
        "Economy": ["economy", "economygdppercapita", "gdp"],
        "Fam":     ["fam", "family", "socialsupport"],
        "Health":  ["health", "healthlifeexpectancy", "lifeexpectancy"],
        "Freedom": ["freedom"],
    }
    raw = load_numeric_columns(filename, wanted)
    attributes = ["H_Score", "Economy", "Fam", "Health", "Freedom"]
    rows = []
    for i in range(len(raw["H_Score"])):
        complete = True
        for a in attributes:
            if raw[a][i] is None:
                complete = False
        if complete:
            rows.append({a: raw[a][i] for a in attributes})
    print("Countries (transactions):", len(rows))
    cutoffs = {}
    for a in attributes:
        cutoffs[a] = equal_frequency_cutoffs([r[a] for r in rows])
        print("  ", a.ljust(8), "Low <=", round(cutoffs[a][0], 4),
              "| Medium <=", round(cutoffs[a][1], 4), "| High above")
    transactions = []
    for r in rows:
        t = set()
        for a in attributes:
            if r[a] <= cutoffs[a][0]:
                level = "Low"
            elif r[a] <= cutoffs[a][1]:
                level = "Medium"
            else:
                level = "High"
            t.add(a + "=" + level)
        transactions.append(t)
    return transactions

# ============================================================
# PART A: BINNING COMPARISON - COCOA (Rating)
# ============================================================
print("=" * 65)
print("PART A: BINNING OF RATING (COCOA)")
print("=" * 65)

cocoa_file = "Cocoa.csv"
raw = load_numeric_columns(cocoa_file, {"Rating": ["rating"]})
ratings = [v for v in raw["Rating"] if v is not None]
n = len(ratings)
sorted_ratings = my_sort(ratings)
print("Records with a Rating:", n)
print("Minimum =", sorted_ratings[0], "| Maximum =", sorted_ratings[-1],
      "| Mean =", round(calculate_mean(ratings), 4),
      "| Std =", round(calculate_std(ratings), 4))

NUM_BINS = 3

# -------------------- EQUAL-WIDTH BINNING -------------------
def equal_width_bins(sorted_values, k):
    lo = sorted_values[0]
    hi = sorted_values[-1]
    width = (hi - lo) / k
    bins = [[] for _ in range(k)]
    for v in sorted_values:
        index = int((v - lo) / width)
        if index >= k:                  # the maximum falls in the last bin
            index = k - 1
        bins[index].append(v)
    return bins, width

# -------------------- EQUAL-FREQUENCY BINNING ---------------
def equal_frequency_bins(sorted_values, k):
    """Same number of records in each bin (extra records go to the first bins)."""
    size = len(sorted_values) // k
    extra = len(sorted_values) % k
    bins = []
    start = 0
    for b in range(k):
        length = size + (1 if b < extra else 0)
        bins.append(sorted_values[start:start + length])
        start += length
    return bins

width_bins, width = equal_width_bins(sorted_ratings, NUM_BINS)
freq_bins = equal_frequency_bins(sorted_ratings, NUM_BINS)

# -------------------- SMOOTHING FUNCTIONS -------------------
def smooth_by_means(bins):
    result = []
    for b in bins:
        m = calculate_mean(b) if len(b) > 0 else 0
        result.append([m] * len(b))
    return result

def smooth_by_medians(bins):
    result = []
    for b in bins:
        med = find_median(b) if len(b) > 0 else 0       # bins are already sorted
        result.append([med] * len(b))
    return result

def smooth_by_boundaries(bins):
    result = []
    for b in bins:
        lo = b[0]
        hi = b[-1]
        result.append([lo if (v - lo) <= (hi - v) else hi for v in b])
    return result

def sse(original_bins, smoothed_bins):
    """Total squared error introduced by smoothing."""
    total = 0.0
    for ob, sb in zip(original_bins, smoothed_bins):
        for a, b in zip(ob, sb):
            total += (a - b) ** 2
    return total

# -------------------- PRINT THE BINS ------------------------
print("\nEQUAL-WIDTH BINS (width =", round(width, 4), ")")
print("bin | range                | count | mean   | median | boundaries (min, max)")
for i, b in enumerate(width_bins):
    lo = sorted_ratings[0] + i * width
    hi = lo + width
    if len(b) > 0:
        print(" " + str(i + 1), " | [" + str(round(lo, 3)) + ", " + str(round(hi, 3)) +
              (")" if i < NUM_BINS - 1 else "]"), "|", str(len(b)).rjust(5), "|",
              str(round(calculate_mean(b), 3)).ljust(6), "|",
              str(find_median(b)).ljust(6), "|", (b[0], b[-1]))
    else:
        print(" " + str(i + 1), " | [" + str(round(lo, 3)) + ", " + str(round(hi, 3)) +
              ") |     0 | empty bin")

print("\nEQUAL-FREQUENCY BINS")
print("bin | range (min - max)    | count | mean   | median | boundaries (min, max)")
for i, b in enumerate(freq_bins):
    print(" " + str(i + 1), " | " + (str(b[0]) + " - " + str(b[-1])).ljust(20), "|",
          str(len(b)).rjust(5), "|", str(round(calculate_mean(b), 3)).ljust(6), "|",
          str(find_median(b)).ljust(6), "|", (b[0], b[-1]))

print("\nBIN COUNT COMPARISON")
print("Equal-width     :", [len(b) for b in width_bins])
print("Equal-frequency :", [len(b) for b in freq_bins])

# -------------------- SMOOTHING -----------------------------
methods = [("bin means", smooth_by_means), ("bin medians", smooth_by_medians),
           ("bin boundaries", smooth_by_boundaries)]
results = {}
print("\nSMOOTHING - first 12 sorted ratings and their smoothed values")
print("original      :", sorted_ratings[:12])
for bin_name, bins in [("Equal-width", width_bins), ("Equal-freq", freq_bins)]:
    flat_smoothed = {}
    for method_name, method in methods:
        smoothed = method(bins)
        flat = []
        for sb in smoothed:
            flat += sb
        flat_smoothed[method_name] = flat
        results[(bin_name, method_name)] = sse(bins, smoothed)
        print((bin_name + " / " + method_name).ljust(32), [round(x, 2) for x in flat[:12]])

print("\nSUM OF SQUARED ERROR INTRODUCED BY SMOOTHING (smaller = less distortion)")
print("Binning".ljust(14), "means".rjust(10), "medians".rjust(10), "boundaries".rjust(12))
for bin_name in ["Equal-width", "Equal-freq"]:
    print(bin_name.ljust(14),
          str(round(results[(bin_name, "bin means")], 3)).rjust(10),
          str(round(results[(bin_name, "bin medians")], 3)).rjust(10),
          str(round(results[(bin_name, "bin boundaries")], 3)).rjust(12))

# -------------------- WHICH METHOD SUITS? -------------------
print("\nWHICH METHOD SUITS THIS TIGHTLY CLUSTERED ATTRIBUTE?")
print("-" * 65)
width_counts = [len(b) for b in width_bins]
biggest = my_max(width_counts)
share_big = 100.0 * biggest / n
print("Equal-width: the fullest bin holds", biggest, "of", n, "records (" +
      str(round(share_big, 1)) + " %); the emptiest holds", my_min(width_counts), ".")
print("Equal-frequency: every bin holds about", n // NUM_BINS, "records.")
if share_big > 50:
    print("Ratings are tightly clustered, so equal-width bins are very unbalanced: one")
    print("wide bin swallows most records and the outer bins are almost empty, so")
    print("smoothing tells us very little. Equal-frequency binning splits the dense")
    print("middle into several bins, keeps the bin sizes equal and is the better choice.")
else:
    print("Equal-width bins are reasonably balanced here; equal-frequency binning is")
    print("still preferred because every bin is guaranteed to hold the same number of")
    print("records, whatever the shape of the distribution.")
print("(Note: equal-frequency can place identical ratings in two bins, because the")
print("ratings are discrete with many ties.)")

# -------------------- PLOT BIN COUNTS -----------------------
plt.figure(figsize=(9, 5))
positions = [0, 1, 2]
plt.bar([p - 0.2 for p in positions], [len(b) for b in width_bins], width=0.4,
        label="Equal-width")
plt.bar([p + 0.2 for p in positions], [len(b) for b in freq_bins], width=0.4,
        label="Equal-frequency")
plt.xticks(positions, ["Bin 1", "Bin 2", "Bin 3"])
plt.title("Number of Ratings in each bin")
plt.ylabel("Count")
plt.legend()
plt.grid(axis="y")
plt.show()

# ============================================================
# PART B: ASSOCIATION RULES - GHI
# ============================================================
print("\n")
print("=" * 65)
print("PART B: ASSOCIATION RULES (GHI), consequent = H_Score=High")
print("=" * 65)
MIN_SUPPORT = 0.1
MIN_CONFIDENCE = 0.6
TARGET = frozenset(["H_Score=High"])

print("\nBuilding the transactions of Assignment 13, Part B")
transactions = build_ghi_transactions("GHI_Report.csv")
n_tx = len(transactions)

levels = apriori(transactions, MIN_SUPPORT)
support = {}
for level in levels:
    for itemset in level:
        support[itemset] = level[itemset]
print("Frequent itemsets at min support", MIN_SUPPORT, ":", len(support))

# -------------------- RULE GENERATION -----------------------
rules = []
for itemset in support:
    if len(itemset) < 2:
        continue
    items = my_sort(itemset)
    for r in range(1, len(items)):
        for antecedent in combinations(items, r):
            antecedent = frozenset(antecedent)
            consequent = itemset - antecedent
            confidence = support[itemset] / support[antecedent]
            if confidence >= MIN_CONFIDENCE - 1e-12:
                lift = confidence / support[consequent]
                rules.append({"antecedent": antecedent, "consequent": consequent,
                              "support": support[itemset], "confidence": confidence,
                              "lift": lift})
print("Rules with confidence >=", MIN_CONFIDENCE, ":", len(rules))

target_rules = [r for r in rules if r["consequent"] == TARGET]
target_rules = my_sort(target_rules, key=lambda r: (-r["lift"], -r["confidence"], -r["support"]))
print("Rules whose consequent is H_Score=High:", len(target_rules))

print("\nRULES RANKED BY LIFT")
print("rank | rule                                              | support | conf  | lift")
for rank, r in enumerate(target_rules, 1):
    text = " AND ".join(my_sort(r["antecedent"])) + " => H_Score=High"
    print(str(rank).rjust(4), "|", text.ljust(49), "|", str(round(r["support"], 3)).rjust(7),
          "|", str(round(r["confidence"], 3)).rjust(5), "|", round(r["lift"], 3))

# -------------------- INTERPRET THE TOP 3 -------------------
base = support[TARGET] if TARGET in support else 0
print("\nINTERPRETATION OF THE TOP 3 RULES")
print("(Baseline: P(H_Score=High) =", round(base, 3), ")")
for rank, r in enumerate(target_rules[:3], 1):
    text = " AND ".join(my_sort(r["antecedent"]))
    print("\nRule", rank, ":", text, "=> H_Score=High")
    print("  Support    =", round(r["support"], 3), ": about", round(r["support"] * 100, 1),
          "% of all countries have all these items together.")
    print("  Confidence =", round(r["confidence"], 3), ": among countries with", text +
          ", " + str(round(r["confidence"] * 100, 1)), "% are High on H_Score.")
    print("  Lift       =", round(r["lift"], 3), ": such countries are", round(r["lift"], 2),
          "times more likely to have High H_Score than a random country.")
    if r["lift"] > 1:
        print("  Lift > 1 -> positive association (not just a chance co-occurrence).")
if len(target_rules) == 0:
    print("No rule with consequent H_Score=High reached the thresholds.")

print("\nASSIGNMENT 14 EXECUTED SUCCESSFULLY")
