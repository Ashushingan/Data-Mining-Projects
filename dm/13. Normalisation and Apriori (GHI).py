
# ============================================================
# ASSIGNMENT 13
# Part A: Normalisation (Exp 3) - GHI
#         (Min-Max to [-1, 1], Z-score, decimal scaling)
# Part B: Apriori (Exp 7) - GHI
#         (Low / Medium / High items, Apriori, min support = 0.15)
#
# Normalisation and Apriori are implemented manually.
# No NumPy, Pandas or Scikit-learn used.
# mlxtend is used ONLY at the end to verify the Apriori result.
# ============================================================
import csv
import math
from itertools import combinations

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
# PART A: NORMALISATION - GHI
# ============================================================
print("=" * 65)
print("PART A: NORMALISATION (Economy and H_Score)")
print("=" * 65)

filename = "GHI_Report.csv"
aliases = {
    "Economy": ["economy", "economygdppercapita", "gdp"],
    "H_Score": ["hscore", "happinessscore", "score"],
}
raw = load_numeric_columns(filename, aliases)

def min_max_normalise(values, new_min=-1.0, new_max=1.0):
    """v' = (v - min) / (max - min) * (new_max - new_min) + new_min"""
    lo = my_min(values)
    hi = my_max(values)
    return [(v - lo) / (hi - lo) * (new_max - new_min) + new_min for v in values]

def z_score_normalise(values):
    """v' = (v - mean) / std   (population standard deviation)"""
    m = calculate_mean(values)
    s = calculate_std(values)
    return [(v - m) / s for v in values]

def decimal_scaling(values):
    """v' = v / 10^j, j = smallest integer such that max(|v'|) < 1"""
    biggest = my_max([my_abs(v) for v in values])
    j = 0
    while biggest / (10 ** j) >= 1:
        j += 1
    return [v / (10 ** j) for v in values], j

summary_rows = []
for name in ["Economy", "H_Score"]:
    values = [v for v in raw[name] if v is not None]
    mm = min_max_normalise(values, -1.0, 1.0)
    zs = z_score_normalise(values)
    ds, j = decimal_scaling(values)
    summary_rows.append((name, "Original", values))
    summary_rows.append((name, "Min-Max [-1, 1]", mm))
    summary_rows.append((name, "Z-score", zs))
    summary_rows.append((name, "Decimal scaling (j=" + str(j) + ")", ds))
    print("\n" + name, "- first 5 values:")
    print("  original      :", [round(v, 4) for v in values[:5]])
    print("  min-max       :", [round(v, 4) for v in mm[:5]])
    print("  z-score       :", [round(v, 4) for v in zs[:5]])
    print("  decimal (j=" + str(j) + ") :", [round(v, 4) for v in ds[:5]])

print("\nCOMPARISON OF THE THREE METHODS")
print("-" * 65)
print("Attribute".ljust(9), "Method".ljust(24), "Min".rjust(9), "Max".rjust(9),
      "Mean".rjust(9), "Std".rjust(9))
for name, method, vals in summary_rows:
    print(name.ljust(9), method.ljust(24), str(round(my_min(vals), 4)).rjust(9),
          str(round(my_max(vals), 4)).rjust(9), str(round(calculate_mean(vals), 4)).rjust(9),
          str(round(calculate_std(vals), 4)).rjust(9))

print("\nWHICH METHOD BEFORE K-MEANS?")
print("-" * 65)
print("Z-score normalisation is the better choice before K-Means because:")
print("1. K-Means uses Euclidean distance, so every attribute must have a similar")
print("   spread. Z-score gives every attribute mean 0 and std 1, so no attribute")
print("   dominates the distance.")
print("2. Min-Max to [-1, 1] also equalises the range, but a single outlier fixes")
print("   the minimum / maximum and squeezes all other values together.")
print("3. Decimal scaling only divides by a power of 10; Economy and H_Score keep")
print("   different spreads (see the Std column), so it does not make the")
print("   attributes comparable.")

# ============================================================
# PART B: APRIORI - GHI
# ============================================================
print("\n")
print("=" * 65)
print("PART B: APRIORI ON GHI (min support = 0.15)")
print("=" * 65)
MIN_SUPPORT = 0.15

print("\nDiscretising every attribute into Low / Medium / High (equal frequency)")
transactions = build_ghi_transactions(filename)
print("\nFirst 5 transactions:")
for t in transactions[:5]:
    print("  ", my_sort(t))

levels = apriori(transactions, MIN_SUPPORT)
print("\nFREQUENT ITEMSETS (min support =", MIN_SUPPORT, ")")
show_levels(levels, len(transactions))
total_frequent = 0
for l in levels:
    total_frequent += len(l)
print("\nTotal frequent itemsets:", total_frequent)

# -------------------- VERIFICATION 1: DIRECT COUNTING -------
print("\nVERIFICATION BY DIRECT COUNTING (brute force over all itemsets)")
all_items = []
for t in transactions:
    for item in t:
        all_items.append(item)
all_items = my_sort(my_unique(all_items))
brute = {}
for size in range(1, len(all_items) + 1):
    found_any = False
    for combo in combinations(all_items, size):
        c = count_support(transactions, combo)
        if c / len(transactions) >= MIN_SUPPORT - 1e-9:
            brute[frozenset(combo)] = c / len(transactions)
            found_any = True
    if not found_any:
        break
mine = {}
for level in levels:
    for itemset in level:
        mine[itemset] = level[itemset]
print("Brute-force frequent itemsets:", len(brute),
      "| my Apriori:", len(mine),
      "| identical:", brute.keys() == mine.keys())

# -------------------- VERIFICATION 2: MLXTEND ---------------
print("\nVERIFICATION WITH MLXTEND")
try:
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori as mlx_apriori
    encoder = TransactionEncoder()
    table = pd.DataFrame(encoder.fit(transactions).transform(transactions),
                         columns=encoder.columns_)
    result = mlx_apriori(table, min_support=MIN_SUPPORT, use_colnames=True)
    mlx = {}
    for _, row in result.iterrows():
        mlx[frozenset(row["itemsets"])] = row["support"]
    print("mlxtend frequent itemsets:", len(mlx), "| my Apriori:", len(mine))
    print("Same itemsets:", set(mlx.keys()) == set(mine.keys()))
    same_support = True
    for k in mine:
        if k in mlx and my_abs(mlx[k] - mine[k]) > 1e-9:
            same_support = False
    print("Same supports:", same_support)
except ImportError:
    print("mlxtend is not installed. Run:  pip install mlxtend")
    print("(The brute-force check above already confirms the result.)")

print("\nASSIGNMENT 13 EXECUTED SUCCESSFULLY")
