
# ============================================================
# ASSIGNMENT 17
# Part A: Data preprocessing (Exp 5) - Cocoa
#         (clean Bean Origin, missing Review Dates, Cocoa Percent
#          smoothing, Min-Max Rating, drop useless attributes)
# Part B: Apriori (Exp 7) - Cocoa
#         (transactions of Assignment 3 Part B, support 0.02/0.05/0.10)
#
# All cleaning, smoothing, normalisation and Apriori are implemented
# manually. No NumPy, Pandas or Scikit-learn used.
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

cocoa_file = "Cocoa.csv"

# ============================================================
# PART A: DATA PREPROCESSING - COCOA
# ============================================================
print("=" * 65)
print("PART A: DATA PREPROCESSING (COCOA)")
print("=" * 65)

headers, rows = load_csv(cocoa_file)
print("Records read:", len(rows), "| Attributes:", len(headers))
i_origin = find_column(headers, ["beanorigin", "specificbeanorigin", "origin"])
i_date = find_column(headers, ["reviewdate", "year", "reviewyear"])
i_cocoa = find_column(headers, ["cocoapercent", "cocoa", "percent"])
i_rating = find_column(headers, ["rating"])
print("Bean Origin column:", headers[i_origin], "| Review Date:", headers[i_date],
      "| Cocoa Percent:", headers[i_cocoa], "| Rating:", headers[i_rating])

def top_values(values, count=10):
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    ordered = my_sort(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:count], len(freq)

# -------------------- STEP 1: CLEAN BEAN ORIGIN -------------
print("\nSTEP 1: Clean the Bean Origin column")
print("-" * 65)
before = [r[i_origin] for r in rows]              # exactly as stored
top_before, unique_before = top_values(before)

def clean_origin(text):
    text = " ".join(text.split())                 # trim + collapse inner spaces
    text = text.title()                           # standardise case
    if text == "":
        text = "Unknown"
    return text

for r in rows:
    r[i_origin] = clean_origin(r[i_origin])
after = [r[i_origin] for r in rows]
top_after, unique_after = top_values(after)

print("Distinct values BEFORE cleaning:", unique_before)
print("10 most frequent values BEFORE (shown in quotes to expose spaces):")
for value, count in top_before:
    print("   ", repr(value).ljust(34), count)
print("\nDistinct values AFTER cleaning :", unique_after)
print("10 most frequent values AFTER:")
for value, count in top_after:
    print("   ", repr(value).ljust(34), count)

# -------------------- STEP 2: MISSING REVIEW DATES ----------
print("\nSTEP 2: Handle the missing Review Dates")
print("-" * 65)
dates = []
for r in rows:
    v = to_float(r[i_date])
    dates.append(None if v is None else int(v))
valid_dates = [d for d in dates if d is not None]
missing_dates = len(dates) - len(valid_dates)
freq = {}
for d in valid_dates:
    freq[d] = freq.get(d, 0) + 1
mode_year = my_max(list(freq), key=lambda y: (freq[y], y))
print("Missing Review Dates:", missing_dates, "of", len(dates),
      "(" + str(round(100.0 * missing_dates / len(dates), 2)) + " %)")
print("Most frequent year (mode):", mode_year, "(" + str(freq[mode_year]), "reviews)")
for i in range(len(rows)):
    if dates[i] is None:
        dates[i] = mode_year
        rows[i][i_date] = str(mode_year)
print("Each missing date was filled with the mode year", mode_year, ".")
print("Reason: a review year is a categorical-like label, so the mean would give an")
print("invented year, while the mode keeps the value that already occurs most often.")

# -------------------- STEP 3: COCOA PERCENT -> NUMBER -------
print("\nSTEP 3: Convert Cocoa Percent to a number")
print("-" * 65)
cocoa = [to_float(r[i_cocoa]) for r in rows]
valid = [c for c in cocoa if c is not None]
if my_max(valid) <= 1.0:                      # 0.70 -> 70
    cocoa = [None if c is None else c * 100 for c in cocoa]
    valid = [c for c in cocoa if c is not None]
cocoa_median = find_median(my_sort(valid))
missing_cocoa = len(cocoa) - len(valid)
for i in range(len(cocoa)):
    if cocoa[i] is None:
        cocoa[i] = cocoa_median
print("Examples (text -> number):", [(rows[i][i_cocoa], cocoa[i]) for i in range(5)])
print("Cocoa % values that were missing (set to the median", str(cocoa_median) + "):",
      missing_cocoa)
print("Cocoa %: min =", my_min(cocoa), "| max =", my_max(cocoa),
      "| mean =", round(calculate_mean(cocoa), 3))

# Rows without a Rating cannot be used for mining, so they are removed.
keep = [i for i in range(len(rows)) if to_float(rows[i][i_rating]) is not None]
print("\nRecords without a Rating removed:", len(rows) - len(keep))
rows = [rows[i] for i in keep]
cocoa = [cocoa[i] for i in keep]
ratings = [to_float(r[i_rating]) for r in rows]
n = len(rows)

# -------------------- STEP 4: SMOOTH COCOA % BY BIN MEANS ---
print("\nSTEP 4: Smooth Cocoa % by bin means (5 equal-frequency bins)")
print("-" * 65)
NUM_BINS = 5
order = my_sort(list(range(n)), key=lambda i: cocoa[i])          # record numbers, sorted by cocoa
size = n // NUM_BINS
extra = n % NUM_BINS
smoothed = [0.0] * n
start = 0
print("bin | records | range           | bin mean (smoothed value)")
for b in range(NUM_BINS):
    length = size + (1 if b < extra else 0)
    members = order[start:start + length]
    values = [cocoa[i] for i in members]
    bin_mean = calculate_mean(values)
    for i in members:
        smoothed[i] = bin_mean
    print(" " + str(b + 1), " |", str(length).rjust(7), "|",
          (str(my_min(values)) + " - " + str(my_max(values))).ljust(15), "|", round(bin_mean, 3))
    start += length
print("\nFirst 8 records: original -> smoothed Cocoa %")
for i in range(8):
    print("   ", cocoa[i], "->", round(smoothed[i], 3))

# -------------------- STEP 5: MIN-MAX NORMALISE RATING ------
print("\nSTEP 5: Min-Max normalise Rating to [0, 1]")
print("-" * 65)
r_min = my_min(ratings)
r_max = my_max(ratings)
rating_norm = [(r - r_min) / (r_max - r_min) for r in ratings]
print("Rating min =", r_min, "| max =", r_max, "| formula: (r - min) / (max - min)")
print("Examples (rating -> normalised):", [(ratings[i], round(rating_norm[i], 3)) for i in range(6)])

# -------------------- STEP 6: DROP USELESS ATTRIBUTES -------
print("\nSTEP 6: Drop attributes that are not useful for mining")
print("-" * 65)
ID_NAMES = ("ref", "id", "index", "srno", "sno", "serialno")
keep_columns = []
print("Attribute".ljust(26), "Decision".ljust(9), "Justification")
for j in range(len(headers)):
    cells = [r[j] for r in rows]
    present = [c for c in cells if c.strip().lower() not in MISSING_TEXT]
    missing_share = 1.0 - len(present) / len(cells)
    unique = len(my_unique(present))
    numeric = len([c for c in present if to_float(c) is not None])
    name = norm_name(headers[j])
    decision = "KEEP"
    reason = "useful: informative attribute for mining"
    if j == i_cocoa:
        decision = "REPLACE"
        reason = "text like '70%' replaced by numeric Cocoa_Percent_Num (+ smoothed version)"
    elif j in (i_origin, i_date, i_rating):
        reason = "needed: cleaned / completed / normalised above"
    elif name in ID_NAMES:
        decision = "DROP"
        reason = "record / batch identifier - no pattern to mine (Review Date already gives time)"
    elif unique <= 1:
        decision = "DROP"
        reason = "constant - the same value in every record carries no information"
    elif missing_share > 0.40:
        decision = "DROP"
        reason = ("mostly empty (" + str(round(missing_share * 100)) +
                  " % missing) - imputing it would invent data")
    elif unique > 0.9 * len(present) and numeric < 0.5 * len(present):
        decision = "DROP"
        reason = "almost unique text per record - behaves like an ID, no generalisable pattern"
    if decision in ("KEEP",) or j in (i_origin, i_date, i_rating):
        keep_columns.append(j)
    print(headers[j].ljust(26), decision.ljust(9), reason)

# -------------------- SAVE THE CLEANED FILE -----------------
out_headers = [headers[j] for j in keep_columns] + \
              ["Cocoa_Percent_Num", "Cocoa_Percent_Smoothed", "Rating_Norm"]
with open("Cocoa_cleaned.csv", "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(out_headers)
    for i in range(n):
        writer.writerow([rows[i][j] for j in keep_columns] +
                        [cocoa[i], round(smoothed[i], 4), round(rating_norm[i], 4)])
print("\nCleaned dataset saved as Cocoa_cleaned.csv with", n, "records and",
      len(out_headers), "attributes:")
print(out_headers)

# ============================================================
# PART B: APRIORI - COCOA (TRANSACTIONS OF ASSIGNMENT 3, PART B)
# ============================================================
print("\n")
print("=" * 65)
print("PART B: APRIORI ON COCOA AT MIN SUPPORT 0.02, 0.05 AND 0.10")
print("=" * 65)

def rating_class(rating):
    if rating < 3.0:
        return "Low"
    elif rating <= 3.5:
        return "Medium"
    return "High"

def review_period(year):
    if year <= 2009:
        return "2006-09"
    elif year <= 2013:
        return "2010-13"
    return "2014-17"

def cocoa_band(percent):
    if percent < 70:
        return "<70"
    elif percent == 70:
        return "70"
    return ">70"

# One transaction per review with 4 items:
# Company Location (top 8, the rest "Other"), Cocoa % band, Rating class, Review period.
headers2, rows2 = load_csv(cocoa_file)
j_loc = find_column(headers2, ["companylocation", "location"])
j_date = find_column(headers2, ["reviewdate", "year", "reviewyear"])
j_cocoa = find_column(headers2, ["cocoapercent", "cocoa", "percent"])
j_rating = find_column(headers2, ["rating"])

reviews = []
for r in rows2:
    loc = " ".join(r[j_loc].split())
    year = to_float(r[j_date])
    percent = to_float(r[j_cocoa])
    rating = to_float(r[j_rating])
    if loc == "" or year is None or percent is None or rating is None:
        continue                                       # incomplete review
    reviews.append((loc, int(year), percent, rating))
if my_max([rv[2] for rv in reviews]) <= 1.0:
    reviews = [(a, b, c * 100, d) for (a, b, c, d) in reviews]
print("Reviews with all four fields:", len(reviews), "of", len(rows2))

loc_freq = {}
for rv in reviews:
    loc_freq[rv[0]] = loc_freq.get(rv[0], 0) + 1
top8 = [loc for loc, _ in my_sort(loc_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]
print("Top 8 company locations:", top8)

transactions = []
for loc, year, percent, rating in reviews:
    transactions.append({
        "Location=" + (loc if loc in top8 else "Other"),
        "Cocoa=" + cocoa_band(percent),
        "Rating=" + rating_class(rating),
        "Period=" + review_period(year),
    })
print("Transactions:", len(transactions), "| items per transaction: 4")
print("First 3 transactions:")
for t in transactions[:3]:
    print("  ", my_sort(t))

THRESHOLDS = [0.02, 0.05, 0.10]
all_levels = {}
for s in THRESHOLDS:
    all_levels[s] = apriori(transactions, s)

max_size = 0
for s in THRESHOLDS:
    if len(all_levels[s]) > max_size:
        max_size = len(all_levels[s])
print("\nNUMBER OF FREQUENT ITEMSETS OF EACH SIZE")
print("-" * 65)
line = "Min support".ljust(13)
for size in range(1, max_size + 1):
    line += ("L" + str(size)).rjust(8)
line += "Total".rjust(9)
print(line)
totals = {}
for s in THRESHOLDS:
    line = str(s).ljust(13)
    total = 0
    for size in range(1, max_size + 1):
        cnt = len(all_levels[s][size - 1]) if size <= len(all_levels[s]) else 0
        total += cnt
        line += str(cnt).rjust(8)
    totals[s] = total
    line += str(total).rjust(9)
    print(line)

print("\nFREQUENT ITEMSETS AT THE HIGHEST THRESHOLD (0.10)")
show_levels(all_levels[0.10], len(transactions))

# -------------------- EXPLAIN THE EFFECT --------------------
print("\nEFFECT OF THE MINIMUM SUPPORT THRESHOLD")
print("-" * 65)
print("Total frequent itemsets:", totals[0.02], "(at 0.02) ->", totals[0.05], "(at 0.05) ->",
      totals[0.10], "(at 0.10).")
if totals[0.02] > 0:
    drop = 100.0 * (totals[0.02] - totals[0.10]) / totals[0.02]
    print("Raising the threshold from 0.02 to 0.10 removes", round(drop, 1), "% of the itemsets.")
longest = {}
for s in THRESHOLDS:
    longest[s] = len(all_levels[s])
print("Largest itemset size found:", longest)
print("1. Support is anti-monotone: a bigger itemset can never be more frequent than")
print("   its subsets, so every itemset that is frequent at 0.10 is also frequent at")
print("   0.05 and 0.02; lowering the threshold only ADDS itemsets.")
print("2. Large itemsets disappear first, because the more items that must occur")
print("   together, the smaller the fraction of reviews that contain all of them.")
print("3. A low threshold (0.02) finds rare, specific combinations but produces many")
print("   itemsets (more noise, more computation). A high threshold (0.10) keeps only")
print("   the strongest, most common patterns and runs fast, but it may miss")
print("   interesting rare combinations.")

print("\nASSIGNMENT 17 EXECUTED SUCCESSFULLY")
