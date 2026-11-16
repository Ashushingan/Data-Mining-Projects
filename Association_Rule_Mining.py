"""
Data Mining Lab - Experiment 8: Association Rule Mining
Support, confidence and lift are calculated manually.
"""

transactions = [
    {"Bread", "Milk", "Butter"}, {"Bread", "Milk"}, {"Bread", "Butter"},
    {"Milk", "Butter", "Eggs"}, {"Bread", "Milk", "Butter"}, {"Milk", "Eggs"},
    {"Bread", "Milk", "Eggs"}, {"Bread", "Butter"}
]
minimum_support = 0.25
minimum_confidence = 0.60
transaction_count = len(transactions)

def support_count(itemset):
    count = 0
    for transaction in transactions:
        contains_all = True
        for item in itemset:
            if item not in transaction:
                contains_all = False
                break
        if contains_all:
            count += 1
    return count

def frequent_itemsets():
    items = []
    for transaction in transactions:
        for item in transaction:
            if item not in items:
                items.append(item)
    items.sort()
    candidates = [frozenset([item]) for item in items]
    all_frequent = []
    level = 1
    while candidates:
        frequent = []
        for candidate in candidates:
            if support_count(candidate) / transaction_count >= minimum_support:
                frequent.append(candidate)
                all_frequent.append(candidate)
        next_candidates = []
        for i in range(len(frequent)):
            for j in range(i + 1, len(frequent)):
                joined = set(frequent[i]) | set(frequent[j])
                if len(joined) == level + 1:
                    candidate = frozenset(joined)
                    if candidate not in next_candidates:
                        valid = True
                        for item in candidate:
                            subset = frozenset(x for x in candidate if x != item)
                            if subset not in frequent:
                                valid = False
                        if valid:
                            next_candidates.append(candidate)
        candidates = next_candidates
        level += 1
    return all_frequent

def proper_nonempty_subsets(itemset):
    items = sorted(itemset)
    subsets = []
    mask = 1
    while mask < (2 ** len(items)) - 1:
        subset = set()
        for index in range(len(items)):
            if mask & (1 << index):
                subset.add(items[index])
        subsets.append(frozenset(subset))
        mask += 1
    return subsets

frequent = frequent_itemsets()
print("MINIMUM SUPPORT:", minimum_support)
print("MINIMUM CONFIDENCE:", minimum_confidence)
print("\nASSOCIATION RULES")
rule_count = 0
for itemset in frequent:
    if len(itemset) < 2:
        continue
    for antecedent in proper_nonempty_subsets(itemset):
        consequent = frozenset(item for item in itemset if item not in antecedent)
        union_count = support_count(itemset)
        antecedent_count = support_count(antecedent)
        consequent_count = support_count(consequent)
        support = union_count / transaction_count
        confidence = union_count / antecedent_count if antecedent_count else 0
        consequent_support = consequent_count / transaction_count
        lift = confidence / consequent_support if consequent_support else 0
        if confidence >= minimum_confidence:
            rule_count += 1
            left = " + ".join(sorted(antecedent))
            right = " + ".join(sorted(consequent))
            print(f"{left} -> {right} | support={support:.3f} "
                  f"confidence={confidence:.3f} lift={lift:.3f}")
print("\nRules meeting minimum confidence:", rule_count)
print("Interpretation: lift > 1 suggests positive association; lift = 1 suggests independence.")
