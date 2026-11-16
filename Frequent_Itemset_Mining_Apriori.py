"""
Data Mining Lab - Experiment 7: Frequent Itemset Mining (Apriori)
Support counts and candidate/frequent itemsets are computed manually.
No statistical library or itertools is used.
"""
transactions = [
    {"Bread", "Milk", "Butter"}, {"Bread", "Milk"}, {"Bread", "Butter"},
    {"Milk", "Butter", "Eggs"}, {"Bread", "Milk", "Butter"}, {"Milk", "Eggs"},
    {"Bread", "Milk", "Eggs"}, {"Bread", "Butter"}
]
minimum_support = 0.25
transaction_count = len(transactions)
items = []
for transaction in transactions:
    for item in transaction:
        if item not in items:
            items.append(item)
items.sort()

def get_support_count(itemset):
    count = 0
    for transaction in transactions:
        present = True
        for item in itemset:
            if item not in transaction:
                present = False
                break
        if present:
            count += 1
    return count

def all_subsets_frequent(candidate, previous_frequent):
    for item in candidate:
        subset = set()
        for other in candidate:
            if other != item:
                subset.add(other)
        if frozenset(subset) not in previous_frequent:
            return False
    return True

current_candidates = [frozenset([item]) for item in items]
all_frequent = []
level = 1
while current_candidates:
    print("\nCANDIDATE", level, "- ITEMSETS")
    for candidate in current_candidates:
        print(sorted(candidate), "count =", get_support_count(candidate))
    current_frequent = []
    print("FREQUENT", level, "- ITEMSETS")
    for candidate in current_candidates:
        count = get_support_count(candidate)
        support = count / transaction_count
        if support >= minimum_support:
            current_frequent.append(candidate)
            all_frequent.append(candidate)
            print(sorted(candidate), "count =", count, "support =", round(support, 3))
    if not current_frequent:
        break
    next_candidates = []
    for i in range(len(current_frequent)):
        for j in range(i + 1, len(current_frequent)):
            joined = set(current_frequent[i]) | set(current_frequent[j])
            if len(joined) == level + 1:
                candidate = frozenset(joined)
                if candidate not in next_candidates and all_subsets_frequent(candidate, current_frequent):
                    next_candidates.append(candidate)
    current_candidates = next_candidates
    level += 1

print("\nFINAL FREQUENT ITEMSETS")
for itemset in all_frequent:
    count = get_support_count(itemset)
    print(sorted(itemset), "support count =", count,
          "support =", round(count / transaction_count, 3))
