"""
Naive Bayes Classification
Experiment 11 - Data Mining Lab

Implements a categorical Naive Bayes classifier using the Play Tennis dataset.
The program:
1. Splits the dataset into training and testing records.
2. Trains a Naive Bayes model using class priors and conditional probabilities.
3. Uses Laplace smoothing to handle unseen attribute values.
4. Classifies each test instance.
5. Evaluates the model using accuracy, precision, recall, F1-score,
   and a confusion matrix.

No machine-learning or statistical library is required.
"""

from collections import defaultdict

# ---------------------------------------------------------------------
# 1. Dataset
# Attributes: Outlook, Temperature, Humidity, Wind
# Target: Play (Yes / No)
# ---------------------------------------------------------------------

DATASET = [
    (("Sunny", "Hot", "High", "Weak"), "No"),
    (("Sunny", "Hot", "High", "Strong"), "No"),
    (("Overcast", "Hot", "High", "Weak"), "Yes"),
    (("Rain", "Mild", "High", "Weak"), "Yes"),
    (("Rain", "Cool", "Normal", "Weak"), "Yes"),
    (("Rain", "Cool", "Normal", "Strong"), "No"),
    (("Overcast", "Cool", "Normal", "Strong"), "Yes"),
    (("Sunny", "Mild", "High", "Weak"), "No"),
    (("Sunny", "Cool", "Normal", "Weak"), "Yes"),
    (("Rain", "Mild", "Normal", "Weak"), "Yes"),
    (("Sunny", "Mild", "Normal", "Strong"), "Yes"),
    (("Overcast", "Mild", "High", "Strong"), "Yes"),
    (("Overcast", "Hot", "Normal", "Weak"), "Yes"),
    (("Rain", "Mild", "High", "Strong"), "No"),
]

ATTRIBUTE_NAMES = ["Outlook", "Temperature", "Humidity", "Wind"]
CLASS_LABELS = ["No", "Yes"]

# Keep the split deterministic and reproducible.
TRAINING_DATA = DATASET[:10]
TEST_DATA = DATASET[10:]


# ---------------------------------------------------------------------
# 2. Train the categorical Naive Bayes classifier
# ---------------------------------------------------------------------

def train_naive_bayes(training_data):
    total_records = len(training_data)
    class_counts = defaultdict(int)
    # value_counts[class][attribute_index][attribute_value] = count
    value_counts = defaultdict(
        lambda: [defaultdict(int) for _ in ATTRIBUTE_NAMES]
    )
    attribute_values = [set() for _ in ATTRIBUTE_NAMES]

    for features, label in training_data:
        class_counts[label] += 1

        for index, value in enumerate(features):
            value_counts[label][index][value] += 1
            attribute_values[index].add(value)

    # Calculate prior probability P(class).
    priors = {
        label: class_counts[label] / total_records
        for label in class_counts
    }

    return class_counts, value_counts, attribute_values, priors


# ---------------------------------------------------------------------
# 3. Predict a class using Bayes' theorem
# ---------------------------------------------------------------------

def predict(features, model):
    class_counts, value_counts, attribute_values, priors = model
    total_records = sum(class_counts.values())
    scores = {}

    for label in class_counts:
        # Start with log(P(class)); logarithms avoid very small products.
        score = log_probability(priors[label])

        for index, value in enumerate(features):
            # Laplace smoothing:
            # (count(value, class) + 1) / (count(class) + number_of_values)
            possible_values = len(attribute_values[index])
            numerator = value_counts[label][index][value] + 1
            denominator = class_counts[label] + possible_values
            conditional_probability = numerator / denominator

            score += log_probability(conditional_probability)

        scores[label] = score

    return max(scores, key=scores.get)


def log_probability(probability):
    """Return natural logarithm without requiring an external library."""
    import math
    return math.log(probability)


# ---------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------

def calculate_metrics(actual, predicted, positive_label="Yes"):
    labels = sorted(set(actual) | set(predicted))

    confusion = {
        actual_label: {predicted_label: 0 for predicted_label in labels}
        for actual_label in labels
    }

    for actual_label, predicted_label in zip(actual, predicted):
        confusion[actual_label][predicted_label] += 1

    correct = sum(
        1 for actual_label, predicted_label in zip(actual, predicted)
        if actual_label == predicted_label
    )
    accuracy = correct / len(actual) if actual else 0.0

    true_positive = confusion.get(positive_label, {}).get(positive_label, 0)
    false_positive = sum(
        confusion[label].get(positive_label, 0)
        for label in labels if label != positive_label
    )
    false_negative = sum(
        confusion.get(positive_label, {}).get(label, 0)
        for label in labels if label != positive_label
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return labels, confusion, accuracy, precision, recall, f1_score


# ---------------------------------------------------------------------
# 5. Main program
# ---------------------------------------------------------------------

def main():
    print("=" * 65)
    print("              NAIVE BAYES CLASSIFICATION")
    print("=" * 65)

    print("\nDataset: Play Tennis")
    print("Total records :", len(DATASET))
    print("Training records:", len(TRAINING_DATA))
    print("Testing records :", len(TEST_DATA))

    model = train_naive_bayes(TRAINING_DATA)
    class_counts, _, _, priors = model

    print("\nClass Prior Probabilities:")
    for label, probability in priors.items():
        print(f"P({label}) = {probability:.4f}")

    actual_labels = []
    predicted_labels = []

    print("\nTest Instance Predictions:")
    print("-" * 65)
    print(f"{'Outlook':<12}{'Temperature':<13}{'Humidity':<12}"
          f"{'Wind':<10}{'Actual':<10}{'Predicted'}")
    print("-" * 65)

    for features, actual in TEST_DATA:
        predicted = predict(features, model)
        actual_labels.append(actual)
        predicted_labels.append(predicted)

        print(f"{features[0]:<12}{features[1]:<13}{features[2]:<12}"
              f"{features[3]:<10}{actual:<10}{predicted}")

    labels, confusion, accuracy, precision, recall, f1_score = (
        calculate_metrics(actual_labels, predicted_labels)
    )

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(" " * 14 + "".join(f"{label:>10}" for label in labels))
    for actual_label in labels:
        row = f"{actual_label:<14}"
        for predicted_label in labels:
            row += f"{confusion[actual_label][predicted_label]:>10}"
        print(row)

    print("\nEvaluation Metrics (positive class: Yes):")
    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1_score:.4f}")

    print("\nConclusion:")
    print("The categorical Naive Bayes classifier was trained on the")
    print("training records and used to predict the test records.")
    print("Its performance is summarized by the evaluation metrics above.")


if __name__ == "__main__":
    main()
