
import numpy as np


def calculate_confusion_matrix(y_true, y_pred, num_classes=6):
    """
    Calculate confusion matrix manually.

    Rows represent actual classes.
    Columns represent predicted classes.
    """

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=int
    )

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1

    return matrix


def calculate_classification_metrics(confusion_matrix):
    """
    Calculate precision, recall, F1-score,
    and accuracy for every class.
    """

    num_classes = confusion_matrix.shape[0]

    results = []

    for i in range(num_classes):

        true_positive = confusion_matrix[i, i]

        false_positive = (
            confusion_matrix[:, i].sum() - true_positive
        )

        false_negative = (
            confusion_matrix[i, :].sum() - true_positive
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        class_accuracy = (
            true_positive / confusion_matrix[i, :].sum()
            if confusion_matrix[i, :].sum() > 0
            else 0.0
        )

        results.append({
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": class_accuracy
        })

    return results


def calculate_overall_accuracy(y_true, y_pred):
    """
    Calculate overall classification accuracy.
    """

    correct = np.sum(np.array(y_true) == np.array(y_pred))

    return correct / len(y_true)
