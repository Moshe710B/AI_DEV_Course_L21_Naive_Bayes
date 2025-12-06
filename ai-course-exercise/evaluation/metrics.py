"""
Metrics calculation utilities.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """
    Calculate comprehensive evaluation metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    n_classes : int
        Number of classes

    Returns:
    --------
    dict
        Dictionary containing various metrics
    """
    overall_accuracy = accuracy_score(y_true, y_pred)

    per_class_accuracy = []
    for class_idx in range(n_classes):
        class_mask = y_true == class_idx
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0.0)

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
