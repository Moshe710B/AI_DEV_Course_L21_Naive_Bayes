"""
Training utilities for Gaussian Naive Bayes.
"""

import numpy as np


def calculate_class_parameters(X: np.ndarray, y: np.ndarray, epsilon: float = 1e-9) -> tuple:
    """
    Calculate parameters for Gaussian Naive Bayes.

    Parameters:
    -----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    epsilon : float
        Small value to prevent division by zero

    Returns:
    --------
    tuple
        (classes, priors, means, stds, n_classes, n_features)
    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    means = np.zeros((n_classes, n_features))
    stds = np.zeros((n_classes, n_features))
    priors = np.zeros(n_classes)

    print("\n" + "="*60)
    print("CUSTOM NAIVE BAYES - TRAINING")
    print("="*60)

    for idx, c in enumerate(classes):
        X_c = X[y == c]
        priors[idx] = X_c.shape[0] / n_samples
        means[idx, :] = X_c.mean(axis=0)
        stds[idx, :] = X_c.std(axis=0)
        stds[idx, :] += epsilon

        print(f"\nClass {idx} ({c}):")
        print(f"  Prior probability: {priors[idx]:.4f}")
        print(f"  Samples: {X_c.shape[0]}")
        print(f"  Feature means: {means[idx, :]}")
        print(f"  Feature stds:  {stds[idx, :]}")

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)

    return classes, priors, means, stds, n_classes, n_features
