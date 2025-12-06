"""
Probability calculation utilities for Gaussian Naive Bayes.
"""

import numpy as np


def calculate_gaussian_probability(x: float, mean: float, std: float) -> float:
    """
    Calculate Gaussian probability density function.

    Formula: P(x) = (1 / √(2πσ²)) * exp(-(x - μ)² / (2σ²))

    Parameters:
    -----------
    x : float
        Feature value
    mean : float
        Mean (μ) of the feature distribution
    std : float
        Standard deviation (σ) of the feature distribution

    Returns:
    --------
    float
        Probability density value
    """
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    coefficient = 1.0 / (np.sqrt(2 * np.pi) * std)
    probability = coefficient * np.exp(exponent)
    return probability


def calculate_log_likelihood(X: np.ndarray, means: np.ndarray, stds: np.ndarray,
                             class_idx: int, epsilon: float = 1e-9) -> np.ndarray:
    """
    Calculate log-likelihood for a class.

    Uses log-space to prevent numerical underflow.

    Parameters:
    -----------
    X : np.ndarray
        Features to evaluate
    means : np.ndarray
        Mean values for all classes and features
    stds : np.ndarray
        Standard deviations for all classes and features
    class_idx : int
        Index of the class
    epsilon : float
        Small value to prevent log(0)

    Returns:
    --------
    np.ndarray
        Log-likelihood values
    """
    n_samples, n_features = X.shape
    log_likelihood = np.zeros(n_samples)

    mean = means[class_idx, :]
    std = stds[class_idx, :]

    for feature_idx in range(n_features):
        probabilities = calculate_gaussian_probability(
            X[:, feature_idx],
            mean[feature_idx],
            std[feature_idx]
        )
        log_likelihood += np.log(probabilities + epsilon)

    return log_likelihood
