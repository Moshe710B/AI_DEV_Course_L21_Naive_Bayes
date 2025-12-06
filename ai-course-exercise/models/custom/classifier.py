"""
Custom Gaussian Naive Bayes Classifier.
"""

import numpy as np
from .trainer import calculate_class_parameters
from .probability import calculate_log_likelihood


class NaiveBayesCustom:
    """
    Gaussian Naive Bayes Classifier implemented from scratch.
    """

    def __init__(self, epsilon: float = 1e-9):
        """Initialize the classifier."""
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.stds_ = None
        self.epsilon = epsilon
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Gaussian Naive Bayes classifier."""
        results = calculate_class_parameters(X, y, self.epsilon)
        self.classes_, self.priors_, self.means_, self.stds_, self.n_classes_, self.n_features_ = results
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples."""
        n_samples = X.shape[0]
        log_probabilities = np.zeros((n_samples, self.n_classes_))

        for class_idx in range(self.n_classes_):
            log_prior = np.log(self.priors_[class_idx])
            log_likelihood = calculate_log_likelihood(
                X, self.means_, self.stds_, class_idx, self.epsilon
            )
            log_probabilities[:, class_idx] = log_prior + log_likelihood

        log_probabilities -= np.max(log_probabilities, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy on test data."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'priors': self.priors_,
            'means': self.means_,
            'stds': self.stds_,
            'classes': self.classes_
        }

    def print_model_summary(self, feature_names: list = None, class_names: list = None):
        """Print a summary of the learned model parameters."""
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.n_features_)]
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.n_classes_)]

        print("\n" + "="*60)
        print("MODEL SUMMARY - CUSTOM NAIVE BAYES")
        print("="*60)

        for class_idx in range(self.n_classes_):
            print(f"\n{class_names[class_idx]}:")
            print(f"  Prior probability: {self.priors_[class_idx]:.4f}")
            print(f"  Parameters:")

            for feature_idx, feature_name in enumerate(feature_names):
                mean = self.means_[class_idx, feature_idx]
                std = self.stds_[class_idx, feature_idx]
                print(f"    {feature_name:15s}: mean={mean:.4f}, std={std:.4f}")

        print("="*60)
