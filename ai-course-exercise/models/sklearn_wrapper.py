"""
Scikit-learn Gaussian Naive Bayes wrapper.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix


class NaiveBayesSklearn:
    """
    Wrapper class for scikit-learn's Gaussian Naive Bayes classifier.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """Initialize the sklearn Naive Bayes classifier."""
        self.model = GaussianNB(var_smoothing=var_smoothing)
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.stds_ = None
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Gaussian Naive Bayes classifier using sklearn."""
        print("\n" + "="*60)
        print("SKLEARN NAIVE BAYES - TRAINING")
        print("="*60)

        self.model.fit(X, y)

        self.classes_ = self.model.classes_
        self.priors_ = self.model.class_prior_
        self.means_ = self.model.theta_
        self.stds_ = np.sqrt(self.model.var_)

        print(f"\nModel trained successfully!")
        print(f"Number of classes: {len(self.classes_)}")
        print(f"Number of features: {self.means_.shape[1]}")
        print(f"Variance smoothing: {self.var_smoothing}")

        for idx, class_label in enumerate(self.classes_):
            print(f"\nClass {idx} ({class_label}):")
            print(f"  Prior probability: {self.priors_[idx]:.4f}")
            print(f"  Feature means: {self.means_[idx, :]}")
            print(f"  Feature stds:  {self.stds_[idx, :]}")

        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples."""
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy on test data."""
        return self.model.score(X, y)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'priors': self.priors_,
            'means': self.means_,
            'stds': self.stds_,
            'variances': self.model.var_,
            'classes': self.classes_,
            'var_smoothing': self.var_smoothing
        }

    def print_model_summary(self, feature_names: list = None, class_names: list = None):
        """Print a summary of the learned model parameters."""
        n_classes, n_features = self.means_.shape

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(n_classes)]

        print("\n" + "="*60)
        print("MODEL SUMMARY - SKLEARN NAIVE BAYES")
        print("="*60)
        print(f"Variance smoothing parameter: {self.var_smoothing}")

        for class_idx in range(n_classes):
            print(f"\n{class_names[class_idx]}:")
            print(f"  Prior probability: {self.priors_[class_idx]:.4f}")
            print(f"  Parameters:")

            for feature_idx, feature_name in enumerate(feature_names):
                mean = self.means_[class_idx, feature_idx]
                std = self.stds_[class_idx, feature_idx]
                var = self.model.var_[class_idx, feature_idx]
                print(f"    {feature_name:15s}: mean={mean:.4f}, std={std:.4f}, var={var:.6f}")

        print("="*60)

    def get_classification_report(self, X: np.ndarray, y: np.ndarray,
                                  class_names: list = None) -> str:
        """Generate a detailed classification report."""
        predictions = self.predict(X)
        if class_names is None:
            class_names = [str(c) for c in self.classes_]
        return classification_report(y, predictions, target_names=class_names)

    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate confusion matrix."""
        predictions = self.predict(X)
        return confusion_matrix(y, predictions)
