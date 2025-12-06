"""
Main pipeline executor.
"""

import os
import sys
import numpy as np
from datetime import datetime

from data import prepare_data
from models import NaiveBayesCustom, NaiveBayesSklearn
from visualizations import (
    plot_feature_distributions, plot_feature_relationships,
    plot_confusion_matrix, plot_confusion_matrices_comparison,
    plot_accuracy_comparison, plot_learned_distributions
)
from evaluation import (
    calculate_metrics, compare_predictions, compare_parameters,
    compare_metrics, generate_comparison_summary
)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def train_and_evaluate_models(X_train, y_train, X_test, y_test,
                               feature_names, class_names):
    """Train both models and evaluate performance."""
    print_header("STEP 3: CUSTOM NAIVE BAYES IMPLEMENTATION")
    print("\nTraining custom Naive Bayes model...")
    model_custom = NaiveBayesCustom(epsilon=1e-9)
    model_custom.fit(X_train, y_train)

    print("\nMaking predictions on test set...")
    y_pred_custom = model_custom.predict(X_test)
    y_proba_custom = model_custom.predict_proba(X_test)
    metrics_custom = calculate_metrics(y_test, y_pred_custom, len(class_names))

    print(f"\n[OK] Custom model accuracy: {metrics_custom['overall_accuracy']:.4f} "
          f"({metrics_custom['overall_accuracy']*100:.2f}%)")
    model_custom.print_model_summary(feature_names, class_names)

    print_header("STEP 4: SCIKIT-LEARN NAIVE BAYES IMPLEMENTATION")
    print("\nTraining sklearn Naive Bayes model...")
    model_sklearn = NaiveBayesSklearn(var_smoothing=1e-9)
    model_sklearn.fit(X_train, y_train)

    print("\nMaking predictions on test set...")
    y_pred_sklearn = model_sklearn.predict(X_test)
    y_proba_sklearn = model_sklearn.predict_proba(X_test)
    metrics_sklearn = calculate_metrics(y_test, y_pred_sklearn, len(class_names))

    print(f"\n[OK] Sklearn model accuracy: {metrics_sklearn['overall_accuracy']:.4f} "
          f"({metrics_sklearn['overall_accuracy']*100:.2f}%)")
    model_sklearn.print_model_summary(feature_names, class_names)

    return {
        'custom': {
            'model': model_custom,
            'predictions': y_pred_custom,
            'probabilities': y_proba_custom,
            'metrics': metrics_custom
        },
        'sklearn': {
            'model': model_sklearn,
            'predictions': y_pred_sklearn,
            'probabilities': y_proba_sklearn,
            'metrics': metrics_sklearn
        }
    }
