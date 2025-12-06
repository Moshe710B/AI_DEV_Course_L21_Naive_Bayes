"""
Visualization pipeline.
"""

import os
import numpy as np
from visualizations import (
    plot_feature_distributions, plot_feature_relationships,
    plot_confusion_matrix, plot_confusion_matrices_comparison,
    plot_accuracy_comparison, plot_learned_distributions
)


def create_exploratory_visualizations(X_all, y_all, feature_names, class_names, plots_dir):
    """Create exploratory data visualizations."""
    print("\nGenerating exploratory visualizations...")
    plot_feature_distributions(X_all, y_all, feature_names, class_names, plots_dir)
    plot_feature_relationships(X_all, y_all, feature_names, class_names, plots_dir)
    print("\n[OK] Exploratory visualizations completed!")


def create_comparison_visualizations(results, class_names, feature_names,
                                     X_train, y_train, plots_dir):
    """Create comparison visualizations."""
    print("\nCreating visualization plots...")

    metrics_custom = results['custom']['metrics']
    metrics_sklearn = results['sklearn']['metrics']
    model_custom = results['custom']['model']
    model_sklearn = results['sklearn']['model']

    plot_confusion_matrix(metrics_custom['confusion_matrix'], class_names,
                         "Confusion Matrix - Custom Implementation",
                         plots_dir, "confusion_matrix_custom.png")

    plot_confusion_matrix(metrics_sklearn['confusion_matrix'], class_names,
                         "Confusion Matrix - Scikit-learn Implementation",
                         plots_dir, "confusion_matrix_sklearn.png")

    plot_confusion_matrices_comparison(metrics_custom['confusion_matrix'],
                                      metrics_sklearn['confusion_matrix'],
                                      class_names, plots_dir)

    plot_accuracy_comparison(metrics_custom, metrics_sklearn, class_names, plots_dir)

    plot_learned_distributions(model_custom, model_sklearn, feature_names,
                              class_names, X_train, y_train, plots_dir)

    print("\n[OK] All visualizations generated successfully!")
