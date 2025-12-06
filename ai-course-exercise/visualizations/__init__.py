"""
Visualizations package for Naive Bayes analysis.
"""

from .features import plot_feature_distributions, plot_feature_relationships
from .confusion import plot_confusion_matrix, plot_confusion_matrices_comparison
from .comparisons import plot_accuracy_comparison
from .distributions import plot_learned_distributions

__all__ = [
    'plot_feature_distributions',
    'plot_feature_relationships',
    'plot_confusion_matrix',
    'plot_confusion_matrices_comparison',
    'plot_accuracy_comparison',
    'plot_learned_distributions'
]
