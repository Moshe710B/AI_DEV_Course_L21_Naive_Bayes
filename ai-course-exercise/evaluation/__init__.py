"""
Evaluation package for model comparison.
"""

from .metrics import calculate_metrics
from .comparator import compare_predictions, compare_parameters
from .reporter import compare_metrics, generate_comparison_summary

__all__ = [
    'calculate_metrics',
    'compare_predictions',
    'compare_parameters',
    'compare_metrics',
    'generate_comparison_summary'
]
