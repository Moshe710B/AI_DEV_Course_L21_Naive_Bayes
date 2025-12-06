"""
Comparison visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from .config import setup_plot_directories


def plot_accuracy_comparison(custom_metrics: dict, sklearn_metrics: dict,
                            class_names: list, save_path: str = "plots") -> None:
    """Plot bar chart comparing accuracies between implementations."""
    setup_plot_directories(save_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['Overall Accuracy'] + [f'{name}\nAccuracy' for name in class_names]
    custom_values = [custom_metrics['overall_accuracy']] + custom_metrics['per_class_accuracy']
    sklearn_values = [sklearn_metrics['overall_accuracy']] + sklearn_metrics['per_class_accuracy']

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, custom_values, width, label='Custom NumPy',
                   color='#FF6B6B', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, sklearn_values, width, label='Scikit-learn',
                   color='#4ECDC4', edgecolor='black', linewidth=1)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy Comparison: Custom vs Scikit-learn',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(save_path, 'accuracy_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filepath}")
    plt.close()
