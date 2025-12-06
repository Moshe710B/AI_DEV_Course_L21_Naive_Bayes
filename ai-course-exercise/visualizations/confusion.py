"""
Confusion matrix visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .config import setup_plot_directories


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str,
                         save_path: str = "plots", filename: str = "confusion_matrix.png") -> None:
    """Plot confusion matrix as a heatmap."""
    setup_plot_directories(save_path)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')

    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filepath}")
    plt.close()


def plot_confusion_matrices_comparison(cm_custom: np.ndarray, cm_sklearn: np.ndarray,
                                       class_names: list, save_path: str = "plots") -> None:
    """Plot side-by-side comparison of confusion matrices."""
    setup_plot_directories(save_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='gray')
    axes[0].set_title('Custom NumPy Implementation', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='gray')
    axes[1].set_title('Scikit-learn Implementation', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

    plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_path, 'confusion_matrices_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filepath}")
    plt.close()
