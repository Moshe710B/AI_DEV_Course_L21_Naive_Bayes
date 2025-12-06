"""
Feature visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from .config import setup_plot_directories, SPECIES_COLORS


def plot_feature_distributions(X: np.ndarray, y: np.ndarray,
                               feature_names: list, class_names: list,
                               save_path: str = "plots") -> None:
    """Plot histograms of feature distributions for each class."""
    setup_plot_directories(save_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]

        for class_idx in range(len(class_names)):
            class_data = X[y == class_idx, idx]
            ax.hist(class_data, bins=15, alpha=0.6,
                   label=class_names[class_idx],
                   color=SPECIES_COLORS[class_idx],
                   edgecolor='black', linewidth=0.5)

        ax.set_xlabel(f'{feature_name} (cm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'Distribution of {feature_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Feature Distributions by Species', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    filepath = os.path.join(save_path, 'feature_distributions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filepath}")
    plt.close()


def plot_feature_relationships(X: np.ndarray, y: np.ndarray,
                               feature_names: list, class_names: list,
                               save_path: str = "plots") -> None:
    """Plot scatter plots showing relationships between feature pairs."""
    setup_plot_directories(save_path)

    feature_pairs = [(0, 1), (2, 3), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for plot_idx, (feat_x, feat_y) in enumerate(feature_pairs):
        ax = axes[plot_idx]

        for class_idx, class_name in enumerate(class_names):
            class_mask = y == class_idx
            ax.scatter(X[class_mask, feat_x], X[class_mask, feat_y],
                      c=SPECIES_COLORS[class_idx], label=class_name,
                      alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'{feature_names[feat_x]} (cm)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{feature_names[feat_y]} (cm)', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature_names[feat_x]} vs {feature_names[feat_y]}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Feature Relationships by Species', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_path, 'feature_relationships.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filepath}")
    plt.close()
