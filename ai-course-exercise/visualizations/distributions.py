"""
Distribution visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from .config import setup_plot_directories, SPECIES_COLORS


def plot_learned_distributions(model_custom, model_sklearn,
                               feature_names: list, class_names: list,
                               X: np.ndarray, y: np.ndarray,
                               save_path: str = "plots") -> None:
    """Plot learned Gaussian distributions for each feature and class."""
    setup_plot_directories(save_path)

    n_features = len(feature_names)
    n_classes = len(class_names)

    fig, axes = plt.subplots(n_features, n_classes, figsize=(15, 12))

    for feat_idx, feature_name in enumerate(feature_names):
        for class_idx, class_name in enumerate(class_names):
            ax = axes[feat_idx, class_idx]

            class_data = X[y == class_idx, feat_idx]
            ax.hist(class_data, bins=12, alpha=0.4, color=SPECIES_COLORS[class_idx],
                   density=True, label='Actual Data', edgecolor='black', linewidth=0.5)

            custom_mean = model_custom.means_[class_idx, feat_idx]
            custom_std = model_custom.stds_[class_idx, feat_idx]
            sklearn_mean = model_sklearn.means_[class_idx, feat_idx]
            sklearn_std = model_sklearn.stds_[class_idx, feat_idx]

            x_min, x_max = class_data.min(), class_data.max()
            x_range = x_max - x_min
            x = np.linspace(x_min - 0.2*x_range, x_max + 0.2*x_range, 100)

            custom_pdf = (1 / (np.sqrt(2 * np.pi) * custom_std)) * \
                         np.exp(-0.5 * ((x - custom_mean) / custom_std) ** 2)
            sklearn_pdf = (1 / (np.sqrt(2 * np.pi) * sklearn_std)) * \
                          np.exp(-0.5 * ((x - sklearn_mean) / sklearn_std) ** 2)

            ax.plot(x, custom_pdf, 'r-', linewidth=2, label='Custom', alpha=0.8)
            ax.plot(x, sklearn_pdf, 'b--', linewidth=2, label='Sklearn', alpha=0.8)

            if feat_idx == 0:
                ax.set_title(f'{class_name}', fontsize=10, fontweight='bold')
            if class_idx == 0:
                ax.set_ylabel(f'{feature_name}\nDensity', fontsize=9, fontweight='bold')

            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

            text_str = f'Custom: mean={custom_mean:.2f}, std={custom_std:.2f}\n' \
                      f'Sklearn: mean={sklearn_mean:.2f}, std={sklearn_std:.2f}'
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=7,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Learned Gaussian Distributions - Custom vs Scikit-learn',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_path, 'learned_distributions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filepath}")
    plt.close()
