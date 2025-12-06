"""
Visualization configuration and styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color palette for the 3 iris species
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']
SPECIES_COLORS = {
    0: COLORS[0],  # Setosa - Red
    1: COLORS[1],  # Versicolor - Teal
    2: COLORS[2]   # Virginica - Blue
}


def setup_plot_directories(base_path: str = "plots") -> str:
    """
    Create directory for saving plots.

    Parameters:
    -----------
    base_path : str
        Base directory path for plots

    Returns:
    --------
    str
        Full path to plots directory
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return base_path
