"""
Encoding utilities for species labels.
"""

import numpy as np
import pandas as pd


def encode_species(species_series: pd.Series) -> tuple:
    """
    Convert species names to numerical labels.

    Parameters:
    -----------
    species_series : pd.Series
        Series containing species names

    Returns:
    --------
    tuple
        - Encoded labels as numpy array
        - Mapping from species name to integer
        - Mapping from integer to species name
    """
    unique_species = sorted(species_series.unique())

    species_to_int = {species: idx for idx, species in enumerate(unique_species)}
    int_to_species = {idx: species for species, idx in species_to_int.items()}

    encoded = species_series.map(species_to_int).values

    print(f"\n[OK] Species encoding:")
    for species, idx in species_to_int.items():
        count = np.sum(encoded == idx)
        print(f"  {species}: {idx} ({count} samples)")

    return encoded, species_to_int, int_to_species


def get_class_distribution(y: np.ndarray, class_names: list) -> None:
    """
    Display class distribution.

    Parameters:
    -----------
    y : np.ndarray
        Label array
    class_names : list
        List of class names
    """
    print("\nClass distribution:")
    for idx, name in enumerate(class_names):
        count = np.sum(y == idx)
        percentage = (count / len(y)) * 100
        print(f"  {name}: {count} ({percentage:.1f}%)")
