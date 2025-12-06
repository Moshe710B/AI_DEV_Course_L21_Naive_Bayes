"""
Data preprocessing and preparation utilities.
"""

import numpy as np
from .loader import load_iris_data, split_train_test
from .encoder import encode_species


def prepare_data(filepath: str) -> dict:
    """
    Complete data preparation pipeline.

    Parameters:
    -----------
    filepath : str
        Path to the IRIS.csv file

    Returns:
    --------
    dict
        Dictionary containing prepared datasets and metadata
    """
    print("="*60)
    print("IRIS DATASET PREPARATION")
    print("="*60)

    df = load_iris_data(filepath)
    train_df, test_df = split_train_test(df)

    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values

    y_train, species_to_int, int_to_species = encode_species(train_df['species'])
    y_test, _, _ = encode_species(test_df['species'])

    class_names = [int_to_species[i] for i in range(len(int_to_species))]

    print(f"\n[OK] Feature extraction completed:")
    print(f"  Features: {feature_columns}")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")

    print(f"\n[OK] Feature statistics (Training set):")
    for idx, feature in enumerate(feature_columns):
        print(f"  {feature:15s}: min={X_train[:, idx].min():.2f}, "
              f"max={X_train[:, idx].max():.2f}, "
              f"mean={X_train[:, idx].mean():.2f}, "
              f"std={X_train[:, idx].std():.2f}")

    print("="*60)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_columns,
        'class_names': class_names,
        'species_to_int': species_to_int,
        'int_to_species': int_to_species
    }
