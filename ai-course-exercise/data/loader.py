"""
Data loading utilities for IRIS dataset.
"""

import pandas as pd


def load_iris_data(filepath: str) -> pd.DataFrame:
    """
    Load IRIS dataset from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the IRIS.csv file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with all columns
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Successfully loaded data from {filepath}")
        print(f"  Total samples: {len(df)}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def split_train_test(df: pd.DataFrame) -> tuple:
    """
    Split dataframe into training and test sets based on 'Training / Test' column.

    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataframe with 'Training / Test' column

    Returns:
    --------
    tuple
        Training dataframe and test dataframe
    """
    test_marker_column = df.columns[0]

    train_df = df[df[test_marker_column].isna() | (df[test_marker_column] == '')].copy()
    test_df = df[df[test_marker_column] == 'TEST'].copy()

    print(f"\n[OK] Data split completed:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    return train_df, test_df
