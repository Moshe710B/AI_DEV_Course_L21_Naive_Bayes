"""
Data package for IRIS dataset handling.
"""

from .loader import load_iris_data, split_train_test
from .encoder import encode_species, get_class_distribution
from .preprocessor import prepare_data

__all__ = [
    'load_iris_data',
    'split_train_test',
    'encode_species',
    'get_class_distribution',
    'prepare_data'
]
