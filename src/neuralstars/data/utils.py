"""Helper functions for data handling."""
import pandas as pd
from typing import Tuple

def split_data(data: pd.DataFrame, train_ratio=0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): Preprocessed dataset.
        train_ratio (float): Proportion of the dataset to use for training.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.
    """
    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    print(f"Dataset split: {len(train_data)} training rows, {len(test_data)} testing rows.")
    return train_data, test_data

