"""Data normalization, splitting, and preparation."""
import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.

    Steps:
    - Convert columns to appropriate types.
    - Handle missing values (NaN).
    - Sort the data by the index (timestamp).

    Args:
        data (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Sort data by timestamp
    data = data.sort_index()

    # Handle missing values (interpolate NaN values)
    data = data.interpolate(method="time").fillna(method="bfill").fillna(method="ffill")
    
    print("Data preprocessing complete.")
    return data

