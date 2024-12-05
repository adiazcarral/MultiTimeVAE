"""Helper functions for data handling."""
import pandas as pd
import numpy as np
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

def generate_synthetic_dataset():
    # Define time range: hourly data from 2012-10-01 to 2016-09-30
    time_index = pd.date_range(start="2012-10-01 00:00", end="2016-09-30 23:00", freq="h")
    
    # Number of data points
    n_points = len(time_index)
    
    # Generate synthetic variables
    # Variable 1: Sinusoidal pattern with seasonal trend
    var1 = 10 + 5 * np.sin(2 * np.pi * (np.arange(n_points) / (24 * 365))) + np.random.normal(0, 0.5, n_points)
    
    # Variable 2: Linear trend + random noise
    var2 = np.linspace(5, 20, n_points) + np.random.normal(0, 1, n_points)
    
    # Variable 3: Random walk process
    var3 = np.cumsum(np.random.normal(0, 1, n_points))
    
    # Create DataFrame
    data = pd.DataFrame({
        "Date-UTC": time_index,
        "Variable1": var1,
        "Variable2": var2,
        "Variable3": var3
    })
    
    # Ensure the 'Date-UTC' column is a datetime object
    data['Date-UTC'] = pd.to_datetime(data['Date-UTC'])

    # Save to CSV
    data.to_csv("synthetic_caos_dataset.csv", index=False, sep=";")
    
    # print("Synthetic dataset generated and saved as 'synthetic_caos_dataset.csv'.")
    return data

# Generate and preview dataset
synthetic_data = generate_synthetic_dataset()
# print(synthetic_data.head())
