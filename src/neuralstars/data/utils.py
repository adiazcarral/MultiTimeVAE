"""Helper functions for data handling."""
import pandas as pd
import numpy as np
from typing import Tuple
import scipy.io

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

def load_mat(file_path: str, key: str):
    """
    Load and inspect a .mat file.

    Args:
        file_path (str): Path to the .mat file.
        key (str): The key to extract the main data from the .mat file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    data = scipy.io.loadmat(file_path, math_dtype=False)

    # Inspect the keys in the loaded data
    print("Keys in the .mat file:", data.keys())

    if key in data:
        df = pd.DataFrame(data[key])
        # Display the first few rows of the dataframe
        print("Head of the data:")
        print(df.head())
        return df
    else:
        raise KeyError(f"Key '{key}' not found in the .mat file.")

def convert_mat_to_csv(mat_file_path: str, csv_file_path: str):
    """
    Convert a .mat file to a CSV file.

    Args:
        mat_file_path (str): Path to the .mat file.
        csv_file_path (str): Path to save the CSV file.
    """
    data = scipy.io.loadmat(mat_file_path)
    
    # Extract the data
    if all(key in data for key in ['obs_p', 'obs_q', 'obs_teta']):
        df = pd.DataFrame({
            'obs_p': data['obs_p'].flatten(),
            'obs_q': data['obs_q'].flatten(),
            'obs_teta': data['obs_teta'].flatten(),
        })

        # Handle missing values in obs_teta (for example, using forward fill method)
        df['obs_teta'].fillna(method='ffill', inplace=True)
        
        # Check if obs_DateTime is present
        if 'obs_DateTime' in data and data['obs_DateTime'] is not None:
            df['obs_DateTime'] = pd.to_datetime(data['obs_DateTime'].flatten(), unit='s')
        else:
            df['obs_DateTime'] = pd.Series([pd.NaT] * len(df))
        
        # Save to CSV
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted {mat_file_path} to {csv_file_path}")
        return df
    else:
        raise KeyError("One or more keys not found in the .mat file.")




def inspect_mat_file(mat_file_path: str):
    """
    Inspect the contents of a .mat file and list all keys.

    Args:
        mat_file_path (str): Path to the .mat file.
    """
    data = scipy.io.loadmat(mat_file_path)
    print("Keys in the .mat file:", data.keys())


