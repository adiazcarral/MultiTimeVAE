"""Data loader."""
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(
            file_path,
            delimiter=";",
            decimal=".",
            parse_dates=["Date-UTC"],
            index_col="Date-UTC",
        )
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")
    except Exception as e:
        raise ValueError(f"An error occurred while loading the dataset: {e}")

