"""Main module."""
import numpy as np
import torch
import sys
import pandas as pd
print(sys.executable)
print(np.__version__)
print(torch.__version__)
from src.data.loader import load_data
from src.data.preprocessing import preprocess_data
from src.data.utils import split_data

# Dummy model (replace with your real model later)
class DummyTimeSeriesModel:
    def __init__(self):
        self.mean = None

    def fit(self, train_data):
        self.mean = train_data.mean()

    def predict(self, steps):
        return [self.mean] * steps

def main():
    # Path to the dataset file
    dataset_path = "path/to/attert_dataset.csv"  # Update with the actual file path

    # 1. Load the dataset
    data = load_data(dataset_path)

    # 2. Preprocess the data
    processed_data = preprocess_data(data)

    # 3. Split the data into training and testing sets
    train_data, test_data = split_data(processed_data)

    # 4. Train a dummy time series model
    model = DummyTimeSeriesModel()
    model.fit(train_data)

    # 5. Generate predictions
    forecast_steps = len(test_data)
    predictions = model.predict(forecast_steps)

    # 6. Compare predictions with the test data
    print("Predictions:")
    print(predictions[:5])  # Display the first 5 predictions
    print("Actual values:")
    print(test_data.iloc[:5].values)  # Display the first 5 actual values

if __name__ == "__main__":
    main()
