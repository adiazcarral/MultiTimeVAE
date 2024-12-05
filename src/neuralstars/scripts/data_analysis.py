import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from neuralstars.data.loader import load_data
from neuralstars.data.preprocessing import normalize_data, split_data
from neuralstars.data.utils import print_summary

def analyze_data(file_path):
    # Load the data
    data = load_data(file_path)

    # Display basic information
    print("\n--- Dataset Info ---")
    print(data.info())

    print("\n--- First Few Rows ---")
    print(data.head())

    # Descriptive statistics
    print("\n--- Descriptive Statistics ---")
    print(data.describe())

    # Check for missing values
    print("\n--- Missing Values ---")
    missing_counts = data.isnull().sum()
    print(missing_counts[missing_counts > 0])

    # Plot missing values heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

    # Time series trends (e.g., rain rate and reflectivity at a station)
    station_column = "RR0_DIS_ELL"  # Example column
    if station_column in data.columns:
        plt.figure(figsize=(15, 5))
        plt.plot(data["Date-UTC"], data[station_column], label=station_column, color="blue")
        plt.title(f"Time Series Plot for {station_column}")
        plt.xlabel("Date")
        plt.ylabel("Rain Rate (mm/h)")
        plt.legend()
        plt.grid()
        plt.show()

    # Correlation matrix
    numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns
    correlation_matrix = data[numerical_columns].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.show()

    print("\n--- Highly Correlated Features ---")
    for col in numerical_columns:
        correlated_features = correlation_matrix[col][abs(correlation_matrix[col]) > 0.7]
        if len(correlated_features) > 1:  # Exclude self-correlation
            print(f"{col} highly correlates with:")
            print(correlated_features.drop(col))
            print()

if __name__ == "__main__":
    file_path = "/Users/diaz/data/caos/CAOS_IT_Disdro_MRR_Data.txt"  # Update with the dataset's file path
    analyze_data(file_path)
