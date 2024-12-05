import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from neuralstars.data.loader import load_data

def analyze_data(dataset_path=None, use_synthetic=False):
    """
    Perform statistical analysis and visualization of the dataset. Choose between synthetic or real dataset.
    
    Args:
        dataset_path (str): Path to the real dataset.
        use_synthetic (bool): If True, generates and uses a synthetic dataset.
        
    Returns:
        None
    """
    # Load data
    if use_synthetic:
        from neuralstars.data.utils import generate_synthetic_dataset  # Assuming it's in the project
        print("Using synthetic dataset...")
        data = generate_synthetic_dataset()
    else:
        if not dataset_path:
            raise ValueError("Please provide a dataset path when use_synthetic is False.")
        print(f"Loading real dataset from {dataset_path}...")
        data = load_data(dataset_path)
    
    # Ensure 'Date-UTC' is a datetime type
    data['Date-UTC'] = pd.to_datetime(data['Date-UTC'])
    data.set_index('Date-UTC', inplace=True)
    
    # Summary statistics
    print("\nBasic Statistics:")
    print(data.describe())

    print("\nCorrelation Matrix:")
    print(data.corr())
    
    # Time series visualization
    plt.figure(figsize=(16, 8))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Time Series Overview')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plots and statistics
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    # 2. Histograms of Variables
    data.hist(figsize=(16, 10), bins=30, color='blue', alpha=0.7)
    plt.suptitle('Histograms of Variables', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3. Rolling Mean and Standard Deviation
    plt.figure(figsize=(16, 8))
    rolling_window = 24 * 30  # 30 days
    for column in data.columns:
        rolling_mean = data[column].rolling(window=rolling_window).mean()
        rolling_std = data[column].rolling(window=rolling_window).std()
        plt.plot(rolling_mean, label=f'{column} Rolling Mean', alpha=0.7)
        plt.fill_between(rolling_mean.index,
                         rolling_mean - rolling_std,
                         rolling_mean + rolling_std,
                         alpha=0.2)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Rolling Mean and Standard Deviation')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 4. Seasonal Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose

    for column in data.columns:
        decomposition = seasonal_decompose(data[column], model='additive', period=24*365)
        decomposition.plot()
        plt.suptitle(f"Seasonal Decomposition for {column}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # 5. Lag Plot
    for column in data.columns:
        plt.figure(figsize=(6, 4))
        pd.plotting.lag_plot(data[column], lag=1)
        plt.title(f"Lag Plot for {column} (lag=1)")
        plt.show()

    # 6. ACF and PACF
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    for column in data.columns:
        print(f"\nAutocorrelation and Partial Autocorrelation for {column}")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plot_acf(data[column].dropna(), ax=axes[0])
        plot_pacf(data[column].dropna(), ax=axes[1])
        axes[0].set_title(f"ACF for {column}")
        axes[1].set_title(f"PACF for {column}")
        plt.tight_layout()
        plt.show()

    # Summary
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Perform statistical analysis and visualization of a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the real dataset file.")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic dataset instead of a real one.")

    args = parser.parse_args()

    # Ensure correct usage
    if not args.use_synthetic and not args.dataset_path:
        raise ValueError(
            "You must provide a dataset path with --dataset_path or use --use_synthetic for synthetic data."
        )

    # Call the analysis function
    analyze_data(dataset_path=args.dataset_path, use_synthetic=args.use_synthetic)
