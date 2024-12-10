import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
import matplotlib.pyplot as plt
from neuralstars.core.multimodal_vae import VAE, loss_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path: str) -> pd.DataFrame:
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

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.seq_len+1])

def iterative_forecast(model, initial_sequence, num_steps):
    model.eval()
    predictions = []
    current_sequence = initial_sequence.to(device)

    with torch.no_grad():
        for _ in range(num_steps):
            current_sequence = current_sequence.view(1, -1, len(input_dims))
            with autocast('cuda', enabled=True):
                _, _, _, recons = model(current_sequence)
                recons = recons.view(1, -1, len(input_dims))
                next_step = recons[:, -1, :].cpu().numpy()
            predictions.append(next_step)
            next_step_tensor = torch.tensor(next_step, dtype=torch.float32).to(device).unsqueeze(0)
            current_sequence = torch.cat((current_sequence[:, 1:, :], next_step_tensor), dim=1)
    return np.concatenate(predictions, axis=0)

if __name__ == "__main__":
    # Load dataset from CSV
    csv_file = 'synthetic_caos_dataset1.csv'
    seq_len = 500
    data = load_data(csv_file).values

    # Split dataset into train, validation, and test sets based on time
    train_size = int(0.7 * len(data))
    val_size = int(0.2 * len(data))
    test_size = len(data) - train_size - val_size

    test_data = data[train_size+val_size:]

    test_dataset = TimeSeriesDataset(test_data, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    input_dims = [1, 1, 1]
    hidden_dim = 128
    latent_dim = 16

    model = VAE(input_dims, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load('vae_model.pth', map_location=device, weights_only=True))

    # Use the first sequence from the test set to start the forecasting
    initial_sequence = torch.tensor(test_data[:seq_len], dtype=torch.float32).to(device)
    
    print("Starting forecasting future steps...")
    future_steps = iterative_forecast(model, initial_sequence, len(test_data) - seq_len)
    print(f"Forecasted future steps:\n{future_steps}")

    # Compare results with actual test data
    actual_future_steps = test_data[seq_len:]
    print(f"Actual future steps:\n{actual_future_steps}")

    # Calculate metrics for comparison
    mse = np.mean((future_steps - actual_future_steps) ** 2)
    print(f'Mean Squared Error (MSE) of the forecast: {mse:.4f}')

    # Plot the forecasted vs. actual values
    plt.figure(figsize=(15, 10))
    time_range = np.arange(len(actual_future_steps))

    for i, variable in enumerate(["Variable1", "Variable2", "Variable3"]):
        plt.subplot(3, 1, i+1)
        plt.plot(time_range, actual_future_steps[:, i], label='Actual ' + variable)
        plt.plot(time_range, future_steps[:, i], label='Forecasted ' + variable)
        plt.title(f'{variable} Forecast vs. Actual')
        plt.xlabel('Time Steps')
        plt.ylabel(variable)
        plt.legend()

    plt.tight_layout()
    plt.show()
