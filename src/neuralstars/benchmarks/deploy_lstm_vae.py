import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from neuralstars.core.unimodal_vae import LSTM_VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
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
        return torch.tensor(self.data[idx:idx+self.seq_len])

def main():
    # File path
    csv_file_path = 'toydata.csv'

    seq_len = 500

    # Load dataset
    df = load_data(csv_file_path)

    # Normalize the data
    scaler = MinMaxScaler()
    obs_p = scaler.fit_transform(df[['obs_p']].values)

    # Split dataset into test set (use the same split as in training)
    train_size = int(0.7 * len(obs_p))  # 70% for training
    valid_size = int(0.15 * len(obs_p))  # 15% for validation
    test_size = len(obs_p) - train_size - valid_size  # 15% for testing

    test_obs_p = obs_p[train_size + valid_size:]

    # Prepare test dataset
    test_dataset = TimeSeriesDataset(test_obs_p, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    input_dim = 1
    hidden_dim = 64  # Match the hidden dimension used in training
    latent_dim = 16  # Match the latent dimension used in training
    num_layers = 2  # Match the number of LSTM layers used in training

    # Load the trained model
    model = LSTM_VAE(input_dim, hidden_dim, latent_dim, num_layers, seq_len).to(device)
    model.load_state_dict(torch.load('lstm_vae_model.pth', map_location=device))
    model.eval()

    # Generate reconstructed sequences for the first test window
    reconstructions = []
    original_sequence = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            obs_p = batch.view(batch.size(0), -1, input_dim)  # Ensure input shape is (batch_size, seq_len, input_dim)
            recon, _, _ = model(obs_p)
            reconstructions.append(recon.cpu().numpy())
            original_sequence.append(obs_p.cpu().numpy())
            break  # Only process the first batch

    reconstructions = np.concatenate(reconstructions)
    original_sequence = np.concatenate(original_sequence)

    # Inverse transform the sequences to the original scale
    reconstructions = scaler.inverse_transform(reconstructions.reshape(-1, 1)).flatten()
    original_sequence = scaler.inverse_transform(original_sequence.reshape(-1, 1)).flatten()

    # Plot the original test window and the reconstructed sequence
    plt.figure(figsize=(15, 8))
    plt.plot(original_sequence, label='Original Test Sequence')
    plt.plot(reconstructions, linestyle='--', label='Reconstructed Sequence')
    plt.xlabel('Time Step')
    plt.ylabel('obs_p Value')
    plt.title('Original vs Reconstructed Test Sequence')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
