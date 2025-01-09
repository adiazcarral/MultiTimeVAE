import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from neuralstars.core.unimodal_vae import LSTM_VAE, loss_function

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

    obs_p = df[['obs_p']].values

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(obs_p))  # 70% for training
    valid_size = int(0.15 * len(obs_p))  # 15% for validation
    test_size = len(obs_p) - train_size - valid_size  # 15% for testing

    train_obs_p = obs_p[:train_size]
    valid_obs_p = obs_p[train_size:train_size+valid_size]
    test_obs_p = obs_p[train_size+valid_size:]

    # Prepare datasets
    train_dataset = TimeSeriesDataset(train_obs_p, seq_len)
    valid_dataset = TimeSeriesDataset(valid_obs_p, seq_len)
    test_dataset = TimeSeriesDataset(test_obs_p, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    input_dim = 1
    hidden_dim = 128
    latent_dim = 16
    num_layers = 2  # Number of LSTM layers

    model = LSTM_VAE(input_dim, hidden_dim, latent_dim, num_layers, seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training

    print("Starting training...")

    train_losses = []
    valid_losses = []

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            obs_p = batch.view(batch.size(0), -1, input_dim)  # Ensure input shape is (batch_size, seq_len, input_dim)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):  # Mixed precision context
                recon, mean, logvar = model(obs_p)
                loss = loss_function(recon, obs_p, mean, logvar)
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            epoch_loss += loss.item()
    
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                obs_p = batch.view(batch.size(0), -1, input_dim)  # Ensure input shape is (batch_size, seq_len, input_dim)
                with autocast(device_type='cuda'):  # Mixed precision context
                    recon, mean, logvar = model(obs_p)
                    loss = loss_function(recon, obs_p, mean, logvar)
                epoch_loss += loss.item()
        avg_valid_loss = epoch_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        print(f'Epoch {epoch+1}, Average Validation Loss: {avg_valid_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'lstm_vae_model.pth')

    # Plot training vs validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()

    # Reconstruct the test set
    model.eval()
    reconstructed = []
    originals = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            obs_p = batch.view(batch.size(0), -1, input_dim)  # Ensure input shape is (batch_size, seq_len, input_dim)
            with autocast(device_type='cuda'):  # Mixed precision context
                recon, _, _ = model(obs_p)
            reconstructed.append(recon.cpu().numpy())
            originals.append(obs_p.cpu().numpy())

    reconstructed = np.concatenate(reconstructed)
    originals = np.concatenate(originals)

    # Plot the first 3 test sequences and their generated counterparts in the same plot
    plt.figure(figsize=(15, 8))
    for i in range(3):
        plt.plot(originals[i], label=f'Actual Test Sequence {i+1}')
        plt.plot(reconstructed[i], linestyle='--', label=f'Generated Test Sequence {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('obs_p Value')
    plt.title('Actual vs Generated Test Sequences')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
