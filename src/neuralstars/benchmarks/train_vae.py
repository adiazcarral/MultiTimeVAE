import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from neuralstars.core.unimodal_vae import VAE, loss_function

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
        return torch.tensor(self.data[idx:idx+self.seq_len+1])

def main():
    # File path
    csv_file_path = 'toydata.csv'

    seq_len = 500

    # Load dataset
    df = load_data(csv_file_path)
    
    # Use only the first half of the dataset
    half_index = len(df) // 2
    df_half = df[:half_index]

    obs_p = df_half[['obs_p']].values

    # Split dataset into train and test sets
    train_size = int(0.8 * len(obs_p))  # 80% for training, 20% for testing
    train_obs_p = obs_p[:train_size]
    test_obs_p = obs_p[train_size:]

    # Prepare datasets
    train_dataset = TimeSeriesDataset(train_obs_p, seq_len)
    test_dataset = TimeSeriesDataset(test_obs_p, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    input_dim = seq_len
    hidden_dim = 128
    latent_dim = 16

    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training

    print("Starting training...")

    train_losses = []
    test_losses = []

    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            obs_p = batch[:, :-1, 0]  # Use only obs_p for training
            optimizer.zero_grad()
            
            with autocast():  # Mixed precision context
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
            for batch in test_loader:
                batch = batch.to(device)
                obs_p = batch[:, :-1, 0]
                with autocast():  # Mixed precision context
                    recon, mean, logvar = model(obs_p)
                    loss = loss_function(recon, obs_p, mean, logvar)
                epoch_loss += loss.item()
        avg_test_loss = epoch_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f'Epoch {epoch+1}, Average Test Loss: {avg_test_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'vae_model.pth')

    # Plot training vs test loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Test Loss')
    plt.legend()
    plt.show()

    # Reconstruct the test set
    model.eval()
    reconstructed = []
    originals = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            obs_p = batch[:, :-1, 0]
            with autocast():  # Mixed precision context
                recon, _, _ = model(obs_p)
            reconstructed.append(recon.cpu().numpy())
            originals.append(obs_p.cpu().numpy())

    reconstructed = np.concatenate(reconstructed)
    originals = np.concatenate(originals)

    # Plot original vs reconstructed for the first sequence
    plt.figure(figsize=(10, 5))
    plt.plot(originals[0], label='Original')
    plt.plot(reconstructed[0], label='Reconstructed')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Original vs Reconstructed Sequence')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

