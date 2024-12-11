import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from neuralstars.core.unimodal_cvae import ConditionalVAE, loss_function
from neuralstars.data.utils import load_toy_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = load_toy_data(file_path)
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
    csv_file_path = 'C:/Users/diaz/work/neural-stars/neuralstars/src/neuralstars/scripts/toydata.csv'  # Just the filename
    seq_len = 500

    # Load dataset
    df = load_data(csv_file_path)
    print("Head of the dataset after interpolation:")
    print(df.head())

    # Use only the first half of the dataset
    half_index = len(df) // 2
    df_half = df[:half_index]

    obs_p = df_half[['obs_p']].values
    cond_data = df_half[['obs_q', 'obs_teta']].values

    # Split dataset into train and test sets
    train_size = int(0.8 * len(obs_p))  # 80% for training, 20% for testing
    train_obs_p = obs_p[:train_size]
    test_obs_p = obs_p[train_size:]
    train_cond_data = cond_data[:train_size]
    test_cond_data = cond_data[train_size:]

    # Prepare datasets
    train_dataset = TimeSeriesDataset(np.hstack([train_obs_p, train_cond_data]), seq_len)
    test_dataset = TimeSeriesDataset(np.hstack([test_obs_p, test_cond_data]), seq_len)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    input_dim = 1
    cond_dim = 2
    hidden_dim = 128
    latent_dim = 16

    model = ConditionalVAE(input_dim, cond_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda')  # Initialize GradScaler for mixed precision training

    print("Starting training...")

    train_losses = []
    test_losses = []

    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            obs_p = batch[:, :, :1]
            cond_data = batch[:, :, 1:]
            optimizer.zero_grad()
            
            with autocast('cuda'):  # Mixed precision context
                mean, logvar, z, recons = model(obs_p[:, :-1, :], cond_data[:, :-1, :])
                loss = loss_function(recons, obs_p[:, :-1, :], mean, logvar)
        
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
                obs_p = batch[:, :, :1]
                cond_data = batch[:, :, 1:]
                with autocast('cuda'):  # Mixed precision context
                    mean, logvar, z, recons = model(obs_p[:, :-1, :], cond_data[:, :-1, :])
                    loss = loss_function(recons, obs_p[:, :-1, :], mean, logvar)
                epoch_loss += loss.item()
        avg_test_loss = epoch_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f'Epoch {epoch+1}, Average Test Loss: {avg_test_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'conditional_vae_model.pth')

    # Plot training vs test loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Test Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
