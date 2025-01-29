import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from neuralstars.core.unimodal_vae import VAE, loss_function

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.seq_len])

def main():
    # Load dataset
    df = pd.read_csv('toydata.csv')
    obs_p = df[['obs_p']].values

    # Normalize the data
    scaler = MinMaxScaler()
    obs_p = scaler.fit_transform(obs_p)

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(obs_p))
    valid_size = int(0.15 * len(obs_p))
    test_size = len(obs_p) - train_size - valid_size

    train_obs_p = obs_p[:train_size]
    valid_obs_p = obs_p[train_size:train_size+valid_size]
    test_obs_p = obs_p[train_size+valid_size:]

    seq_len = 50
    train_dataset = TimeSeriesDataset(train_obs_p, seq_len)
    valid_dataset = TimeSeriesDataset(valid_obs_p, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=seq_len, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=seq_len, shuffle=False)

    input_dim = seq_len
    hidden_dim = 128
    latent_dim = 32

    model = VAE(input_dim, hidden_dim, latent_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.view(batch.size(0), -1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer.zero_grad()
            recon, mean, logvar = model(batch)
            loss = loss_function(recon, batch, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Training loss: {train_loss:.4f}")

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.view(batch.size(0), -1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                recon, mean, logvar = model(batch)
                loss = loss_function(recon, batch, mean, logvar)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader.dataset)
        print(f"Epoch {epoch+1}, Validation loss: {valid_loss:.4f}")

    torch.save(model.state_dict(), 'vae_model.pth')

if __name__ == "__main__":
    main()
