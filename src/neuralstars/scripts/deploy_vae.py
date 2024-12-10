# deploy_lstm_vae.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
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

# Load dataset from CSV
csv_file = 'synthetic_caos_dataset.csv'
seq_len = 500  # Same sequence length as used in training
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
model.load_state_dict(torch.load('vae_model.pth'))
model.eval()

def evaluate(model, data_loader):
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            batch = batch.view(batch.size(0), seq_len+1, len(input_dims))
            with autocast():  # Mixed precision context
                means, logvars, zs, recons = model(batch[:, :-1, :])
                loss = loss_function(recons, batch, means, logvars)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

print("Evaluating on test set...")
test_loss = evaluate(model, test_loader)
print(f'Test Loss: {test_loss:.4f}')
