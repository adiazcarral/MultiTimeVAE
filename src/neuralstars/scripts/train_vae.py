import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.amp import autocast, GradScaler
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

def main():
    # Load dataset from CSV
    csv_file = 'synthetic_caos_dataset.csv'
    seq_len = 500  # Reduced sequence length to 500
    data = load_data(csv_file).values

    # Split dataset into train, validation, and test sets based on time
    train_size = int(0.7 * len(data))
    val_size = int(0.2 * len(data))
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    train_dataset = TimeSeriesDataset(train_data, seq_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    input_dims = [1, 1, 1]
    hidden_dim = 128
    latent_dim = 16

    model = VAE(input_dims, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    scaler = GradScaler('cuda')  # Initialize GradScaler for mixed precision training

    print("Starting training...")

    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            batch = batch.view(batch.size(0), seq_len+1, len(input_dims))

            with autocast('cuda'):  # Mixed precision context
                means, logvars, zs, recons = model(batch[:, :-1, :])
                loss = loss_function(recons, batch, means, logvars)
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            epoch_loss += loss.item()
    
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}, Average Training Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                val_batch = val_batch.view(val_batch.size(0), seq_len+1, len(input_dims))
                with autocast('cuda'):  # Mixed precision context
                    means, logvars, zs, recons = model(val_batch[:, :-1, :])
                    loss = loss_function(recons, val_batch, means, logvars)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}, Average Validation Loss: {avg_val_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'vae_model.pth')

if __name__ == "__main__":
    main()
