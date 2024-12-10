import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from neuralstars.core.multimodal_cvae import MultiModalCVAE
from neuralstars.data.loader import load_data

def train_cvae(model, train_loader, val_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_list = [x.to(device) for x in batch]
            
            # Ensure x_list length matches the model's modalities
            assert len(x_list) == model.num_modalities-1, \
                f"x_list length ({len(x_list)}) does not match model modalities ({model.num_modalities})."

            # Randomly select the target variable to predict (condition on others)
            target_index = torch.randint(0, model.num_modalities, (1,)).item()  # Ensure within bounds

            target = x_list[target_index]  # The variable to predict
            inputs = [x for i, x in enumerate(x_list) if i != target_index]  # Exclude target from inputs

            optimizer.zero_grad()

            # Forward pass
            # print(x_list[:][0])
            reconstruction, z_mean, z_log_var = model(x_list, target_index)

            # Compute reconstruction loss for the target variable
            recon_loss = nn.MSELoss()(reconstruction, target)  # Compare reconstruction with target

            # KL divergence loss
            kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
            loss = recon_loss + kl_divergence

            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_list = [x.to(device) for x in batch]
                assert len(x_list) == model.num_modalities, \
                    f"x_list length ({len(x_list)}) does not match model modalities ({model.num_modalities})."

                target_index = torch.randint(0, model.num_modalities, (1,)).item()  # Ensure within bounds

                target = x_list[target_index]
                reconstruction, z_mean, z_log_var = model(x_list, target_index)

                recon_loss = nn.MSELoss()(reconstruction, target)
                kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                val_loss += recon_loss.item() + kl_divergence.item()

        # Print training and validation losses
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "cvae_model.pth")
    print("Model saved as cvae_model.pth")

def main(dataset_path, input_dims, latent_dim, hidden_dim, epochs, learning_rate, batch_size, device):
    # Load and preprocess dataset
    data = load_data(dataset_path)

    # Convert data columns into separate tensors
    variables = [torch.tensor(data[col].values, dtype=torch.float32).unsqueeze(-1) for col in data.columns]

    # Ensure the number of variables matches the number of input dimensions
    assert len(variables) == len(input_dims), \
        f"Number of variables ({len(variables)}) does not match input dimensions ({len(input_dims)})."

    # Split into train and validation sets
    train_vars, val_vars = train_test_split(variables, test_size=0.2, random_state=42)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(*train_vars), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_vars), batch_size=batch_size)

    # Initialize the model
    model = MultiModalCVAE(input_dims, hidden_dim, latent_dim)

    # Train the model
    train_cvae(model, train_loader, val_loader, epochs, learning_rate, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MultiModal CVAE.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--input_dims", nargs="+", type=int, required=True, help="Input dimensions of the variables.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension.")
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent space dimension.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train the model.")

    args = parser.parse_args()
    main(
        args.dataset_path,
        args.input_dims,
        args.hidden_dim,
        args.latent_dim,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.device
    )
