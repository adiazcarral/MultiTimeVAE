import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from neuralstars.core.multimodal_cvae import MultiModalCVAE
from neuralstars.data.loader import load_dataset
import matplotlib.pyplot as plt
import numpy as np

def plot_results(original, generated, title, variable_name):
    """
    Plots original vs generated data.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(original, label="Original", alpha=0.7)
    plt.plot(generated, label="Generated", alpha=0.7)
    plt.title(f"{title} - {variable_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def evaluate_generation(test_loader, model, device):
    """
    Evaluates the model on the test data and plots the results.
    """
    model.eval()
    mse_scores = []

    for batch in test_loader:
        x_list = [x.to(device) for x in batch[:-1]]
        condition = batch[-1].to(device)

        # Target variable index is pre-defined or sequentially selected
        target_index = np.random.choice(len(x_list))
        target = x_list[target_index]
        inputs = [x for i, x in enumerate(x_list) if i != target_index]

        with torch.no_grad():
            reconstruction, _, _ = model(x_list, target_index)

        # Compute mean squared error for the generated sequence
        mse = torch.mean((reconstruction - target[:, 1:, :]) ** 2).item()
        mse_scores.append(mse)

        # Plot the results
        plot_results(
            target.squeeze(2).cpu().numpy(),
            reconstruction.squeeze(2).cpu().numpy(),
            title="Test Data",
            variable_name=f"Variable {target_index + 1}"
        )

    avg_mse = np.mean(mse_scores)
    print(f"Average MSE across test variables: {avg_mse:.4f}")

def main(dataset_path, input_dims, latent_dim, condition_dim, batch_size, device):
    # Load and preprocess dataset
    data = load_dataset(dataset_path)
    variables = [
        torch.tensor(data[col].values, dtype=torch.float32).unsqueeze(-1)
        for col in data.columns[1:4]
    ]
    conditions = torch.tensor(data[data.columns[0]].values, dtype=torch.float32).unsqueeze(-1)

    # Split into train, validation, and test sets
    train_vars, test_vars, train_cond, test_cond = train_test_split(
        variables, conditions, test_size=0.2, random_state=42
    )
    val_vars, test_vars, val_cond, test_cond = train_test_split(
        test_vars, test_cond, test_size=0.5, random_state=42
    )

    # Prepare test DataLoader
    test_dataset = TensorDataset(*test_vars, test_cond)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load trained model
    model = MultiModalCVAE(input_dims, latent_dim, condition_dim)
    model.load_state_dict(torch.load("cvae_model.pth", map_location=device))
    model.to(device)

    # Evaluate the model on the test dataset
    evaluate_generation(test_loader, model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a MultiModal CVAE.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--input_dims", nargs="+", type=int, required=True, help="Input dimensions of the variables.")
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent space dimension.")
    parser.add_argument("--condition_dim", type=int, default=1, help="Conditioning variable dimension.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the test.")

    args = parser.parse_args()
    main(args.dataset_path, args.input_dims, args.latent_dim, args.condition_dim, args.batch_size, args.device)

