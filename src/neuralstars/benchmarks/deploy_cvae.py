import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from neuralstars.core.unimodal_cvae import ConditionalVAE, loss_function
from neuralstars.data.utils import load_toy_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def iterative_forecast(model, initial_sequence, num_steps, cond_data):
    model.eval()
    predictions = []
    current_sequence = initial_sequence

    with torch.no_grad():
        for i in range(num_steps):
            current_sequence = current_sequence.to(device)
            cond_step = cond_data[i:i+1, :].to(device)
            with autocast(enabled=True):  # Mixed precision context
                _, _, _, recons = model(current_sequence.view(1, -1, 1), cond_step.view(1, -1, 2))
                next_step = recons[:, -1, :].cpu().numpy()  # Get the last predicted step
            predictions.append(next_step)
            next_step_tensor = torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).to(device)
            current_sequence = torch.cat((current_sequence[:, 1:, :], next_step_tensor), dim=1)  # Update sequence
    return np.concatenate(predictions, axis=0)

def main():
    # File path
    csv_file_path = 'path/to/toydata.csv'
    seq_len = 500

    # Load dataset
    df = load_toy_data(csv_file_path)
    obs_p = df[['obs_p']].values
    cond_data = df[['obs_q', 'obs_teta']].values

    input_dim = 1
    cond_dim = 2
    hidden_dim = 128
    latent_dim = 16

    model = ConditionalVAE(input_dim, cond_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load('conditional_vae_model.pth', map_location=device))

    # Use the first sequence from the dataset to start the forecasting
    initial_sequence = torch.tensor(obs_p[:seq_len], dtype=torch.float32).to(device)
    
    print("Starting forecasting future steps...")
    future_steps = iterative_forecast(model, initial_sequence, len(obs_p) - seq_len, torch.tensor(cond_data[seq_len:], dtype=torch.float32))

    # Compare results with actual test data
    actual_future_steps = obs_p[seq_len:]
    mse = np.mean((future_steps - actual_future_steps) ** 2)
    print(f'Mean Squared Error (MSE) of the forecast: {mse:.4f}')

    # Plot the forecasted vs. actual values
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(actual_future_steps)), actual_future_steps, label='Actual obs_p')
    plt.plot(np.arange(len(future_steps)), future_steps, label='Forecasted obs_p')
    plt.xlabel('Time Steps')
    plt.ylabel('obs_p')
    plt.title('Forecasted vs. Actual obs_p')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
