import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # hidden shape: (1, batch_size, hidden_dim)
        hidden = hidden.squeeze(0)  # Remove the extra dimension
        z_mean = self.fc_mean(hidden)
        z_log_var = self.fc_log_var(hidden)
        return z_mean, z_log_var


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim + condition_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, condition):
        # Repeat latent vector to match sequence length
        z_repeated = z.unsqueeze(1).repeat(1, condition.size(1), 1)
        lstm_input = torch.cat([z_repeated, condition], dim=2)
        lstm_output, _ = self.lstm(lstm_input)
        reconstruction = self.fc_out(lstm_output)
        return reconstruction


class MultiModalCVAE(nn.Module):
    def __init__(self, input_dims, hidden_dim, latent_dim):
        """
        MultiModalCVAE for predicting a target variable conditioned on the rest.

        Parameters:
        - input_dims (list[int]): List of input dimensions for each variable.
        - hidden_dim (int): Hidden dimension for LSTM layers.
        - latent_dim (int): Dimension of the latent space.
        """
        super().__init__()
        self.num_modalities = len(input_dims)

        # Encoders for each variable
        self.encoders = nn.ModuleList([
            LSTMEncoder(input_dim, hidden_dim, latent_dim) for input_dim in input_dims
        ])

        # Decoders for each variable
        self.decoders = nn.ModuleList([
            LSTMDecoder(latent_dim, latent_dim * (self.num_modalities - 1), hidden_dim, input_dim)
            for input_dim in input_dims
        ])

    def forward(self, x_list, target_index):
        """
        Parameters:
        - x_list (list of tensors): List of input tensors, one for each variable. 
                                    Each tensor has shape [batch_size, seq_length, feature_dim].
        - target_index (int): Index of the variable to predict.

        Returns:
        - reconstruction (Tensor): Reconstructed target variable.
        - z_mean_target (Tensor): Latent mean for the target variable.
        - z_log_var_target (Tensor): Latent log variance for the target variable.
        """
        # Latent representations of all variables
        z_means = []
        z_log_vars = []

        for i, encoder in enumerate(self.encoders):
            z_mean, z_log_var = encoder(x_list[i])  # Encode each variable
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)

        # Separate target latent space and condition latent space
        z_mean_target = z_means[target_index]
        z_log_var_target = z_log_vars[target_index]

        # Combine latent representations of conditioning variables
        condition = torch.cat(
            [z_means[i] for i in range(self.num_modalities) if i != target_index],
            dim=1
        )

        # Decode the target variable using its latent space and the condition
        target_decoder = self.decoders[target_index]
        reconstruction = target_decoder(z_mean_target, condition)

        return reconstruction, z_mean_target, z_log_var_target

