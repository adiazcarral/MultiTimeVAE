import torch
import torch.nn.functional as F

class LSTM_VAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, seq_len):
        super(LSTM_VAE, self).__init__()
        self.seq_len = seq_len
        self.encoder_lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_z = torch.nn.Linear(hidden_dim, latent_dim * 2)  # Latent mean and log variance
        self.z_to_hidden = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = torch.nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_output = torch.nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h, _) = self.encoder_lstm(x)
        h = h[-1]  # Use the last layer's hidden state
        h = self.hidden_to_z(h)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # Repeat z for each time step
        h = self.z_to_hidden(z)
        out, _ = self.decoder_lstm(h)
        return self.decoder_output(out)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def loss_function(recon_x, x, mean, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
