import torch
import torch.nn.functional as F

class VAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder_fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_z = torch.nn.Linear(hidden_dim, latent_dim * 2)  # Latent mean and log variance

        # Decoder
        self.z_to_hidden = torch.nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.decoder_fc2 = torch.nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.encoder_fc1(x))
        h = torch.relu(self.encoder_fc2(h))
        h = self.hidden_to_z(h)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = torch.relu(self.z_to_hidden(z))
        h = torch.relu(self.decoder_fc1(h))
        return torch.sigmoid(self.decoder_fc2(h))  # Using sigmoid to ensure output is within [0, 1]

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def loss_function(recon_x, x, mean, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
