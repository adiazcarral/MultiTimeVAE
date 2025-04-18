# multimodal_vae.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = z.unsqueeze(1)  # Adding the time dimension
        out, _ = self.lstm(z)
        x_recon = self.fc(out)
        return x_recon.squeeze(1)  # Removing the time dimension

class VAE(nn.Module):
    def __init__(self, input_dims, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoders = nn.ModuleList([Encoder(dim, hidden_dim, latent_dim) for dim in input_dims])
        self.decoders = nn.ModuleList([Decoder(latent_dim, hidden_dim, dim) for dim in input_dims])

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        means, logvars, zs, recons = [], [], [], []
        for i, encoder in enumerate(self.encoders):
            input_slice = x[:, :, i:i+1]
            mean, logvar = encoder(input_slice)
            z = self.reparameterize(mean, logvar)
            recon = self.decoders[i](z)
            means.append(mean)
            logvars.append(logvar)
            zs.append(z)
            recons.append(recon)
        return means, logvars, zs, torch.cat(recons, dim=1)

def loss_function(recons, x, means, logvars):
    recon_loss = torch.sum((recons - x[:, -1, :])**2)
    kl_loss = -0.5 * sum([torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) for mean, logvar in zip(means, logvars)])
    return recon_loss + kl_loss
