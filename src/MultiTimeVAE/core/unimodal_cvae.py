import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

   # In Encoder class
    def forward(self, x, cond):
        cond = cond.unsqueeze(1).repeat(1, x.size(1), 1)  # Expand cond to match the sequence length of x
        x_cond = torch.cat([x, cond], dim=-1)  # Concatenate x and cond
        x_cond = self.embedding(x_cond)
        _, (h, _) = self.lstm(x_cond)
        h = h[-1]
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, cond_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim + cond_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    # In Decoder class
    def forward(self, z, cond, hidden):
        cond = cond.unsqueeze(1).repeat(1, z.size(1), 1)  # Expand cond to match the sequence length of z
        z_cond = torch.cat([z, cond], dim=-1)  # Concatenate z and cond
        out, hidden = self.lstm(z_cond, hidden)
        x_recon = self.fc(out)
        return x_recon.squeeze(1), hidden


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, cond_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, cond_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, cond):
        mean, logvar = self.encoder(x, cond)
        z = self.reparameterize(mean, logvar)
        x_recon, _ = self.decoder(z, cond, None)
        return mean, logvar, z, x_recon

def loss_function(recons, x, mean, logvar):
    recon_loss = nn.MSELoss()(recons, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_loss
