import torch
import torch.nn.functional as F

# Direct Alignment
def kl_divergence(mu1, logvar1, mu2, logvar2):
    kl = -0.5 * torch.sum(1 + logvar1 - logvar2 - ((mu1 - mu2).pow(2) + logvar1.exp()) / logvar2.exp())
    return kl

def wasserstein_distance(mu1, mu2):
    wd = torch.sum((mu1 - mu2).abs())
    return wd

# Functional Alignment
def cross_reconstruction_loss(recon_x1, x2, recon_x2, x1):
    loss = F.mse_loss(recon_x1, x2) + F.mse_loss(recon_x2, x1)
    return loss

class SharedLatentLayer(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SharedLatentLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        latent = self.fc1(x)
        recon_x = self.fc2(latent)
        return latent, recon_x

# Statistical Alignment
def common_prior(mu, logvar):
    prior_mu = torch.zeros_like(mu)
    prior_logvar = torch.zeros_like(logvar)
    kl = kl_divergence(mu, logvar, prior_mu, prior_logvar)
    return kl

def info_nce_loss(features, temperature=0.1):
    batch_size = features.shape[0]
    labels = torch.arange(batch_size).cuda()
    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix / temperature
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# Representation Structure
def contrastive_loss(z1, z2, temperature=0.1):
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size).cuda()
    similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix / temperature
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def manifold_loss(z1, z2, margin=1.0):
    distances = torch.norm(z1 - z2, dim=1)
    loss = torch.mean(F.relu(distances - margin))
    return loss
