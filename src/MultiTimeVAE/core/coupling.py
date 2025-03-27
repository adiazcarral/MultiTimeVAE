import torch
import torch.nn.functional as F

# Define Coupling Techniques

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

# New Coupling Functions

# Cross-Alignment (CA) Loss
def cross_alignment_loss(encoder_i, decoder_j, x_i, x_j):
    latent_i = encoder_i(x_i)
    recon_j = decoder_j(latent_i)
    ca_loss = F.l1_loss(recon_j, x_j)
    return ca_loss

# Distribution-Alignment (DA) Loss (Wasserstein Distance)
def wasserstein_distance(mu_i, logvar_i, mu_j, logvar_j):
    # Calculate the square root of the covariance matrices
    cov_i = torch.exp(logvar_i / 2)
    cov_j = torch.exp(logvar_j / 2)
    
    # Calculate the 2-Wasserstein distance
    wd = torch.sum((mu_i - mu_j).pow(2)) + torch.sum((cov_i - cov_j).pow(2))
    return wd

def distribution_alignment_loss(encoders, samples):
    da_loss = 0
    num_modalities = len(encoders)
    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:
                mu_i, logvar_i = encoders[i](samples[i])
                mu_j, logvar_j = encoders[j](samples[j])
                da_loss += wasserstein_distance(mu_i, logvar_i, mu_j, logvar_j)
    return da_loss

# Cross- and Distribution Alignment (CADA-VAE) Loss
def cada_vae_loss(vae_loss, gamma, delta, encoders, decoders, samples):
    lca_loss = 0
    lda_loss = distribution_alignment_loss(encoders, samples)
    
    num_modalities = len(samples)
    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:
                lca_loss += cross_alignment_loss(encoders[i], decoders[j], samples[i], samples[j])
    
    cada_loss = vae_loss + gamma * lca_loss + delta * lda_loss
    return cada_loss

""" # Example usage
# Assuming encoder and decoder models for each modality, and samples from each modality
encoders = [encoder1, encoder2, encoder3]
decoders = [decoder1, decoder2, decoder3]
samples = [sample1, sample2, sample3]

vae_loss = ...  # Calculated VAE loss
gamma = 1.0
delta = 1.0

total_loss = cada_vae_loss(vae_loss, gamma, delta, encoders, decoders, samples)
 """