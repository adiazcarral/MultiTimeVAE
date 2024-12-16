import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, backend as K
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Suppress oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_data(file_path: str):
    return pd.read_csv(file_path)

def split_data(data, valid_perc=0.1, shuffle=True):
    return train_test_split(data, test_size=valid_perc, shuffle=shuffle)

def scale_data(train_data, valid_data):
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_valid_data = scaler.transform(valid_data)
    return scaled_train_data, scaled_valid_data, scaler

# Load, split, and scale the data
data = load_data('toydata.csv')
train_data, valid_data = split_data(data, valid_perc=0.1)
scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

train_conditions = scaled_train_data[:, :-1]
train_target = scaled_train_data[:, -1].reshape(-1, 1)
valid_conditions = scaled_valid_data[:, :-1]
valid_target = scaled_valid_data[:, -1].reshape(-1, 1)

def instantiate_vae_model(sequence_length, feature_dim, latent_dim):
    # Encoder
    inputs = layers.Input(shape=(sequence_length,))
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(inputs)
    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim + feature_dim,))
    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(decoder_inputs)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    outputs = layers.Dense(1, activation='linear')(x)

    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = models.Model(decoder_inputs, outputs, name='decoder')

    return encoder, decoder

def train_vae(vae, train_data, max_epochs, verbose=1):
    vae.fit(train_data, epochs=max_epochs, verbose=verbose)

latent_dim = 4
sequence_length = train_conditions.shape[1]
feature_dim = 1

hyperparameters = {'latent_dim': latent_dim}
encoder, decoder = instantiate_vae_model(sequence_length, feature_dim, latent_dim)

class CVAEModel(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(layers.concatenate([z, inputs[:, :sequence_length]]))

        reconstruction_loss = tf.keras.losses.mse(inputs[:, sequence_length:], reconstruction)
        reconstruction_loss *= feature_dim

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= 0.001
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)

        return reconstruction

cvae_model = CVAEModel(encoder, decoder)
cvae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
train_vae(cvae_model, np.concatenate((train_conditions, train_target), axis=1), max_epochs=100, verbose=1)

import joblib

def save_scaler(scaler, dir_path):
    joblib.dump(scaler, os.path.join(dir_path, 'scaler.pkl'))

def save_vae_model(vae, dir_path):
    vae.save_weights(os.path.join(dir_path, 'vae_weights.h5'))

model_save_dir = 'saved_models/toydata'
os.makedirs(model_save_dir, exist_ok=True)
save_scaler(scaler, model_save_dir)
save_vae_model(cvae_model, model_save_dir)

def get_posterior_samples(vae, data):
    return vae.predict(data)

x_decoded = get_posterior_samples(cvae_model, np.concatenate((train_conditions, train_target), axis=1))

def plot_samples(samples1, samples1_name, samples2, samples2_name, num_samples):
    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        plt.plot(samples1[i], label=f'{samples1_name} {i+1}')
        plt.plot(samples2[i], linestyle='--', label=f'{samples2_name} {i+1}')
    plt.legend()
    plt.show()

plot_samples(
    samples1=scaled_train_data[:5, -1].reshape(-1, 1),
    samples1_name="Original Train",
    samples2=x_decoded[:5, -1].reshape(-1, 1),
    samples2_name="Reconstructed Train",
    num_samples=5
)

def get_prior_samples(vae, num_samples):
    sampled_latent = np.random.normal(size=(num_samples, latent_dim))
    generated_samples = []
    for i in range(num_samples):
        generated_samples.append(vae.decoder.predict(np.concatenate([sampled_latent[i].reshape(1, -1), train_conditions[i % len(train_conditions)].reshape(1, -1)], axis=1)))
    return np.array(generated_samples).squeeze()

prior_samples = get_prior_samples(cvae_model, train_data.shape[0])

plot_samples(
    samples1=prior_samples[:5],
    samples1_name="Prior Samples",
    num_samples=5
)

# Saving the samples
inverse_scaled_prior_samples = scaler.inverse_transform(prior_samples)
np.save('saved_models/toydata_prior_samples.npy', inverse_scaled_prior_samples)

from sklearn.manifold import TSNE

def visualize_and_save_tsne(samples1, samples1_name, samples2, samples2_name, scenario_name, save_dir, max_samples=2000):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result_1 = tsne.fit_transform(samples1[:max_samples])
    tsne_result_2 = tsne.fit_transform(samples2[:max_samples])

    plt.figure(figsize=(12, 6))
    plt.scatter(tsne_result_1[:, 0], tsne_result_1[:, 1], label=samples1_name, alpha=0.5)
    plt.scatter(tsne_result_2[:, 0], tsne_result_2[:, 1], label=samples2_name, alpha=0.5)
    plt.legend()
    plt.title(scenario_name)
    plt.savefig(os.path.join(save_dir, 'tsne_plot.png'))
    plt.show()

visualize_and_save_tsne(
    samples1=scaled_train_data,
    samples1_name="Original",
    samples2=prior_samples,
    samples2_name="Generated (Prior)",
    scenario_name=f"Model-CVAE Dataset-toydata",
    save_dir='saved_models/toydata',
    max_samples=2000
)

