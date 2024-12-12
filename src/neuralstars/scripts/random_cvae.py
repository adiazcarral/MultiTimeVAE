import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, backend as K
import matplotlib.pyplot as plt
import os

# Suppress oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset
data = pd.read_csv('toydata.csv')

# Assume the columns are named 'obs_p', 'obs_teta', and 'obs_q'
obs_p = data['obs_p'].values.reshape(-1, 1)
obs_teta = data['obs_teta'].values.reshape(-1, 1)
obs_q = data['obs_q'].values.reshape(-1, 1)

# Concatenate features for the condition
conditions = np.concatenate((obs_p, obs_teta), axis=1)

# The target variable we want to generate
target = obs_q

# Normalize data
mean_conditions = np.mean(conditions, axis=0)
std_conditions = np.std(conditions, axis=0)
conditions = (conditions - mean_conditions) / std_conditions

mean_target = np.mean(target, axis=0)
std_target = np.std(target, axis=0)
target = (target - mean_target) / std_target

# Split the dataset into two halves
split_index = len(conditions) // 2
train_conditions = conditions[split_index:]
train_target = target[split_index:]

# Define the latent dimension
latent_dim = 4  # Increasing the latent dimension

# Encoder
inputs = layers.Input(shape=(train_conditions.shape[1] + train_target.shape[1],))
x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(inputs)
x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Specify the output shape for the Lambda layer
z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_inputs = layers.Input(shape=(latent_dim + train_conditions.shape[1],))
x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(decoder_inputs)
x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
outputs = layers.Dense(train_target.shape[1], activation='linear')(x)

# Define models
encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = models.Model(decoder_inputs, outputs, name='decoder')

# CVAE model
cvae_inputs = layers.Input(shape=(train_conditions.shape[1] + train_target.shape[1],))
z_mean, z_log_var, z = encoder(cvae_inputs)
decoder_input = layers.concatenate([z, cvae_inputs[:, :train_conditions.shape[1]]])
cvae_outputs = decoder(decoder_input)
cvae = models.Model(cvae_inputs, cvae_outputs, name='cvae')

# Custom CVAE model with loss addition
class CVAEModel(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(layers.concatenate([z, inputs[:, :train_conditions.shape[1]]]))

        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.mse(inputs[:, train_conditions.shape[1]:], reconstruction)
        reconstruction_loss *= train_target.shape[1]

        # KL divergence loss
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
        
        # Scale KL loss to balance it with reconstruction loss
        kl_loss *= 0.001  # Adjust this factor as needed
        
        # Total loss
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)

        return reconstruction

# Instantiate and compile the CVAE model
cvae_model = CVAEModel(encoder, decoder)
cvae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Training the model
cvae_model.fit(np.concatenate((train_conditions, train_target), axis=1), train_target, epochs=100, batch_size=16)  # Increase epochs and reduce batch size

# Generate predictions for the last 100 values from the second half of the dataset
last_100_conditions = train_conditions[-100:]
last_100_target = train_target[-100:]

# Encode and sample from the latent space for the last 100 samples
z_mean, z_log_var, _ = encoder.predict(np.concatenate((last_100_conditions, last_100_target), axis=1))
sampled_z = sampling([z_mean, z_log_var])
generated_obs_q = decoder.predict(np.concatenate((sampled_z, last_100_conditions), axis=1))

# Rescale the generated values back to the original range
generated_obs_q = generated_obs_q * std_target + mean_target

# Plotting the comparison
plt.figure(figsize=(10, 6))
plt.plot(range(100), last_100_target * std_target + mean_target, label='Actual obs_q')
plt.plot(range(100), generated_obs_q, label='Generated obs_q', linestyle='--')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Streamflow')
plt.title('Comparison of Actual and Generated obs_q')
plt.show()

# Inspecting latent space variability
samples = np.random.normal(size=(10, latent_dim))
for i, sample in enumerate(samples):
    generated_q_sample = decoder.predict(np.concatenate([sample.reshape(1, -1), last_100_conditions[i % 100].reshape(1, -1)], axis=1))
    print(f"Sample {i+1} generated_obs_q: {generated_q_sample}")
