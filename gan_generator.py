import numpy as np
from tensorflow.python.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from numpy.random import randint


# using GAN method to fake data for training
# purpose we have 1% subsamples. we want more data for training to get
# hight AUC by default config
# basic defind the model has 4 layers, 1 input, 2 hiddens and last layers as
# binary classification by Dense
def define_discriminator_model(image_shape=(28, 28, 1)):
  model = models.Sequential()
  model.add(
          layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                        input_shape=image_shape))
  model.add(layers.LeakyReLU(alpha=0.2))
  model.add(layers.Dropout(0.4))
  model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU(alpha=0.2))
  model.add(layers.Dropout(0.4))
  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))
  # compile model
  opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model
  return model


def generate_real_samples(images, lables, n_samples):
  rand_idx = randint(0, images.shape[0], n_samples)
  rand_images = images[rand_idx]
  rand_labels = np.ones((n_samples, 1))
  return rand_images, rand_labels


def generate_fake_samples(n_samples):
  init_rand_images = np.rand(28 * 28 * n_samples) \
    .reshape((n_samples, 28, 28, 1))
  labels = np.zeros((n_samples, 1))
  return init_rand_images, labels


# defined the generator method to init fake images
def define_generator_model(latent_dim):
  model = models.Sequential()
  numb_nodes = 128 * 7 * 7
  model.add(layers.Dense(numb_nodes, input_dim=latent_dim))
  model.add(layers.LeakyReLU(alpha=0.2))
  model.add(layers.Reshape((7, 7, 128)))
  # upsample to 14x14
  model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU(alpha=0.2))
  # upsample to 28x28
  model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU(alpha=0.2))
  model.add(layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
  return model


def generate_latent_points(latent_dim, numb_samples):
  return np.random.randn(latent_dim * numb_samples) \
    .reshape(numb_samples, latent_dim)


# using generator to generate n fake samples, with class labels:
def generate_fake_samples(g_model, latent_dim, numb_samples):
  input = generate_latent_points(latent_dim, numb_samples)
  X = g_model.predict(input)
  y = np.zeros((numb_samples, 1))
  return X, y


def define_gan(g_model, d_model):
  model = models.Sequential()
  model.add(g_model)
  model.add(d_model)
  opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model


def train_gan(g_model, d_model, gan_model, data, label,
        validate_data, validate_label,
        latent_dim,
        epochs=10,
        numb_batch=12):
  batch_per_epo = int(data.shape[0] / numb_batch)

  haft_data_point_batch = int(batch_per_epo / 2)
  for i in range(epochs):
    for j in range(batch_per_epo):
      x_real, y_real = generate_real_samples(data, label, haft_data_point_batch)
      x_fake, y_fake = generate_fake_samples(g_model, latent_dim,
                                             haft_data_point_batch)
      x, y = np.vstack((x_fake, x_real)), np.vstack((y_fake, y_real))

      d_loss, _ = d_model.train_on_batch(x, y)

      x_gan = generate_latent_points(latent_dim, numb_batch)

      y_gan = np.ones((numb_batch, 1))

      g_loss = gan_model.train_on_batch(x_gan, y_gan)
      print('>%d, %d/%d, d=%.3f, g=%.3f' % (
        i + 1, j + 1, batch_per_epo, d_loss, g_loss))
      # evaluate the model performance, sometimes
    summarize_performance(i, g_model, d_model, validate_data, validate_label,
                          latent_dim)


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, validate_label,
        latent_dim,
        n_samples=100):
  X_real, y_real = generate_real_samples(dataset, validate_label, n_samples)
  _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
  x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
  print('>Accuracy real: %.0f%%, fake: %.0f%%' % (
    acc_real * 100, acc_fake * 100))
  save_plot(x_fake, epoch)
  filename = 'generator_model_%03d.h5' % (epoch + 1)
  g_model.save_weights("/home/shjfu/vinid/weight/" + filename)


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
  # plot images
  for i in range(n * n):
    # define subplot
    plt.subplot(n, n, 1 + i)
    # turn off axis
    plt.axis('off')
    # plot raw pixel data
    plt.imshow(examples[i, :, :, 0], cmap='gray_r')
  # save plot to file
  filename = 'generated_plot_e%03d.png' % (epoch + 1)
  plt.savefig(filename)
  plt.close()
