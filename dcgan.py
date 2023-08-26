import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow as tf

from data_loader import DataLoader
from discriminator import Discriminator
from generator import Generator

class DCGAN:
    def __init__(self, architecture):
        self.random_dim = 100
        self.data_loader = DataLoader(architecture)
        self.discriminator = Discriminator(architecture).discriminator
        self.generator = Generator(architecture).generator
        self.discriminator.trainable = False
        self.gan_input = Input(shape=(self.random_dim,))
        self.x = self.generator(self.gan_input)
        self.gan_output = self.discriminator(self.x)
        self.gan = Model(self.gan_input, self.gan_output)
        self.gan.compile(loss='binary_crossentropy', optimizer='adam')

    def train_gan(self, epochs=1, batch_size=128):
        batch_count = self.data_loader.X_train.shape[0] // batch_size

        for e in range(epochs + 1):
            for _ in range(batch_count):
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                generated_images = self.generator.predict(noise)
                image_batch = self.data_loader.X_train[np.random.randint(0, self.data_loader.X_train.shape[0], size=batch_size)]

                # Discriminator training
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(image_batch, np.ones(batch_size))
                d_loss_fake = self.discriminator.train_on_batch(generated_images, np.zeros(batch_size))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Generator training
                noise = np.random.normal(0, 1, size=[batch_size, self.random_dim])
                self.discriminator.trainable = False
                g_loss = self.gan.train_on_batch(noise, np.ones(batch_size))

            print(f'Epoch {e}, Discriminator: {d_loss}, Generator: {g_loss}')

            if e % 10 == 0:
                self.plot_generated_images(e)

    def plot_generated_images(self, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
        noise = np.random.normal(0, 1, size=[examples, self.random_dim])
        generated_images = self.generator.predict(noise)
        generated_images = generated_images.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'gan2_generated_image_epoch_{epoch}.png')
