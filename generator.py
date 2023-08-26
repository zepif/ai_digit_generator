from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Generator:
    def __init__(self, architecture='gan1'):
        self.random_dim = 100

        self.generator = self.build_generator(architecture)

    def build_generator(self, architecture):
        generator = Sequential()

        if architecture == 'gan1':
            generator.add(Dense(256, input_dim=100))
            generator.add(LeakyReLU(0.2))
            generator.add(BatchNormalization(momentum=0.8))
            generator.add(Dense(512))
            generator.add(LeakyReLU(0.2))
            generator.add(BatchNormalization(momentum=0.8))
            generator.add(Dense(1024))
            generator.add(LeakyReLU(0.2))
            generator.add(BatchNormalization(momentum=0.8))
            generator.add(Dense(784, activation='tanh'))
            generator.compile(loss='binary_crossentropy', optimizer='adam')
        elif architecture == 'gan2':
            generator.add(Dense(128 * 7 * 7, input_dim=self.random_dim))
            generator.add(Reshape((7, 7, 128)))
            generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
            generator.add(BatchNormalization())
            generator.add(LeakyReLU(alpha=0.01))
            generator.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
            generator.add(BatchNormalization())
            generator.add(LeakyReLU(alpha=0.01))
            generator.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh'))
        else:
            raise ValueError("Invalid architecture name. Use 'gan1' or 'gan2'.")

        return generator
