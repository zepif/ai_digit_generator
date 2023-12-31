from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Discriminator:
    def __init__(self, architecture='gan1'):
        self.img_shape = (28, 28, 1)

        self.discriminator = self.build_discriminator(architecture)

    def build_discriminator(self, architecture):
        discriminator = Sequential()

        if architecture == 'gan1':
            discriminator.add(Dense(1024, input_dim=784))
            discriminator.add(LeakyReLU(0.2))
            discriminator.add(Dense(512))
            discriminator.add(LeakyReLU(0.2))
            discriminator.add(Dense(256))
            discriminator.add(LeakyReLU(0.2))
            discriminator.add(Dense(1, activation='sigmoid'))
            discriminator.compile(loss='binary_crossentropy', optimizer='adam')
        elif architecture == 'gan2':
            discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(28, 28, 1)))
            discriminator.add(LeakyReLU(0.2))
            discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
            discriminator.add(LeakyReLU(0.2))
            discriminator.add(Flatten())
            discriminator.add(Dense(1, activation='sigmoid'))
            discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(0.0002, 0.5),
                                   metrics=['accuracy'])
        else:
            raise ValueError("Invalid architecture name. Use 'gan1' or 'gan2'.")
        
        return discriminator
