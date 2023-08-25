from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

class Discriminator:
    def __init__(self):
        self.discriminator = self.build_discriminator()

    def build_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer='adam')
        return discriminator
