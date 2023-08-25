from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape

class Generator:
    def __init__(self):
        self.generator = self.build_generator()

    def build_generator(self):
        generator = Sequential()
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
        return generator
