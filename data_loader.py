import numpy as np
from tensorflow.keras.datasets import mnist

class DataLoader:
    def __init__(self):
        self.X_train = self.load_data()

    def load_data(self):
        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train / 127.5 - 1.0
        X_train = X_train.reshape(X_train.shape[0], 784)
        return X_train
