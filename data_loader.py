import numpy as np
from tensorflow.keras.datasets import mnist

class DataLoader:
    def __init__(self, architecture='gan1'):
        self.X_train = self.load_data(architecture)

    def load_data(self, architecture):
        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train / 127.5 - 1.0
        #print(np.unique(X_train))
        if (architecture == 'gan1'):
            X_train = X_train.reshape(X_train.shape[0], 784)
        elif (architecture == 'gan2'):
            X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        else:
            raise ValueError("Invalid architecture name. Use 'gan1' or 'gan2'.")
        
        return X_train
