import numpy as np
import pandas as pd
from Dense import Dense
from ActivationLayer import ActivationLayer
from Network import Network
from Activations import tanh, tanh_prime
from losses import mse, mse_prime
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":

    df = pd.read_csv (r'input.csv')
    df['species'] = df['species'].replace(['Iris-setosa'],0)
    df['species'] = df['species'].replace(['Iris-versicolor'],1)
    df['species'] = df['species'].replace(['Iris-virginica'],2)
    df = df.sample(frac=1)
    raw_data = df.to_numpy()

    x_train = np.delete(raw_data,-1, axis=-1)
    y_train = np.delete(raw_data,[0,1,2,3], axis=-1)
    y_train = to_categorical(y_train)

    x_train = x_train.reshape(x_train.shape[0], 1, 4)
    x_train = x_train.astype('float32')

    # Network
    net = Network()
    net.add(Dense(4,20))               
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(Dense(20, 3))                    
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)
    net.fit(x_train[0:150], y_train[0:150], epochs=100, learning_rate=0.1)

    out = net.predict(x_train[0:3])
    print("\n")
    print("predicted values : ")
    print(out, end="\n")
    print("true values : ")
    print(y_train[0:3])