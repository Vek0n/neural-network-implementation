import numpy as np
import random

class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        # self.output = np.matmul(self.W, input_data) + self.b 
        self.output = np.dot(self.input, self.W) + self.b
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)

        self.W -= learning_rate * weights_gradient
        self.b -= learning_rate * output_gradient

        return np.dot(output_gradient, self.W.T) #input_gradient