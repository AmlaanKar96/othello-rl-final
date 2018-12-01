import math
import numpy as np
import random
import pickle

class NN:
    def __init__(self, dimensions, lr):
        self.lr = lr
        self.dimensions = dimensions
        self.layers = []
        for i in range(len(dimensions) - 1):
            self.layers.append(np.random.normal(0, 1, size=(dimensions[i+1], dimensions[i]+1)))
        self.activation_func  = None
        self.deactivation_func = None

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.layers, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.layers = pickle.load(f)

    def mkVec(self, vector1D, add_bias = True):
        return np.reshape(vector1D, (len(vector1D), 1))

    def forward_pass(self, inputs):
        activations = inputs
        for i in range(len(self.layers)):
            activations = activation(self.layers[i].dot(np.vstack((activations, 1))))
        return activations

    def backProp(self, sample, target):
        activations = [sample]
        for i in range(len(self.layers)):
            activations.append(activation(self.layers[i].dot(np.vstack((activations[i], 1)))))
        d_layers = np.empty(len(activations) - 1, object)
        d_layers[-1] = (target - activations[-1]) * deactivation(activations[-1])
        for i in range(len(d_layers) - 2, -1, -1):
            layer_derivative = deactivation(activations[i+1])
            d_layers[i] = layer_derivative * (self.layers[i+1].T.dot(d_layers[i + 1])[:-1])
        for i in range(len(self.layers)):
            self.layers[i] += self.lr * np.c_[activations[i].T, 1] * d_layers[i]
        return activations[-1]

def activation(x):
    return np.exp(-x)


def deactivation(x):
    return 1 / (1 + np.exp(-x))
