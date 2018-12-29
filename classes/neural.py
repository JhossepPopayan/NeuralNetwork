import numpy as np
import random
import sys

class NeuralNetwork:
    """A simple example class"""

    init_weight = 0.15

    def __init__(self, inputs, layers, layer_neurons, bias, activation_function):
        self.ne = inputs
        self.nm = layer_neurons
        self.layers = layers
        if bias:
            self.ne += 1
        self.fun = activation_function
    
    def weight_u(self):
        #return self.init_weight*np.random.random((self.ne, self.nm))
        return np.random.normal(self.init_weight, 0.607*self.init_weight, (self.ne,self.nm))
 
    def weight_w(self):
        #return self.init_weight*np.random.random((self.ne, self.nm))
        return np.random.normal(self.init_weight, 0.607*self.init_weight, (self.nm,1))
    
    def f(self, m):
        if self.fun == "sigmoid_1":
            return 1/(1 + np.exp(-m))
        elif self.fun == "sigmoid_2":
            return 2/(1 + np.exp(-m)) - 1
        elif self.fun == "gaussian":
            return np.exp(-np.power(m,2))
        else:
            sys.exit("Wrong activation function - check name")

    def execute(self):
        m = -1
        f = self.f
        return f(m)