from classes.neural import NeuralNetwork
#import numpy as np

ne = 2
nm = 5
layers = 1 
bias = True
act_fun = "sigmoid_1"
x = NeuralNetwork(ne, layers, nm, bias, act_fun)

#print("Weights1: ", x.weight_u())
#print("Weights2: ", x.weight_w())
print(x.execute())
