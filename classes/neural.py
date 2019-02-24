import numpy as np
import random
import sys

class NeuralNetwork:
    """A simple example class"""
    J_old = 1e15 # initial value of the function to be minimized

    def __init__(self, num_inputs, input_data, num_outputs, output_data, layers, layer_neurons, bias, activation_function):
        self.N = input_data.shape[0] # data length
        self.ne = num_inputs
        self.in_data = np.resize(input_data,(self.N,1))
        self.no = num_outputs
        self.out_data = output_data
        self.layers = layers
        self.nm = layer_neurons
        if bias:
            self.ne += 1
            self.in_data = np.column_stack((self.in_data,np.ones(self.N)))
        self.fun = activation_function

    def set_v(self, init_value = 0.15):
        # self.v = np.random.normal(init_value, 0.6*init_value, (self.ne,self.nm))
        self.v = init_value*np.random.random((self.ne,self.nm))

    def get_v(self):
        return self.v

    def set_w(self, init_value = 0.15):
        # self.w = np.random.normal(init_value, 0.6*init_value, (self.nm,self.no))
        self.w = init_value*np.random.random((self.nm,self.no))

    def get_w(self):
        return self.w

    def f(self, m):
        if self.fun == "sigmoid_1":
            return 1/(1 + np.exp(-m))
        elif self.fun == "sigmoid_2":
            return 2/(1 + np.exp(-m)) - 1
        elif self.fun == "gaussian":
            return np.exp(-np.power(m,2))
        else:
            sys.exit("Wrong activation function - check name")

    def dndm(self, n, m):
        if self.fun == "sigmoid_1":
            return n*(1-n)
        elif self.fun == "sigmoid_2":
            return (1-n*n)/2
        elif self.fun == "gaussian":
            return -2*(n*m)
        else:
            sys.exit("Wrong activation function - check name")

    def execute(self, rate = 0.1, max_iter = 10000):
        self.set_v()
        self.set_w()
        y = self.out_data
        yd = np.zeros(self.N)
        error = np.zeros(self.N)
        J = np.zeros(max_iter)
        for i in range(max_iter):
            dJdv = np.zeros((self.ne,self.nm))
            dJdw = np.zeros((self.nm,self.no))
            for k in range(self.N):
                In = self.in_data[k,:]
                m = self.get_v().T.dot(In)
                n = self.f(m)
                out = self.get_w().T.dot(n)
                yd[k] = out # falta corregir para mas salidas
                err = out - y[k]
                error[k] = err
                dJdw = dJdw + np.outer(n,err) # falta corregir para tamano variable
                e2 = np.multiply(self.get_w().dot(err),self.dndm(n,m))
                dJdv = dJdv + np.outer(In,e2)
            
            self.v = self.v - rate*dJdv/self.N
            self.w = self.w - rate*dJdw/self.N
            Jj = 0.5*np.sum(np.multiply(error,error))
            print(i, Jj)
            dJ = np.abs(Jj - self.J_old)
            dJ_per = np.sqrt(dJ/Jj)*100
            if dJ_per < 1:
                #print(i)
                pass #break
            J[i] = Jj
            self.J_old = Jj
            # print("\n")
        J = np.resize(J,i)  
        return yd, J, i