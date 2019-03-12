import numpy as np
import random
import sys

class NeuralNetwork:
    """A simple example class"""

    # init_weight = 0.15 # initial weights of neuron connections
    J_old = 1e15 # initial value of the function to be minimized

    def __init__(self, num_inputs, input_data,X_test, num_outputs, output_data,y_test, layers, layer_neurons, bias, activation_function):
        self.N = input_data.shape[0] # data length
        self.N_test=X_test.shape[0]
        self.ne = num_inputs
        self.in_data = np.resize(input_data,(self.N,2))
        self.in_data_test=np.resize(X_test,(self.N_test,2))
        self.no = num_outputs
        self.out_data = output_data
        self.out_data_test=y_test
        self.layers = layers
        self.nm = layer_neurons
        if bias:
            self.ne += 1
            self.in_data = np.column_stack((self.in_data,np.ones(self.N)))
            self.in_data_test = np.column_stack((self.in_data_test,np.ones(self.N_test)))
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
        Y_test=self.out_data_test
        yd = np.zeros(self.N_test)
        error = np.zeros(self.N)
        error_test= np.zeros(self.N_test)
        J = np.zeros(max_iter)
        J_test= np.zeros(max_iter)
        for i in range(max_iter):
            dJdv = np.zeros((self.ne,self.nm))
            dJdw = np.zeros((self.nm,self.no))
            for k in range(self.N):
                In = self.in_data[k,:]
                m = self.get_v().T.dot(In)
                n = self.f(m)
                out = self.get_w().T.dot(n)
                #yd[k] = out # falta corregir para mas salidas
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
        for kk in range(self.N_test):
            Int = self.in_data_test[kk,:]
            mt = self.get_v().T.dot(Int)
            nt = self.f(mt)
            outt = self.get_w().T.dot(nt)
            yd[kk] = outt
            err_test = outt - Y_test[kk]
            error_test[kk] = err_test
            Jj = 0.5*np.sum(np.multiply(error,error))
        return yd, J, i