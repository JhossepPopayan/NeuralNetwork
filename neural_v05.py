import numpy as np
import random
import sys

class NeuralNetwork:
    """A simple example class"""

    # init_weight = 0.15 # initial weights of neuron connections
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
            #self.ne += 1
            self.in_data = np.column_stack((self.in_data,np.ones(self.N)))
        self.fun = activation_function

    def set_w1(self, init_value = 0.15):
        # self.v = np.random.normal(init_value, 0.6*init_value, (self.ne,self.nm))
        self.w1 = init_value*np.random.random((self.ne,self.nm))

    def get_w1(self):
        return self.w1

    def set_w2(self, init_value = 0.15):
        # self.w = np.random.normal(init_value, 0.6*init_value, (self.nm,self.no))
        self.w2 = init_value*np.random.random((self.nm,self.no))

    def get_w2(self):
        return self.w2

    def sigmoid(self, m,first_derivative=False):
        if self.fun == "sigmoid_1":
            return 1/(1 + np.exp(-m))
        elif self.fun == "sigmoid_2":
            return 2/(1 + np.exp(-m)) - 1
        elif self.fun == "gaussian":
            return np.exp(-np.power(m,2))
        else:
            sys.exit("Wrong activation function - check name")

    def tanh(self, n, first_derivative=True):
        if self.fun == "sigmoid_1":
            return n*(1-n)
        elif self.fun == "sigmoid_2":
            return (1-n*n)/2
        else:
            sys.exit("Wrong activation function - check name")

    def inference(self,data, weights): #CALCULO DEL VALOR PREDICHO
        h1 = self.sigmoid(np.matmul(data, weights[0]))
        yd = np.matmul(h1, weights[1])
        return yd
    
    def execute(self, rate , max_iter):
        self.set_w1()
        self.set_w2()
        y=self.out_data
        X=self.in_data
        NN=int(X.shape[0])
        reg_coeff=1e-6
        losses = []
        accuracies=[]
        yd = np.zeros((X.shape[0],X.shape[1]))
        # Initialize weights:
        np.random.seed(2017)
        w1 = 2.0*np.random.random((self.ne, self.nm))-1.0      #w0=(2,self.nm)
        w2 = 2.0*np.random.random((self.nm, self.no))-1.0     #w1=(self.nm,2)
        #Calibratring variances with 1/sqrt(fan_in)
        w1 /= np.sqrt(self.ne)
        w2 /= np.sqrt(self.nm)
        for i in range(max_iter):

            index = np.arange(X.shape[0])[:NN]
            #is want to shuffle indices: np.random.shuffle(index)
            
            #---------------------------------------------------------------------------------------------------------------
            # Forward step:
            h1 = self.sigmoid(np.matmul(X[index], w1))                   #(N, 3)
            logits = self.sigmoid(np.matmul(h1, w2))                     #(N, 2)
            probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
            h2 = logits.T

            #---------------------------------------------------------------------------------------------------------------
            # Definition of Loss function: mean squared error plus Ridge regularization
            L = np.square(y[index]-h2).sum()/(2*NN) + reg_coeff*(np.square(w1).sum()+np.square(w2).sum())/(2*NN)

            #losses.append([i,L])
        
            #---------------------------------------------------------------------------------------------------------------
            # Backward step: Error = W_l e_l+1 f'_l
            #       dL/dw2 = dL/dh2 * dh2/dz2 * dz2/dw2
            dL_dh2 = -(y[index] - h2)                               #(N, 2)
            dh2_dz2 = self.sigmoid(h2, first_derivative=True)            #(N, 2)
            dz2_dw2 = h1                                            #(N, self.nm)
            #Gradient for weight2:   (self.nm,N)x(N,2)*(N,2)
            dL_dw2 = dz2_dw2.T.dot(dL_dh2.T*dh2_dz2.T) + reg_coeff*np.square(w2).sum()

            #dL/dw1 = dL/dh1 * dh1/dz1 * dz1/dw1
            #       dL/dh1 = dL/dz2 * dz2/dh1
            #       dL/dz2 = dL/dh2 * dh2/dz2
            
            
            dL_dz2 = dL_dh2 * dh2_dz2                               #(N, 2)
            #dL_dz2=dL_dz2[:,0:1]
            dz2_dh1 = w2                                            #z2 = h1*w2
            dL_dh1 =  dL_dz2.T.dot(dz2_dh1.T)                         #(N,2)x(2, self.nm)=(N, hidden_dim)
            dh1_dz1 = self.sigmoid(h1, first_derivative=True)            #(N,self.nm)
            dz1_dw1 = X[index]                                      #(N,2)
            #Gradient for weight1:  (2,N)x((N,self.nm)*(N,self.nm))
            dL_dw1 = dz1_dw1.T.dot(dL_dh1*dh1_dz1) + reg_coeff*np.square(w1).sum()

            #weight updates:
            w2 += -rate*dL_dw2
            w1 += -rate*dL_dw1
            print(w2)
'''
if __name__ == '__main__':
    start = -2
    end = 3
    step = 0.05
    x=np.zeros((101,2))
    x[:,0] = np.arange(start,end+step,step) # Vector from 0 - 1 
    x[:,1] = np.arange(start,end+step,step) # Vector from 0 - 1                                                                 
    N = x.shape[0]
    a, b, c = 1, 2, 3
    noise = np.random.normal(0.2,0.4,(N,))
    y = a*np.power(x[:,0],2) + b*x[:,1] + c + 0*noise
    ne = 2
    nm = 3
    num_out = 2
    layers = 1
    bias = True
    act_fun = "sigmoid_1"

    nn = NeuralNetwork(ne, x, num_out, y, layers, nm, bias, act_fun)

    learn_rate = 1e-3
    max_iter = 1000000

    nn.execute(learn_rate,max_iter)
'''