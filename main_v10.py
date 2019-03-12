from classes.neural_v10 import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

start = -2
end = 3
step = 0.05
y=np.zeros((101,1))
x=np.zeros((101,2))
x[:,0] = np.arange(start,end+step,step) # Vector from 0 - 1 
x[:,1] = np.arange(start,end+step,step) # Vector from 0 - 1                                                   
N = x.shape[0]
x_train=x[0:91,:]
x_test=x[91:102,:]
a, b, c = 1, 2, 3
noise = np.random.normal(0.2,0.4,(N,))
y = a * np.power (x [:, 0 ], 2 ) + b * x [:, 1 ] + c +  0 * noise
y_train=y[0:91]
y_test=y[91:102]
ne = 2
nm = 10
num_out = 1
layers = 1 
bias = True
act_fun = "sigmoid_1"

nn = NeuralNetwork(ne, x_train,x_test, num_out, y_train,y_test, layers, nm, bias, act_fun)

learn_rate = 0.1
max_iter = 10000

yd, J, Iter =nn.execute(learn_rate,max_iter)

plt.figure(1)
plt.plot(x,y,'.g')#data real
plt.plot(x_train[:,1],y_train,'*k')#entrenamiento de la red
plt.plot(x_test[:,1],yd,'-r')#pronostico
plt.figure(2)
plt.plot(np.arange(Iter),J)
plt.show()