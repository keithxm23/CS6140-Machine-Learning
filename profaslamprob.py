from ann import *

#Defining perceptrons
p1 = Perceptron()
p1.w_vector = [1,1]
p1.w_old = p1.w_vector
p1.bias = 1
p1.lrate = 1.0

p2 = Perceptron()
p2.w_vector = [2,2]
p2.w_old = p2.w_vector
p2.bias = 2
p2.lrate = 1.0

p3 = Perceptron()
p3.w_vector = [3,3]
p3.w_old = p3.w_vector
p3.bias = 3
p3.lrate = 1.0

#Defining layers
l1 = Layer()
l1.ptrons = [p1, p2]
l1.x_vector = [-3,1]

l2 = Layer()
l2.ptrons = [p3]

#Defining NeuralNetwork
nnet = Nnet()
nnet.layers=[l1,l2]
nnet.labels=[0]

#print l1.feed_forward()
nnet.begin()
pass