from ann import *

#Defining perceptrons
p1 = Perceptron()#theta4
p1.w_vector = [0.2, 0.4, -0.5]
p1.w_old = p1.w_vector
p1.bias = -0.4
p1.lrate = 0.9

p2 = Perceptron()#theta5
p2.w_vector = [-0.3, 0.1, 0.2]
p2.w_old = p2.w_vector
p2.bias = 0.2
p2.lrate = 0.9

p3 = Perceptron()#theta6
p3.w_vector = [-0.3,-0.2]
p3.w_old = p3.w_vector
p3.bias = 0.1
p3.lrate = 0.9

#Defining layers
l1 = Layer()
l1.ptrons = [p1, p2]
l1.x_vector = [1,0,1]

l2 = Layer()
l2.ptrons = [p3]

#Defining NeuralNetwork
nnet = Nnet()
nnet.layers=[l1,l2]
nnet.labels=[1]

#print l1.feed_forward()
nnet.begin()
pass