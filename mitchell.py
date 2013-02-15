from ann import *


#Defining layers
l1 = Layer([-0.4,0.2], x_vector=[1,0,1], w_vector=[[0.2, 0.4, -0.5],[-0.3, 0.1, 0.2]])
l2 = Layer([0.1], below_layer=l1, w_vector=[[-0.3,-0.2]])

#Defining NeuralNetwork
nnet = Nnet()
nnet.layers=[l1,l2]
nnet.labels=[1]

#print l1.feed_forward()
nnet.begin(0.1)
for p in nnet.layers[0].ptrons:
    print p.output
print "----------------"
for p in nnet.layers[1].ptrons:
    print p.output
pass