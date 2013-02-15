from ann import *


#Defining layers
l1 = Layer([rand(-0.05,0.05) for r in xrange(3)], x_vector=[1,0,0,0,0,0,0,0])
l2 = Layer([rand(-0.05,0.05) for r in xrange(8)], below_layer=l1)

#Defining NeuralNetwork
nnet = Nnet()
nnet.layers=[l1,l2]
nnet.labels=[1,0,0,0,0,0,0,0]
#print l1.feed_forward()
nnet.begin(0.00001)#supply threshold

for p in nnet.layers[0].ptrons:
    print p.output
print "----------------"
for p in nnet.layers[1].ptrons:
    print p.output

pass