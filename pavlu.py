from ann import *


#Defining layers
inputs_and_labels = [
          [1,0,0,0,0,0,0,0],
          [0,1,0,0,0,0,0,0],
          [0,0,1,0,0,0,0,0],
          [0,0,0,1,0,0,0,0],
          [0,0,0,0,1,0,0,0],
          [0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,1,0],
          [0,0,0,0,0,0,0,1],
          ]
l1 = Layer([rand(-0.05,0.05) for r in xrange(3)], x_vector=inputs_and_labels)
l2 = Layer([rand(-0.05,0.05) for r in xrange(8)], below_layer=l1)

#Defining NeuralNetwork
nnet = Nnet()
nnet.layers=[l1,l2]
nnet.labels=inputs_and_labels
#print l1.feed_forward()
nnet.begin(0.000001)#supply threshold

#for p in nnet.layers[0].ptrons:
#    print p.output
#print "----------------"
#for p in nnet.layers[1].ptrons:
#    print p.output

#Feed input data point number 0
input_number = 0
for i in xrange(len(nnet.layers)):
    xvec = nnet.layers[i].feed_forward(input_number)
    try:
        nnet.layers[i+1].x_vector[input_number] = xvec
    except IndexError:
        print xvec
        break #reached output layer
    
pass