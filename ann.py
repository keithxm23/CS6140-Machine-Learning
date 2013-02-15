from math import exp
from utils import MSE

class Perceptron():#TODO add compulsory vars as keyword args
    bias = None
    output = None
    w_vector = [] #weight vector
    error = None
    lrate = None
    
    def weighted_sum(self, x_vector):
        wsum = 0
        for x, w in zip(x_vector, self.w_vector):
            wsum += w*x
        wsum+= self.bias
        return wsum
    
    def sigmoid(self, x_vector): #activation function
        return 1/(1+exp(self.weighted_sum(x_vector)*-1))
        
    
class Layer():
    ptrons = []
    x_vector = [] #input vector
    
    def feed_forward(self):
        xvec = []
        for p in self.ptrons:
            p.output = p.sigmoid(self.x_vector)
            xvec.append(p.output)
        return xvec
    
    def back_propagate(self, above_layer, below_layer):
        for p in self.ptrons:
            error = p.output
            error *= (1 - p.output)
            err_j = 0
            for i, q in enumerate(above_layer.ptrons):
                err_j += q.error*q.w_vector[i]
            error *= err_j
            p.error = error
            
            for w in xrange(p.w_vector):
                p.w_vector[w] += p.lrate * p.error * below_layer[w].output
            


class Nnet():
    layers = []
    labels = []
    
    def begin(self):
        
        #Feed forward
        for i in xrange(len(self.layers)):
            xvec = self.layers[i].feed_forward()
            try:
                self.layers[i+1].x_vector = xvec
            except IndexError:
                print "reached top"
                print xvec
                break #reached output layer
            
        #Check if error has converged
        err = MSE(xvec,self.labels)
        print err
        #TODO: check convergence
        
        #If not, Now Back propagate
        for p in self.layers[-1].ptrons: #updating errors of output layer perceptrons
            error = p.output
            error *= (1 - p.output)
            index = self.layers[-1].ptrons.index(p)
            error *= (self.labels[index] - p.output)
            p.error = error
            
            for w in xrange(len(p.w_vector)):
                p.w_vector[w] += p.lrate * p.error * self.layers[-2].ptrons[w].output
            
        for i in xrange(len(self.layers)-2, -1, -1): #looping from index of output_layer-1 to 0th layer
            if i > 0:#updating errors of hidden layer perceptrons
                self.layers[i].back_propagate(self.layers[i+1], self.layers[i-1])
            else:#updating error for (i=)0-th layer (last hidden layer)
                for p in self.layers[0].ptrons:
                    error = p.output
                    error *= (1 - p.output)
                    err_j = 0
                    for i, q in enumerate(self.layers[1].ptrons):
                        err_j += q.error*q.w_vector[i]
                    error *= err_j
                    p.error = error
                    
                    for w in xrange(len(p.w_vector)):
                        p.w_vector[w] += p.lrate * p.error * self.layers[0].x_vector[w]
            