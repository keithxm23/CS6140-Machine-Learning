from math import exp
from utils import MSE, is_converged
from random import uniform as rand

class Perceptron():#TODO add compulsory vars as keyword args
    bias = None
    output = None
    w_vector = [] #weight vector
    w_old = []
    error = None
    lrate = 0.9#None TODO randomize this
    
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
    
    def __init__(self, ptron_biases, below_layer=None, x_vector=None, w_vector=None):
        self.ptrons = []
        for i, b in enumerate(ptron_biases):
            p = Perceptron()
            p.bias = b
            if below_layer == None:
                if w_vector == None:
                    p.w_vector = [rand(-1,1) for r in xrange(len(x_vector))]
                else:
                    p.w_vector = w_vector[i]
                p.w_old = p.w_vector
            else:
                if w_vector == None:
                    p.w_vector = [rand(-1,1) for r in xrange(len(x_vector))]
                else:
                    p.w_vector = w_vector[i]
                p.w_old = p.w_vector
            self.ptrons.append(p)
        
        if x_vector != None:
            self.x_vector = x_vector
    
    
    
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
#            print error
            
            p.w_old = p.w_vector
            for w in xrange(p.w_vector):
                p.w_vector[w] += p.lrate * p.error * below_layer[w].output
            
            p.bias += p.bias * p.lrate * p.error * 1


class Nnet():
    layers = []
    labels = []
    
    def begin(self):
        prev_error = 0
        curr_error = float("inf")
        thresh = 0.1
        while(not is_converged(prev_error, curr_error, thresh)):

            #Feed forward
            for i in xrange(len(self.layers)):
                xvec = self.layers[i].feed_forward()
                try:
                    self.layers[i+1].x_vector = xvec
                except IndexError:
                    break #reached output layer
                
            #Check if error has converged
            prev_error = curr_error
            curr_error = MSE(xvec,self.labels)
            print curr_error
            #TODO: check convergence
            
            #If not, Now Back propagate
            for p in self.layers[-1].ptrons: #updating errors of output layer perceptrons
                error = p.output
                error *= (1 - p.output)
                index = self.layers[-1].ptrons.index(p)
                error *= (self.labels[index] - p.output)
                p.error = error
#                print error
                
                p.w_old = p.w_vector
                for w in xrange(len(p.w_vector)):
                    p.w_vector[w] = p.w_old[w] + p.lrate * p.error * self.layers[-2].ptrons[w].output
                    
                p.bias += p.bias * p.lrate * p.error * 1
                
                
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
#                        print error
                        
                        p.w_old = p.w_vector
                        for w in xrange(len(p.w_vector)):
                            p.w_vector[w] = p.w_old[w] + p.lrate * p.error * self.layers[0].x_vector[w]
                            
                        p.bias += p.bias * p.lrate * p.error * 1
            