from math import exp
from utils import MSE, is_converged, ABSE
from random import uniform as rand

class Perceptron():#TODO add compulsory vars as keyword args
    bias = None
    output = None
    w_vector = [] #weight vector
    w_old = []
    error = None
    lrate = 0.05#None TODO randomize this
    
    def weighted_sum(self, x_vector):
        wsum = 0
        for x, w in zip(x_vector, self.w_vector):
            wsum += w*x
        #wsum+= self.bias
        return wsum
    
    def sigmoid(self, x_vector): #activation function
        return 1/(1+exp(self.weighted_sum(x_vector)*-1))
        
    
class Layer():
    ptrons = []
    x_vector = [] # array of input vector
    
    def __init__(self, ptron_biases, below_layer=None, x_vector=None, w_vector=None):
        self.ptrons = []
        for i, b in enumerate(ptron_biases):
            p = Perceptron()
            p.bias = b
            if below_layer == None:
                if w_vector == None:
                    p.w_vector = [rand(-0.05,0.05) for r in xrange(len(x_vector[0]))]
                else:
                    p.w_vector = w_vector[i]
                p.w_old = p.w_vector
            else:
                if w_vector == None:
                    p.w_vector = [rand(-0.05,0.05) for r in xrange(len(below_layer.ptrons))]
                else:
                    p.w_vector = w_vector[i]
                p.w_old = p.w_vector
            self.ptrons.append(p)
        
        if x_vector != None:
            self.x_vector = x_vector
        else:
            self.x_vector = [[] for r in xrange(8)] #FIXME: pull number of inputs from Nnet()
    
    
    
    def feed_forward(self, input_index):
        xvec = []
        for p in self.ptrons:
            p.output = p.sigmoid(self.x_vector[input_index])
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
            
            #p.bias += p.bias * p.lrate * p.error * 1


class Nnet():
    layers = []
    labels = [[]]
    
    def begin(self, thresh):
        prev_error_arr = [0 for r in xrange(len(self.layers[0].x_vector))]
        curr_error_arr = [float("inf") for r in xrange(len(self.layers[0].x_vector))]
        run_number = 1
        error_arr = []
        while(not is_converged(prev_error_arr, curr_error_arr, thresh)):
            
            print "Run: %s\t Errors: %s" % (run_number, curr_error_arr)
            run_number+=1
            #Feed forward
            for input_index in xrange(len(self.layers[0].x_vector)):#looping over each of the inputs
                
                for i in xrange(len(self.layers)):
                    xvec = self.layers[i].feed_forward(input_index)
                    try:
                        self.layers[i+1].x_vector[input_index]= xvec
                    except IndexError:
                        break #reached output layer
                    
                #Check if error has converged
                prev_error_arr[input_index] = curr_error_arr[input_index]
                curr_error_arr[input_index] = MSE(xvec,self.labels[input_index])
                
                
                #Now Back propagate
                pass
                for p in self.layers[-1].ptrons: #updating errors of output layer perceptrons
                    error = p.output
                    error *= (1 - p.output)
                    index = self.layers[-1].ptrons.index(p)
                    error *= (self.labels[input_index][index] - p.output)
                    p.error = error
    #                print error
                    
                    p.w_old = p.w_vector
                    for w in xrange(len(p.w_vector)):
                        p.w_vector[w] = p.w_old[w] + p.lrate * p.error * self.layers[-2].ptrons[w].output
                        
                    #p.bias += p.bias * p.lrate * p.error * 1
                    
                    
                for i in xrange(len(self.layers)-2, -1, -1): #looping from index of output_layer-1 to 0th layer
                    if i > 0:#updating errors of hidden layer perceptrons
                        self.layers[i].back_propagate(self.layers[i+1], self.layers[i-1])
                    else:#updating error for (i=)0-th layer (last hidden layer)
                        for j, p in enumerate(self.layers[0].ptrons):
                            error = p.output
                            error *= (1 - p.output)
                            err_j = 0
                            for q in self.layers[1].ptrons:
                                #looping over above layer's perceptrons
                                
                                err_j += q.error*q.w_vector[j]#problematic
                            error *= err_j
                            p.error = error
    #                        print error
                            
                            p.w_old = p.w_vector
                            for w in xrange(len(p.w_vector)):
                                p.w_vector[w] = p.w_old[w] + p.lrate * p.error * self.layers[0].x_vector[input_index][w]
                                
                            #p.bias += p.bias * p.lrate * p.error * 1
            