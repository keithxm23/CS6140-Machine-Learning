from math import exp

class Perceptron():
    bias = None
    output = None
    w_vector = [] #weight vector
    
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
            xvec.append(p.sigmoid(self.x_vector))
        return xvec
    
    def back_propagate(self):
        pass
    
class Nnet():
    layers = []
    
    def begin(self):
        for l in self.layers:
            xvec = l.feed_forward()