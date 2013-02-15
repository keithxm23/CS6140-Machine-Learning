from math import exp

class Perceptron():
    
    bias = None
    x_vector = [] #input vector
    w_vector = [] #weight vector
    output = None
    
    def weighted_sum(self):
        wsum = 0
        for x, w in zip(self.x_vector, self.w_vector):
            wsum += w*x
        wsum+= self.bias
        return wsum
    
    def sigmoid(self): #activation function
        return 1/(1+exp(self.weighted_sum()*-1))

p = Perceptron()
p.x_vector = [-3,1]
p.w_vector = [1,1]
p.bias = 1
print p.sigmoid()
    