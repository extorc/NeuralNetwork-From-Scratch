import numpy as np

class Layer:
    def __init__(self,w,b):
        self.weight = w
        self.bias = b

    def forward_propogat(self,neuron):
        return self.bias + (self.weight @ neuron)

    def back_propogat(self,rate, delta, neuron):
        self.weight += -rate * delta @ np.transpose(neuron)
        self.bias += -rate * delta
