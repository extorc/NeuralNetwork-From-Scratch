import numpy as np

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

def forward_propogate(bias,weight,neuron):
    return bias + (weight @ neuron)

def back_propogate(weight,bias,rate, delta, neuron):
    weight += -rate * delta @ np.transpose(neuron)
    bias += -rate * delta
