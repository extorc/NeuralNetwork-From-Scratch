import sys,os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from nn import *

network = Network([784,20,10],get_mnist,0.01,1)
network.train()
network.save_model('model.npy')
