from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import *
from numpy import save , load

dataset, labels = get_mnist()

data = load('model.npy',allow_pickle = True)
model = []

w1 = data[0][0]
b1 = data[0][1]
w2 = data[1][0]
b2 = data[1][1]

while True:
    index = input("Enter a number (0 - 59999): ")
    if not index == "exit":
        dts = dataset[int(index)]
    else :
        break
    dts.shape += (1,)

    h = sigmoid(-forward_propogate(b1,w1,dts.reshape(784,1)))
    o = sigmoid(-forward_propogate(b2,w2,h))

    mode = input("Display Mode : ")
    if mode == "plot":
        plt.plot(o)
        plt.show()
    elif mode == "argmax":
        print(o.argmax())
    plt.imshow(dts.reshape(28, 28), cmap="Greys")

    plt.show()
