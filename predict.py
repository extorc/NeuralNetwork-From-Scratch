from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from numpy import save , load
from layer import *

dataset, labels = get_mnist()

data = load('model.npy',allow_pickle = True)
model = []

for d in data:
    model.append(Layer(d[0],d[1]))

while True:
    index = input("Enter a number (0 - 59999): ")
    if not index == "exit":
        dts = dataset[int(index)]
    else :
        break
    dts.shape += (1,)
    print(np.shape(dts))
    h = sigmoid(-model[0].forward_propogate(dts))
    o = sigmoid(-model[1].forward_propogate(h))

    mode = input("Display Mode : ")
    if mode == "plot":
        plt.plot(o)
        plt.show()
    elif mode == "argmax":
        print(o.argmax())

    plt.imshow(dts.reshape(28, 28), cmap="Greys")
    plt.show()
