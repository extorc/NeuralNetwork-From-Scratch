from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import *
from numpy import save , load

images, labels = get_mnist()

data = load('model.npy',allow_pickle = True)
w1 = data[0]
b1 = data[1]
w2 = data[2]
b2 = data[3]

while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    img.shape += (1,)

    h = sigmoid(-forward_propogate(b1,w1,img.reshape(784,1)))
    o = sigmoid(-forward_propogate(b2,w2,h))
    print(o.argmax())
    time.sleep(5)
    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.show()
