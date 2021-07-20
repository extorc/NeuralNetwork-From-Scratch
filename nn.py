from data import get_mnist
import numpy as np
import time
from functions import *
from numpy import asarray , save
from layer import Layer

images, labels = get_mnist()
input_layer = Layer(np.random.uniform(-0.5, 0.5, (20, 784)),np.zeros((20, 1)))
hidden_layer1 = Layer(np.random.uniform(-0.5, 0.5, (10, 20)),np.zeros((10, 1)))
layers = [input_layer,hidden_layer1]
learn_rate = 0.01
nr_correct = 0
epochs = 1
p_done = 0
save_model_data = []

for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)

        h = sigmoid(input_layer.forward_propogat(img))
        o = sigmoid(hidden_layer1.forward_propogat(h))

        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        delta_o = o - l
        hidden_layer1.back_propogat(learn_rate, delta_o , h)

        delta_h = np.transpose(l2_weights) @ delta_o * (h * (1 - h))
        input_layer.back_propogat(learn_rate, delta_h , img)

        if p_done % 100 == 0:
            print(f"Accuracy : {round((nr_correct / images.shape[0]) * 100, 2)} , Images : {p_done}" ,end = "\r")
            time.sleep(0.001)
        p_done += 1
    p_done = 0

    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

for l in layers:
    save_model_data.append(l.weight)
    save_model_data.append(l.bias)

data = asarray(save_model_data)

save('model.npy',data)
