from data import get_mnist
import numpy as np
import time
from functions import *
from numpy import asarray , save
from layer import Layer

dataset, labels = get_mnist()
input_layer = Layer(np.random.uniform(-0.5, 0.5, (20, 784)),np.zeros((20, 1)))
hidden_layer1 = Layer(np.random.uniform(-0.5, 0.5, (10, 20)),np.zeros((10, 1)))
layers = [input_layer,hidden_layer1]

learn_rate = 0.01
nr_correct = 0
epochs = 1
p_done = 0
save_model_data = []

for epoch in range(epochs):
    for dts, l in zip(dataset, labels):
        dts.shape += (1,)
        l.shape += (1,)

        h = sigmoid(input_layer.forward_propogate(dts))
        o = sigmoid(hidden_layer1.forward_propogate(h))

        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        delta_o = o - l
        hidden_layer1.back_propogate(learn_rate, delta_o , h)

        delta_h = layer_error(hidden_layer1,delta_o,h)
        input_layer.back_propogate(learn_rate, delta_h , dts)

        if p_done % 100 == 0:
            print(f"Accuracy : {round((nr_correct / dataset.shape[0]) * 100, 2)} , dataset : {p_done}" ,end = "\r")
            time.sleep(0.001)
        p_done += 1
    p_done = 0

    print(f"Acc: {round((nr_correct / dataset.shape[0]) * 100, 2)}%")
    nr_correct = 0

for l in layers:
    save_model_data.append([l.weight,l.bias])

data = asarray(save_model_data)

save('model.npy',data)
