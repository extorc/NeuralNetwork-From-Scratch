from data import get_mnist
import numpy as np
import time
from functions import *
from numpy import asarray , save
from layer import Layer

class Network:
    def __init__(self,shape,inp,rate,ep):
        self.inp = inp
        self.dataset,self.labels = inp()
        # dataset, labels = get_mnist()
        self.input_layer = Layer(np.random.uniform(-0.5, 0.5, (shape[1], shape[0])),np.zeros((shape[1], 1)))
        self.hidden_layer1 = Layer(np.random.uniform(-0.5, 0.5, (shape[2], shape[1])),np.zeros((shape[2], 1)))
        self.layers = [self.input_layer,self.hidden_layer1]
        self.learn_rate = rate
        self.nr_correct = 0
        self.epochs = ep
        self.p_done = 0
        self.save_model_data = []
    def train(self):
        for epoch in range(self.epochs):
            for dts, l in zip(self.dataset, self.labels):
                dts.shape += (1,)
                l.shape += (1,)

                h = sigmoid(self.input_layer.forward_propogate(dts))
                o = sigmoid(self.hidden_layer1.forward_propogate(h))

                e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
                self.nr_correct += int(np.argmax(o) == np.argmax(l))

                self.delta_o = o - l
                self.hidden_layer1.back_propogate(self.learn_rate, self.delta_o , h)

                self.delta_h = layer_error(self.hidden_layer1,self.delta_o,h)
                self.input_layer.back_propogate(self.learn_rate, self.delta_h , dts)

                if self.p_done % 100 == 0:
                    print(f"Accuracy : {round((self.nr_correct / self.dataset.shape[0]) * 100, 2)} , dataset : {self.p_done}" ,end = "\r")
                    time.sleep(0.001)
                self.p_done += 1
            self.p_done = 0

            print(f"Acc: {round((self.nr_correct / self.dataset.shape[0]) * 100, 2)}%")
            self.nr_correct = 0

        for l in self.layers:
            self.save_model_data.append([l.weight,l.bias])

        data = asarray(self.save_model_data)

        save('model.npy',data)
