from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import time

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. l1_weights = weights from input layer to hidden layer
"""

images, labels = get_mnist()
l1_weights = np.random.uniform(-0.5, 0.5, (20, 784))
l2_weights = np.random.uniform(-0.5, 0.5, (10, 20))
l1_bias = np.zeros((20, 1))
l2_bias = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3
p_done = 0

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

def forward_propogate(bias,weight,neuron):
    return bias + (weight @ neuron)

def back_propogate(weight,bias,rate, delta, neuron):
    weight += -rate * delta @ np.transpose(neuron)
    bias += -rate * delta

for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h = sigmoid(forward_propogate(l1_bias,l1_weights,img))

        # Forward propagation hidden -> output
        o = sigmoid(forward_propogate(l2_bias,l2_weights,h))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        delta_o = o - l
        back_propogate(l2_weights,l2_bias,learn_rate ,delta_o ,h)

        delta_h = np.transpose(l2_weights) @ delta_o * (h * (1 - h))
        back_propogate(l1_weights,l1_bias,learn_rate, delta_h , img)

        if p_done % 100 == 0:
            print(f"Accuracy : {round((nr_correct / images.shape[0]) * 100, 2)} , Images : {p_done}" ,end = "\r")
            time.sleep(0.001)
        p_done += 1
    p_done = 0
    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results

while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = l1_bias + l1_weights @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = l2_bias + l2_weights @ h
    o = 1 / (1 + np.exp(-o_pre))

    print(o.argmax())
    plt.show()
