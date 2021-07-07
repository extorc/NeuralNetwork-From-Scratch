from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

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

for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = l1_bias + l1_weights @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = l2_bias + l2_weights @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        l2_weights += -learn_rate * delta_o @ np.transpose(h)
        l2_bias += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(l2_weights) @ delta_o * (h * (1 - h))
        l1_weights += -learn_rate * delta_h @ np.transpose(img)
        l1_bias += -learn_rate * delta_h

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
