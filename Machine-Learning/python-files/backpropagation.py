import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# RANDOM WEIGHTS
w1 = 2 * np.random.random((3, 4)) - 1
w2 = 2 * np.random.random((4, 1)) - 1

num_iterations = 60000

for i in range(num_iterations):

    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, w1))
    layer_2 = sigmoid(np.dot(layer_1, w2))

    # Calculate error
    layer_2_error = y - layer_2

    # Backpropagation
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(w1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights
    w1 += layer_1.T.dot(layer_2_delta)
    w2 += layer_0.T.dot(layer_1_delta)

print("Output after training:")
print(layer_2)
