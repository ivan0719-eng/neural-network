import numpy as np
import pandas as pd
import os
import random
from matplotlib import pyplot as plt
from mnist_dataloader import MnistDataloader


data_dir = "data"

training_images_filepath = os.path.join(data_dir, "train-images-idx3-ubyte", "train-images-idx3-ubyte")
training_labels_filepath = os.path.join(data_dir, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")
test_images_filepath = os.path.join(data_dir, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte")
test_labels_filepath = os.path.join(data_dir, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")


# Load MNIST dataset
mnist = MnistDataloader(training_images_filepath, training_labels_filepath,
                        test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert list of 28x28 arrays to a 2D NumPy array: (num_samples, 784)
x_train_flat = np.array([img.flatten() for img in x_train])  # shape: (m, 784)

# Combine with labels if needed
mnist_data = np.column_stack((y_train, x_train_flat))  # shape: (m, 785)

# Get shape
m, n = mnist_data.shape
print(f"Rows (m): {m}, Columns (n): {n}")

# Shuffle rows
np.random.shuffle(mnist_data)

data_dev = mnist_data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = mnist_data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1)- 0.5
    W2 = np.random.rand(10,10)- 0.5
    b2 = np.random.rand(10,1)- 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z , 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2) # our predictions
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1 # create an array that rnage from 0 to m 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W2, X , Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 /  m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y ) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1 , A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
    

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)