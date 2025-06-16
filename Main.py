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