import os

file_path = "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte"

print("Exists:", os.path.exists(file_path))
print("Is file:", os.path.isfile(file_path))
print("Is dir:", os.path.isdir(file_path))