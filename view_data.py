import numpy as np
import os

# Load the NPZ file
file_path = os.path.join('numpy_matrix_data', 'ordered_data.npy')
data = np.load(file_path, allow_pickle=True)

# Print the contents of the NPZ file
print("Keys in the NPZ file:", data.files)

# Access and print individual arrays
for key in data.files:
    print("Data in", key, ":", data[key])
