import scipy.io
import sys
import numpy as np

file_path = r"C:\Users\iampr\Desktop\MY PAPERS\delamination-ml-project\data\raw\NASA_CFRP\2. Composites\Layup1\L1_S11_F\PZT-data\L1S11_0_0.mat"

try:
    mat = scipy.io.loadmat(file_path)
    print("Keys found:", mat.keys())
    
    for k in mat:
        if k.startswith('__'): continue
        val = mat[k]
        print(f"Key: {k}, Type: {type(val)}, Shape: {np.shape(val)}")
        if isinstance(val, np.ndarray):
            print(f"Proto-content (first 5): {val.flatten()[:5]}")
            print(f"Data Type: {val.dtype}")

except Exception as e:
    print(f"Error: {e}")
