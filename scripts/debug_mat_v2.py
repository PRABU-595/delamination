import scipy.io
import sys
import numpy as np

file_path = r"C:\Users\iampr\Desktop\MY PAPERS\delamination-ml-project\data\raw\NASA_CFRP\2. Composites\Layup1\L1_S11_F\PZT-data\L1S11_0_0.mat"

def inspect_element(key, elem, level=0):
    indent = "  " * level
    print(f"{indent}Key: {key}, Type: {type(elem)}")
    
    if isinstance(elem, np.ndarray):
        print(f"{indent}  Shape: {elem.shape}, Dtype: {elem.dtype}")
        # If it's a structured array (void type), iterate fields
        if elem.dtype.names:
            print(f"{indent}  Fields: {elem.dtype.names}")
            # Recursively inspect fields of the first element if array is non-empty
            if elem.size > 0:
                for field in elem.dtype.names:
                    val = elem[0][field]
                    inspect_element(field, val, level+1)
        # If it's a standard array, print stats
        elif np.issubdtype(elem.dtype, np.number):
            print(f"{indent}  Min: {np.min(elem)}, Max: {np.max(elem)}")
        # If object array, inspect first element
        elif elem.dtype == 'O' and elem.size > 0:
            inspect_element(f"{key}[0]", elem.flat[0], level+1)

try:
    mat = scipy.io.loadmat(file_path)
    print("Top-level keys:", mat.keys())
    
    for k in mat:
        if k.startswith('__'): continue
        inspect_element(k, mat[k])

except Exception as e:
    print(f"Error: {e}")
