import sys
import torch
import numpy
import scipy
import pandas
import matplotlib
import sklearn

print("Environment Validation Successful!")
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"Numpy: {numpy.__version__}")
print(f"Scipy: {scipy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
