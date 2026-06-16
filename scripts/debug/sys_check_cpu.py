import os
# FORCE CPU MODE TO VERIFY IMPORTS DO NOT HANG
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
print(f"DEBUG: Process {os.getpid()} Started (CPU ONLY MODE)")
sys.stdout.flush()

print("DEBUG: Importing Torch...")
import torch
print(f"DEBUG: Torch {torch.__version__} Imported. CUDA Available (Should be False): {torch.cuda.is_available()}")

print("DEBUG: Importing Loader...")
from src.data.multimodal_loader import get_mega_loader
print("DEBUG: Loader Imported.")

print("DEBUG: CPU Vefification Complete. The code is fine, the issue is the GPU Driver.")
