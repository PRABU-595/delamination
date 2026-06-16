import sys
import os
print("1. Starting Import Debug...")
sys.stdout.flush()

try:
    # Force CPU initialization to debug if it's a CUDA hang
    # os.environ['CUDA_VISIBLE_DEVICES'] = '' 
    print("   (Attempting standard torch import...)")
    import torch
    print(f"2. Torch Loaded. CUDA Available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Torch Failed: {e}")
sys.stdout.flush()

try:
    from pathlib import Path
    print("3. Pathlib Loaded.")
except Exception as e:
    print(f"❌ Pathlib Failed: {e}")

try:
    print("4. Attempting Loader Import...")
    from src.data.multimodal_loader import get_mega_loader
    print("5. Loader Imported Successfully.")
except Exception as e:
    print(f"❌ Loader Import FAILED: {e}")

try:
    print("6. Attempting Framework Import...")
    from src.models.integrated.framework import IntegratedDelaminationFramework
    print("7. Framework Imported Successfully.")
except Exception as e:
    print(f"❌ Framework Import FAILED: {e}")

print("8. Import Debug Complete. If you see this, basic imports are fine.")
