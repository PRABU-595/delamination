"""
Delamination Framework - Interactive Prediction Script
------------------------------------------------------
Allows exploring predictions from the championship 'mega_best_model.pt'.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import json
from src.models.integrated.framework import IntegratedDelaminationFramework

# Championship Configuration
MODEL_PATH = PROJECT_ROOT / "src" / "training" / "checkpoints" / "mega_run" / "mega_best_model.pt"
CONFIG = {
    'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
    'cad_former': {'d_model': 128, 'n_layers': 4},
    'al_vtfd': {}
}

def load_model():
    model = IntegratedDelaminationFramework(CONFIG)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f"[INFO] Loaded weights from {MODEL_PATH}")
    else:
        print("[WARNING] No weights found. Using uninitialized model.")
    model.eval()
    return model

def predict_interactive():
    print("=" * 60)
    print("DELAMINATION FRAMEWORK - INTERACTIVE INFERENCE")
    print("=" * 60)
    
    model = load_model()
    
    # Example input generation (Laminate + Loading + Physics)
    # n_steps = 1, n_interfaces = 8
    laminate = torch.randn(1, 8, 64)
    loading = torch.randn(1, 100)
    # Typical Physics [Density, Thick, Plies, Mod, Shear, Yield]
    physics = torch.tensor([[1525.0, 3.0, 16.0, 70.0, 25.0, 0.01]])
    
    print("\n[STEP 1] Running prediction for sample Case...")
    with torch.no_grad():
        outputs = model.predict_delamination(
            laminate, loading, 
            physics_inputs=physics
        )
        
    print("\n[RESULTS]")
    print(f"  Predicted Area: {outputs['delamination_area'].mean().item():.4f}")
    print(f"  Growth Rate:    {outputs['growth_rate'].mean().item():.4f}")
    print(f"  Confidence (Uncertainty): {outputs['uncertainty'].mean().item():.4f}")
    
    migration_probs = outputs['migration_interface'].squeeze().numpy()
    max_interface = np.argmax(migration_probs)
    print(f"  Most Likely Migration Interface: {max_interface} (Prob: {migration_probs[max_interface]:.2%})")
    
    print("\n" + "=" * 60)
    print("Inference Successful.")

if __name__ == "__main__":
    predict_interactive()
