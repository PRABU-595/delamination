import torch
import numpy as np
from src.models.integrated.framework import IntegratedDelaminationFramework

# Diagnostic Script to check model sensitivity
MODEL_PATH = "src/training/checkpoints/mega_run/mega_best_model.pt"
CONFIG = {
    'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
    'cad_former': {'d_model': 128, 'n_layers': 4},
    'al_vtfd': {}
}

def check_sensitivity():
    model = IntegratedDelaminationFramework(CONFIG)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print("Model loaded.")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    model.eval()
    
    # Test cases: [Energy, Rate, Thickness, Humidity, Temp, Noise]
    test_cases = [
        ("Nominal", [1500.0, 1.0, 3.2, 45.0, 25.0, 0.01]),
        ("High Energy (Safe)", [3000.0, 0.1, 5.0, 20.0, 20.0, 0.01]),
        ("Low Energy (Critical?)", [500.0, 20.0, 1.0, 95.0, 180.0, 0.01]),
        ("Mid Stress", [1000.0, 10.0, 2.0, 60.0, 50.0, 0.01])
    ]

    for name, p in test_cases:
        physics = torch.tensor([p])
        history = torch.zeros(1, 100)
        laminate = torch.zeros(1, 8, 64)
        
        with torch.no_grad():
            out = model.predict_delamination(laminate, history, physics_inputs=physics)
        
        area = out['delamination_area'].item()
        print(f"{name} -> Area: {area:.4f} (Status: {'RED' if area > 0.7 else 'YELLOW' if area > 0.3 else 'GREEN'})")

if __name__ == "__main__":
    check_sensitivity()
