import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.al_vtfd.virtual_testing import VirtualTestingFramework
from src.models.integrated.framework import IntegratedFramework

def main():
    print("Initializing Integrated Delamination Framework...")
    
    # Initialize framework with default config
    model = IntegratedFramework()
    
    # Initialize Virtual Testing Environment
    al_framework = VirtualTestingFramework(model)
    
    print("\n--- Starting Active Learning Optimization ---")
    print("Material System: CFRP_Standard_T800")
    print("Objective: Characterize Delamination Migration & Growth with Minimal Tests")
    
    # Create a candidate pool of test configurations [batch, features]
    # Features: [G_I, G_II, Temperature, Rate, PlyAngle1, PlyAngle2]
    candidate_pool = torch.rand(100, 6) 
    
    # Run active learning
    results = al_framework.run_optimization(
        material_system='CFRP_Standard',
        budget_tests=15, # Small budget for demo
        candidate_pool=candidate_pool
    )
    
    print("\n--- Optimization Complete ---")
    print(f"Tests conducted: {results['n_tests_conducted']}")
    print(f"Final Accuracy (Simulated R²): {results['final_accuracy']:.4f}")
    print(f"Cost savings: {results['cost_savings']*100:.1f}%")
    
    # Save dummy results
    os.makedirs('results', exist_ok=True)
    with open('results/al_demo_results.txt', 'w') as f:
        f.write(str(results))
    print("Results saved to results/al_demo_results.txt")

if __name__ == '__main__':
    main()
