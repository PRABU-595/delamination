"""
Active Learning Training Entry Point
------------------------------------
Executes the Active Learning-Guided Virtual Testing Framework (AL-VTFD) loop.
"""
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.al_vtfd.virtual_testing import VirtualTestingFramework
from src.models.integrated.framework import IntegratedDelaminationFramework

def main():
    print("================================================================")
    print("   Active Learning-Guided Virtual Testing Framework (AL-VTFD)   ")
    print("================================================================")
    
    # 1. Initialize Integrated Model (SNPI + CAD-Former)
    # Using default config for demo
    model = IntegratedDelaminationFramework({
        'snpi_net': {'input_dim': 6}, # Example config
        'cad_former': {'d_model': 256}
    })
    
    # 2. Initialize AL Framework
    framework = VirtualTestingFramework(model)
    
    # 3. Run Optimization
    # We use a dummy budget for demonstration
    results = framework.run_optimization(
        material_system="IM7/8552 Carbon/Epoxy",
        budget_tests=15
    )
    
    print("\noptimization Complete!")
    print(f"Tests Conducted: {results['n_tests_conducted']}")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"Cost Savings: {results['cost_savings']*100:.1f}%")

if __name__ == "__main__":
    main()
